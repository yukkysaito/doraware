## Vehicle chassis: RigidBody3D (Jolt) with origin at the rear axle center.
## See docs/architecture.md §8. Planar: linear Y and angular X/Z locked, no
## gravity (static axle loads are computed analytically). Each physics step
## `physics_step()` advances the actuators, gear logic and signal lamps and
## applies the net tire force/torque at the center of gravity; Jolt integrates
## the motion. Signal lamps (turn indicators, hazard lights) never affect motion.
## The pose/velocity are written only by `setup()` (reset).
extends RigidBody3D

const Actuator = preload("res://scripts/sim/actuator.gd")
const Tire = preload("res://scripts/sim/tire.gd")

enum { GEAR_NONE = 0, GEAR_PARK = 1, GEAR_REVERSE = 2, GEAR_NEUTRAL = 3, GEAR_DRIVE = 4 }
enum { TURN_OFF = 0, TURN_LEFT = 1, TURN_RIGHT = 2 }
const G := 9.81

var p: Dictionary = {}                 # VehicleParameters
var act := Actuator.new()
var dt := 0.01
var gear := GEAR_PARK
var shift_target := -1                 # gear being engaged, -1 = none
var shift_timer := 0.0
var gear_refusals := 0                 # cumulative refused gear requests
var wheel_angles := [0.0, 0.0]         # actual left / right front wheel angles
var wheel_forces := []                 # last [f_lon, f_lat] per wheel (rl, rr, fl, fr)
var turn_indicator := TURN_OFF         # latched signal command (architecture §7)
var hazard_lights := false
var left_lamp_on := false              # lamp outputs during the last physics step
var right_lamp_on := false

var _shift_time := 0.3
var _shift_max_speed := 0.05
var _rolling := 0.0
var _wheels := []                      # static wheel data (px, py rel. CoG, fz)
var _v_before := [0.0, 0.0]            # rear-axle world velocity (Godot X, Z) before the last step
var _flash_half_period := 1.0 / 3.0    # [s] flasher on (and off) time
var _flash_steps := 0                  # physics steps since the active signal function changed


func setup(params: Dictionary, actuator_cfg: Dictionary, step_dt: float, seed: int,
		x: float, z: float, rot_y: float, initial_gear: int) -> void:
	p = params
	dt = step_dt
	var L: float = p.wheel_base
	var T: float = p.wheel_tread
	var cog: float = p.cog_x
	var m: float = p.mass
	# body
	mass = m
	gravity_scale = 0.0
	linear_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
	angular_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
	linear_damp = 0.0
	angular_damp = 0.0
	can_sleep = false
	center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
	center_of_mass = Vector3(cog, 0.0, 0.0)
	var iz: float = p.yaw_inertia
	inertia = Vector3(iz, iz, iz)       # only the yaw (Y) component is free
	axis_lock_linear_y = true
	axis_lock_angular_x = true
	axis_lock_angular_z = true
	contact_monitor = true
	max_contacts_reported = 16
	collision_layer = 2
	collision_mask = 1
	transform = Transform3D(Basis(Vector3.UP, rot_y), Vector3(x, 0.0, z))
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	# footprint collision box relative to the rear axle
	var length: float = p.front_overhang + L + p.rear_overhang
	var width: float = T + p.left_overhang + p.right_overhang
	var height: float = p.vehicle_height
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(length, height, width)
	shape.shape = box
	var cx: float = (L + p.front_overhang - p.rear_overhang) / 2.0
	var cy: float = (p.left_overhang - p.right_overhang) / 2.0   # +left (map)
	shape.position = Vector3(cx, 0.05 + height / 2.0, -cy)
	add_child(shape)
	# static axle loads and wheel positions relative to the CoG (x fwd, y left)
	var fz_front := m * G * cog / L / 2.0
	var fz_rear := m * G * (L - cog) / L / 2.0
	_wheels = [
		{"px": -cog, "py": T / 2.0, "fz": fz_rear},       # rear left
		{"px": -cog, "py": -T / 2.0, "fz": fz_rear},      # rear right
		{"px": L - cog, "py": T / 2.0, "fz": fz_front},   # front left
		{"px": L - cog, "py": -T / 2.0, "fz": fz_front},  # front right
	]
	# actuators / powertrain
	act.configure(actuator_cfg, p.max_steer_angle, dt, seed)
	_shift_time = float(actuator_cfg.get("gear_shift_time", 0.3))
	_shift_max_speed = float(actuator_cfg.get("gear_shift_max_speed", 0.05))
	var rr = actuator_cfg.get("rolling_resistance", 0.0)
	_rolling = 0.0 if rr == null else float(rr)
	var ff = actuator_cfg.get("flasher_frequency", 1.5)
	_flash_half_period = 0.5 / (1.5 if ff == null else float(ff))
	turn_indicator = TURN_OFF
	hazard_lights = false
	left_lamp_on = false
	right_lamp_on = false
	_flash_steps = 0
	gear = initial_gear
	shift_target = -1
	shift_timer = 0.0
	gear_refusals = 0
	wheel_angles = [0.0, 0.0]
	_v_before = [0.0, 0.0]


static func gear_direction(g: int) -> int:
	return 1 if g == GEAR_DRIVE else (-1 if g == GEAR_REVERSE else 0)


## Rear-axle-point velocity in the world (Godot X, Z), body yaw, yaw rate, CoG
## velocity in the vehicle frame (x fwd, y left).
func kinematics() -> Dictionary:
	var st := PhysicsServer3D.body_get_direct_state(get_rid())
	var xf := st.transform
	var yaw := atan2(-float(xf.basis.x.z), float(xf.basis.x.x))
	var c := cos(yaw)
	var s := sin(yaw)
	var v := st.linear_velocity           # CoG velocity (world)
	var w := float(st.angular_velocity.y)
	var vx := float(v.x)
	var vz := float(v.z)
	# r_ra - r_cog = R * (-cog_x, 0, 0) = (-cog*c, 0, cog*s); omega x r (omega along Y)
	var cog: float = p.cog_x
	var rx := -cog * c
	var rz := cog * s
	var ra_x := vx + w * rz
	var ra_z := vz - w * rx
	# CoG velocity in map-like vehicle axes: forward = (c, -s) in Godot XZ, left = (-s, -c)
	var ux := vx * c - vz * s
	var uy := -vx * s - vz * c
	return {"x": float(xf.origin.x), "z": float(xf.origin.z), "yaw": yaw, "c": c, "s": s,
		"ra_x": ra_x, "ra_z": ra_z, "w": w, "ux": ux, "uy": uy}


func physics_step(cmd: Dictionary) -> void:
	var k := kinematics()
	_v_before = [k.ra_x, k.ra_z]
	var speed := sqrt(k.ra_x * k.ra_x + k.ra_z * k.ra_z)
	var emergency := bool(cmd.get("emergency", false))
	_update_gear(int(cmd.get("gear", GEAR_NONE)), speed)
	if cmd.has("signals"):
		_latch_signals(cmd.signals)            # idempotent: repeating it for a chunk changes nothing
	_update_lamps()
	var shifting := shift_target >= 0
	var d := 0 if shifting else gear_direction(gear)
	var v_lon: float = k.ra_x * k.c - k.ra_z * k.s
	act.update(float(cmd.get("steer", 0.0)), float(cmd.get("accel", 0.0)), d, emergency, v_lon)

	# Ackermann front wheel angles from the bicycle angle
	var L: float = p.wheel_base
	var T: float = p.wheel_tread
	var delta := act.steer
	if absf(delta) < 1e-9:
		wheel_angles = [0.0, 0.0]
	else:
		var r := L / tan(delta)
		wheel_angles = [atan(L / (r - T / 2.0)), atan(L / (r + T / 2.0))]

	# powertrain / brake: drive force (signed along the wheel) and brake capacity [N]
	var a := act.accel
	var drive_acc := 0.0
	var brake_acc := 0.0
	var hold := 0.0
	if shifting or gear == GEAR_PARK:
		hold = p.tire_friction * G               # parking lock: up to the friction limit
		brake_acc = absf(a) if emergency else 0.0
	elif emergency or d == 0 or a * d <= 0.0:
		brake_acc = absf(a)                      # against the gear direction (or neutral) = brake
	else:
		drive_acc = a                            # propulsion in the gear direction
	var wheels := []
	for i in 4:
		var w: Dictionary = _wheels[i]
		var share: float = w.fz / G              # mass share of this wheel
		var brake: float = share * (brake_acc + hold) + _rolling * w.fz
		wheels.append({"px": w.px, "py": w.py, "fz": w.fz,
			"angle": wheel_angles[i - 2] if i >= 2 else 0.0,
			"drive": share * drive_acc, "brake": brake})
	var res := Tire.solve([k.ux, k.uy, k.w], wheels, p.mass, p.yaw_inertia,
		p.tire_friction, p.cornering_stiffness, dt)
	wheel_forces = res.wheel_forces
	# vehicle frame (x fwd, y left) -> Godot world
	var fwx: float = res.fx * k.c - res.fy * k.s
	var fwz: float = -res.fx * k.s - res.fy * k.c
	apply_central_force(Vector3(fwx, 0.0, fwz))
	apply_torque(Vector3(0.0, res.mz, 0.0))

	if shift_target >= 0:
		shift_timer += dt
		if shift_timer >= _shift_time - 1e-9:
			gear = shift_target
			shift_target = -1


func _update_gear(req: int, speed: float) -> void:
	if shift_target >= 0 and speed > _shift_max_speed:
		shift_target = -1                        # started moving: abort the shift
		gear_refusals += 1
	if req == GEAR_NONE or req == shift_target:
		return
	if req == gear:
		shift_target = -1                        # cancel a pending shift
		return
	if speed <= _shift_max_speed:
		shift_target = req
		shift_timer = 0.0
	else:
		gear_refusals += 1                       # refused while moving; request again later


## Latch a signal command. Invalid turn indicator values are treated as OFF
## (reported in the log). A change restarts the flasher with the lamps on.
func _latch_signals(sig: Dictionary) -> void:
	var t := int(sig.get("turn_indicator", TURN_OFF))
	if t != TURN_OFF and t != TURN_LEFT and t != TURN_RIGHT:
		push_warning("vehicle: invalid turn_indicator %d treated as OFF" % t)
		t = TURN_OFF
	var h := bool(sig.get("hazard_lights", false))
	if t != turn_indicator or h != hazard_lights:
		turn_indicator = t
		hazard_lights = h
		_flash_steps = 0


## Flasher: on for half a period, off for half a period, starting on. Hazard
## lights take priority over the turn indicator and flash both sides.
func _update_lamps() -> void:
	var left := hazard_lights or turn_indicator == TURN_LEFT
	var right := hazard_lights or turn_indicator == TURN_RIGHT
	var phase_on := int(floor(_flash_steps * dt / _flash_half_period + 1e-9)) % 2 == 0
	left_lamp_on = left and phase_on
	right_lamp_on = right and phase_on
	_flash_steps += 1


## State reply fields (architecture §7), rear-axle point, vehicle local axes.
func state_fields() -> Dictionary:
	var k := kinematics()
	var c: float = k.c
	var s: float = k.s
	# world (Godot X,Z) -> local: local X = forward (c, -s), local Z = (s, c)
	var vlx: float = k.ra_x * c - k.ra_z * s
	var vlz: float = k.ra_x * s + k.ra_z * c
	var ax: float = (k.ra_x - _v_before[0]) / dt
	var az: float = (k.ra_z - _v_before[1]) / dt
	return {
		"x": k.x, "z": k.z, "rot_y": k.yaw,
		"v_local": [vlx, 0.0, vlz],
		"ang_vel_y": k.w,
		"accel_local": [ax * c - az * s, 0.0, ax * s + az * c],
		"steer": act.steer, "accel_actual": act.accel,
		"gear": gear, "shifting": shift_target >= 0, "gear_refusals": gear_refusals,
		"turn_indicator": turn_indicator, "hazard_lights": hazard_lights,
		"left_lamp_on": left_lamp_on, "right_lamp_on": right_lamp_on,
	}
