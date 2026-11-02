## Simulator entry point: TCP protocol handling and deterministic lockstep.
##
## Lockstep: physics advances only on `step` requests. Each physics tick that
## has steps left applies the command and lets Jolt integrate exactly one step.
## When no steps are left, the pending `state` reply is sent and the tick waits
## for the next request:
##   * headless: blocks inside _physics_process until a request arrives,
##   * GUI: waits at most a few ms, then deactivates the physics server for the
##     tick (PhysicsServer3D.set_active(false)) so the window keeps rendering.
## Sim time = step count * dt, independent of wall clock and render rate.
extends Node3D

const Protocol = preload("res://scripts/protocol.gd")
const Vehicle = preload("res://scripts/sim/vehicle.gd")
const Gui = preload("res://scripts/gui/gui.gd")

const GUI_WAIT_USEC := 4000
const POLL_DELAY_USEC := 20

var proto := Protocol.new()
var headless := true
var vehicle: RigidBody3D = null
var gui: Node = null                   # GUI only (display; never touches physics)
var obstacles: Array[Node] = []
var dt := 0.01
var step_count := 0
var steps_left := 0
var command: Dictionary = {}
var reply_pending := false
var contacts := 0
var contact_names: Array[String] = []
var quitting := false
var _chunk_t0 := 0
var last_state: Dictionary = {}


func _ready() -> void:
	headless = DisplayServer.get_name() == "headless"
	var args := _parse_args()
	print("doraware_sim: physics engine = ", ProjectSettings.get_setting("physics/3d/physics_engine"),
		", headless = ", headless)
	if not headless:
		gui = Gui.new()
		add_child(gui)
	var host: String = args.get("host", "127.0.0.1")
	var port := int(args.get("port", "0"))
	if port <= 0 or not proto.connect_to(host, port, float(args.get("connect_timeout", "30"))):
		push_error("doraware_sim: cannot connect to %s:%d" % [host, port])
		quitting = true
		get_tree().quit(2)


func _parse_args() -> Dictionary:
	var out := {}
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--") and a.contains("="):
			var kv := a.substr(2).split("=", true, 1)
			out[kv[0]] = kv[1]
	return out


func _physics_process(_delta: float) -> void:
	if quitting:
		return
	if steps_left > 0:
		_do_step()
		return
	if reply_pending:
		_send_state()
		reply_pending = false
	var t0 := Time.get_ticks_usec()
	while true:
		var msg = proto.poll_message()
		if msg == null:
			if not proto.connected:
				print("doraware_sim: bridge disconnected, quitting")
				_quit()
				return
			if not headless and Time.get_ticks_usec() - t0 > GUI_WAIT_USEC:
				PhysicsServer3D.set_active(false)   # no physics in this tick
				return
			OS.delay_usec(POLL_DELAY_USEC)
			continue
		match str(msg.get("type", "")):
			"step":
				command = msg.get("command", {})
				steps_left = int(msg.get("n", 1))
				_chunk_t0 = Time.get_ticks_usec()
				if vehicle == null:
					push_error("doraware_sim: step before reset")
					steps_left = 0
				if steps_left <= 0:
					_send_state()
					continue
				PhysicsServer3D.set_active(true)
				_do_step()
				return
			"reset":
				_reset(msg)
				_send_state()
			"viz":
				if gui != null:
					gui.update_overlay(msg)
			"screenshot":   # GUI only: save a rendered frame from the given camera
				if gui != null:
					gui.queue_screenshot(str(msg.get("path", "")), int(msg.get("camera", -1)))
			"quit":
				_quit()
				return
			_:
				push_error("doraware_sim: unknown request %s" % str(msg.get("type")))


func _do_step() -> void:
	vehicle.physics_step(command)
	step_count += 1
	steps_left -= 1
	if steps_left == 0:
		reply_pending = true   # sent at the next tick, after Jolt integrated this step


func _process(_delta: float) -> void:
	if gui != null and vehicle != null and not last_state.is_empty():
		gui.update_vehicle(vehicle, last_state)


func _quit() -> void:
	quitting = true
	proto.close()
	get_tree().quit()


func _send_state() -> void:
	var s: Dictionary = vehicle.state_fields() if vehicle != null else {}
	s["type"] = "state"
	s["step"] = step_count
	s["time"] = step_count * dt
	s["contacts"] = contacts
	s["contact_names"] = contact_names
	s["step_wall_time"] = (Time.get_ticks_usec() - _chunk_t0) * 1e-6
	proto.send(s)
	last_state = s
	if gui != null and vehicle != null:
		gui.add_trail_point(s.x, s.z)


func _reset(msg: Dictionary) -> void:
	_chunk_t0 = Time.get_ticks_usec()
	var hz := float(msg.get("physics_hz", 100.0))
	dt = 1.0 / hz
	Engine.physics_ticks_per_second = int(round(hz))
	step_count = 0
	steps_left = 0
	reply_pending = false
	contacts = 0
	contact_names = []
	command = {}
	# rebuild the physics world from scratch (determinism across resets)
	if vehicle != null:
		vehicle.free()
		vehicle = null
	for o in obstacles:
		o.free()
	obstacles.clear()
	var scene: Dictionary = msg.get("scene", {})
	var i := 0
	for ob in scene.get("obstacles", []):
		obstacles.append(_make_obstacle(ob, i))
		i += 1
	var init: Dictionary = msg.get("initial", {})
	vehicle = Vehicle.new()
	vehicle.name = "ego"
	vehicle.setup(msg.get("vehicle", {}), msg.get("actuator", {}), dt, int(msg.get("seed", 0)),
		float(init.get("x", 0.0)), float(init.get("z", 0.0)), float(init.get("rot_y", 0.0)),
		int(init.get("gear", Vehicle.GEAR_PARK)))
	add_child(vehicle)
	vehicle.body_entered.connect(_on_body_entered)
	if gui != null:
		gui.build_scene(scene, msg.get("vehicle", {}), vehicle)


func _make_obstacle(ob: Dictionary, idx: int) -> StaticBody3D:
	var body := StaticBody3D.new()
	body.name = "obstacle_%d" % idx
	body.collision_layer = 1
	body.collision_mask = 0
	var h := float(ob.get("height", 1.5))
	var size := Vector3(float(ob.length), h, float(ob.width))
	body.transform = Transform3D(Basis(Vector3.UP, float(ob.get("rot_y", 0.0))),
		Vector3(float(ob.x), 0.0, float(ob.z)))
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = size
	shape.shape = box
	shape.position = Vector3(0.0, h / 2.0, 0.0)
	body.add_child(shape)
	body.add_to_group("obstacle")
	add_child(body)
	return body


func _on_body_entered(body: Node) -> void:
	if body.is_in_group("obstacle"):
		contacts += 1
		if not contact_names.has(String(body.name)):
			contact_names.append(String(body.name))
