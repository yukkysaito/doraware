## Visual car model attached to the physics body (display only): body,
## cabin, wheels (steered and spinning), headlights, tail/brake lights,
## reverse lights and turn indicators driven by the simulated vehicle state.
## Vehicle local axes: +X forward, +Y up, -Z left; origin at the rear axle.
extends Node3D

const D = preload("res://scripts/gui/draw3d.gd")

const PAINT := Color(0.16, 0.42, 0.86)
const GLASS := Color(0.06, 0.08, 0.12, 0.92)
const TRIM := Color(0.07, 0.07, 0.08)
const AMBER := Color(1.0, 0.58, 0.05)
const RED := Color(1.0, 0.08, 0.06)
const WHITE := Color(1.0, 0.98, 0.92)
const WHEEL_RADIUS := 0.33

var _front_pivots: Array[Node3D] = []
var _spinners: Array[Node3D] = []
var _spin := 0.0
var _left_mat: StandardMaterial3D
var _right_mat: StandardMaterial3D
var _brake_mat: StandardMaterial3D
var _reverse_mat: StandardMaterial3D


func build(vp: Dictionary, paint_color: Color = PAINT) -> void:
	for c in get_children():
		c.free()
	_front_pivots.clear()
	_spinners.clear()
	var L: float = vp.wheel_base
	var T: float = vp.wheel_tread
	var fo: float = vp.front_overhang
	var ro: float = vp.rear_overhang
	var W: float = T + vp.left_overhang + vp.right_overhang
	var H: float = vp.get("vehicle_height", 1.5)
	var length := fo + L + ro
	var x_front := L + fo
	var x_rear := -ro
	var zc: float = -(vp.left_overhang - vp.right_overhang) / 2.0
	var paint := StandardMaterial3D.new()
	paint.albedo_color = paint_color
	paint.metallic = 0.55
	paint.roughness = 0.32
	var glass := StandardMaterial3D.new()
	glass.albedo_color = GLASS
	glass.metallic = 0.3
	glass.roughness = 0.08
	glass.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	var trim := D.mat(TRIM)

	# body: side profile extruded across the width, wheel arches cut out
	var y0 := 0.2
	var belt := clampf(H * 0.6, 0.75, H - 0.35)
	var roof := H - 0.02
	var xw := minf(L * 0.80, x_front - 0.6)                  # windshield base
	var xb := x_rear + clampf(ro * 0.55, 0.35, 0.8)          # rear window base
	var xt1 := xw - (roof - belt) * 1.45                      # windshield top
	var xt0 := xb + (roof - belt) * 0.85                      # rear roof edge
	var body := _extrusion(PackedVector2Array([
		Vector2(x_rear + 0.06, y0), Vector2(x_front - 0.06, y0), Vector2(x_front, y0 + 0.16),
		Vector2(x_front, belt - 0.18), Vector2(x_front - 0.16, belt - 0.06), Vector2(xw, belt),
		Vector2(xb, belt), Vector2(x_rear + 0.1, belt - 0.04), Vector2(x_rear, belt - 0.2),
		Vector2(x_rear, y0 + 0.16)]), W - 0.02, zc, paint)
	for ax in [0.0, L]:
		var arch := CSGCylinder3D.new()
		arch.operation = CSGShape3D.OPERATION_SUBTRACTION
		arch.radius = WHEEL_RADIUS + 0.07
		arch.height = W + 0.4
		arch.sides = 24
		arch.rotation = Vector3(PI / 2.0, 0, 0)
		arch.position = Vector3(ax, WHEEL_RADIUS, -(W - 0.02) / 2.0)
		body.add_child(arch)
	# greenhouse: glass, roof panel and B pillars
	_extrusion(PackedVector2Array([Vector2(xb, belt - 0.01), Vector2(xw, belt - 0.01), Vector2(xt1, roof - 0.04),
		Vector2(xt0, roof - 0.04)]), W * 0.86, zc, glass)
	_extrusion(PackedVector2Array([Vector2(xt0 - 0.03, roof - 0.05), Vector2(xt1 + 0.03, roof - 0.05),
		Vector2(xt1 - 0.05, roof), Vector2(xt0 + 0.05, roof)]), W * 0.87, zc, paint)
	var xp := (xt0 + xt1) / 2.0
	_part(Vector3(0.09, roof - belt - 0.03, W * 0.875), Vector3(xp, (belt + roof) / 2.0, zc), paint)
	# bumpers and side skirts
	_part(Vector3(0.10, 0.18, W), Vector3(x_front - 0.03, y0 + 0.1, zc), trim)
	_part(Vector3(0.10, 0.18, W), Vector3(x_rear + 0.03, y0 + 0.1, zc), trim)
	for s in [-1.0, 1.0]:
		_part(Vector3(L - 2.0 * WHEEL_RADIUS - 0.2, 0.07, 0.03), Vector3(L / 2.0, y0 + 0.035, zc + s * (W / 2.0 - 0.005)), trim)

	# wheels: tyre + rim; the front pair is steered
	var tyre := D.mat(Color(0.05, 0.05, 0.055))
	var rim := StandardMaterial3D.new()
	rim.albedo_color = Color(0.72, 0.74, 0.77)
	rim.metallic = 0.8
	rim.roughness = 0.3
	for i in 4:
		var pivot := Node3D.new()
		pivot.position = Vector3(L if i >= 2 else 0.0, WHEEL_RADIUS, -T / 2.0 if i % 2 == 0 else T / 2.0)
		add_child(pivot)
		var spinner := Node3D.new()
		pivot.add_child(spinner)
		var t := _cylinder(WHEEL_RADIUS, 0.23, tyre)
		spinner.add_child(t)
		var r := _cylinder(WHEEL_RADIUS * 0.58, 0.245, rim)
		spinner.add_child(r)
		var spoke := D.box(Vector3(WHEEL_RADIUS * 1.0, 0.08, 0.25), rim)
		spinner.add_child(spoke)
		_spinners.append(spinner)
		if i >= 2:
			_front_pivots.append(pivot)

	# lamps
	var head := D.mat(WHITE, false, 2.5)
	_brake_mat = D.mat(RED, false, 0.6)
	_reverse_mat = D.mat(WHITE, false, 0.0)
	_left_mat = D.mat(AMBER, false, 0.0)
	_right_mat = D.mat(AMBER, false, 0.0)
	var lamp_y := belt - 0.16
	for s in [-1.0, 1.0]:
		var side_z: float = zc + s * (W / 2.0 - 0.30)
		_part(Vector3(0.04, 0.11, 0.36), Vector3(x_front + 0.0, lamp_y, side_z), head)
		_part(Vector3(0.04, 0.13, 0.38), Vector3(x_rear - 0.0, lamp_y, side_z), _brake_mat)
		_part(Vector3(0.04, 0.07, 0.14), Vector3(x_rear - 0.0, lamp_y - 0.13, zc + s * (W / 2.0 - 0.62)), _reverse_mat)
		var ind := _left_mat if s < 0.0 else _right_mat   # Godot local -Z is left
		var corner_z: float = zc + s * (W / 2.0 - 0.07)
		_part(Vector3(0.05, 0.08, 0.13), Vector3(x_front + 0.0, lamp_y, corner_z), ind)
		_part(Vector3(0.05, 0.08, 0.13), Vector3(x_rear - 0.0, lamp_y, corner_z), ind)
		_part(Vector3(0.14, 0.05, 0.02), Vector3(L * 0.86, lamp_y, zc + s * (W / 2.0)), ind)   # side repeater


## Called every rendered frame with the latest state reply.
func update_from(vehicle: Node, s: Dictionary, delta: float) -> void:
	if _front_pivots.size() == 2:
		_front_pivots[0].rotation.y = vehicle.wheel_angles[0]
		_front_pivots[1].rotation.y = vehicle.wheel_angles[1]
	var v: float = s.v_local[0]
	_spin = fmod(_spin - v * delta / WHEEL_RADIUS, TAU)
	for sp in _spinners:
		sp.rotation.z = _spin
	_left_mat.emission_energy_multiplier = 4.0 if vehicle.left_lamp_on else 0.0
	_left_mat.albedo_color = AMBER if vehicle.left_lamp_on else AMBER.darkened(0.55)
	_right_mat.emission_energy_multiplier = 4.0 if vehicle.right_lamp_on else 0.0
	_right_mat.albedo_color = AMBER if vehicle.right_lamp_on else AMBER.darkened(0.55)
	var gear := int(s.gear)
	var dir := 1 if gear == 4 else (-1 if gear == 2 else 0)
	var braking: bool = dir == 0 or bool(s.shifting) or float(s.accel_actual) * dir < -0.05
	_brake_mat.emission_energy_multiplier = 3.5 if braking else 0.5
	_reverse_mat.emission_energy_multiplier = 3.0 if gear == 2 else 0.0
	_reverse_mat.albedo_color = WHITE if gear == 2 else WHITE.darkened(0.6)


## Side profile (x forward, y up) extruded over `width`, centred at z = zc.
func _extrusion(profile: PackedVector2Array, width: float, zc: float, material: Material) -> CSGPolygon3D:
	var e := CSGPolygon3D.new()
	e.polygon = profile
	e.mode = CSGPolygon3D.MODE_DEPTH
	e.depth = width
	e.position = Vector3(0, 0, zc + width / 2.0)     # the polygon is extruded along -Z
	e.material = material
	add_child(e)
	return e


func _part(size: Vector3, pos: Vector3, material: Material) -> MeshInstance3D:
	var mi := D.box(size, material)
	mi.position = pos
	add_child(mi)
	return mi


static func _cylinder(radius: float, width: float, material: Material) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var c := CylinderMesh.new()
	c.top_radius = radius
	c.bottom_radius = radius
	c.height = width
	c.radial_segments = 24
	c.material = material
	mi.mesh = c
	mi.rotation = Vector3(PI / 2.0, 0, 0)    # axis along the axle (local Z)
	return mi
