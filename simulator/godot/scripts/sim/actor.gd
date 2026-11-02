## Scripted traffic participant (deterministic, kinematic). Motion model of
## SimulationScene / doraware.simulation.actors: position and yaw linearly
## interpolated between waypoints [t, x, z, rot_y] by sim time (yaw the shorter
## way round), standing at the first / last waypoint outside the script.
## main.gd moves it to the pose of the end of each physics step before Jolt
## integrates, so a state reply at time T shows the actors at T.
extends AnimatableBody3D

var actor_id := 0
var cls := 0
var size := Vector3.ONE                 # length, height, width (Godot local X, Y, Z)
var _wps: Array = []


func setup(spec: Dictionary) -> void:
	actor_id = int(spec.get("id", 0))
	cls = int(spec.get("cls", 0))
	size = Vector3(float(spec.get("length", 1.0)), float(spec.get("height", 1.5)), float(spec.get("width", 1.0)))
	_wps = spec.get("waypoints", [])
	name = "actor_%d" % actor_id
	sync_to_physics = false
	collision_layer = 1
	collision_mask = 0
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = size
	shape.shape = box
	shape.position = Vector3(0.0, size.y / 2.0, 0.0)
	add_child(shape)
	add_to_group("actor")
	move_to(0.0)


func pose_at(t: float) -> Vector3:
	## (x, z, rot_y) at sim time t
	if _wps.is_empty():
		return Vector3.ZERO
	var first: Array = _wps[0]
	var last: Array = _wps[_wps.size() - 1]
	if _wps.size() == 1 or t <= float(first[0]):
		return Vector3(float(first[1]), float(first[2]), float(first[3]))
	if t >= float(last[0]):
		return Vector3(float(last[1]), float(last[2]), float(last[3]))
	var k := 0
	while k < _wps.size() - 2 and float(_wps[k + 1][0]) <= t:
		k += 1
	var a: Array = _wps[k]
	var b: Array = _wps[k + 1]
	var s := (t - float(a[0])) / (float(b[0]) - float(a[0]))
	var da := wrapf(float(b[3]) - float(a[3]), -PI, PI)
	return Vector3(lerpf(float(a[1]), float(b[1]), s), lerpf(float(a[2]), float(b[2]), s),
		wrapf(float(a[3]) + s * da, -PI, PI))


func move_to(t: float) -> void:
	var p := pose_at(t)
	transform = Transform3D(Basis(Vector3.UP, p.z), Vector3(p.x, 0.0, p.y))
