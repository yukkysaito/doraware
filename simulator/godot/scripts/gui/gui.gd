## GUI root (created only when not headless). Composes the static world, the
## car model, the overlays (planned path, trajectory window, controller
## prediction, driven path), the cameras and the HUD. It only reads the
## simulation: nothing here changes physics, so GUI and headless runs are
## bit-identical.
extends Node

const D = preload("res://scripts/gui/draw3d.gd")
const WorldView = preload("res://scripts/gui/world_view.gd")
const VehicleView = preload("res://scripts/gui/vehicle_view.gd")
const CameraRig = preload("res://scripts/gui/camera_rig.gd")
const Hud = preload("res://scripts/gui/hud.gd")

const PATH_FWD := Color(0.35, 0.65, 1.0, 0.85)
const PATH_REV := Color(1.0, 0.62, 0.22, 0.85)
const TRAJ := Color(0.25, 0.92, 1.0, 0.75)
const PRED := Color(1.0, 0.35, 0.85, 0.95)
const TRAIL := Color(1.0, 0.85, 0.35, 0.65)
const GEAR_REVERSE := 2
## Perceived object colours by ObjectClass (interfaces/perception.py)
const CLASS_COLORS := [Color(0.80, 0.80, 0.80), Color(0.30, 0.75, 1.0), Color(0.35, 0.55, 1.0),
	Color(0.55, 0.45, 1.0), Color(1.0, 0.55, 0.85), Color(0.35, 1.0, 0.75), Color(1.0, 0.85, 0.25),
	Color(0.85, 0.65, 0.40), Color(1.0, 0.45, 0.35)]
const CLASS_NAMES := ["unknown", "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "animal",
	"static"]

var world := WorldView.new()
var car: Node3D = null      # car model, rebuilt under each new physics body
var rig := CameraRig.new()
var hud := Hud.new()
var info: Dictionary = {}

var _path_mesh := ArrayMesh.new()
var _traj_mesh := ArrayMesh.new()
var _pred_mesh := ArrayMesh.new()
var _trail_mesh := ArrayMesh.new()
var _objects_mesh := ArrayMesh.new()
var _object_labels := Node3D.new()
var _trail: Array[Vector2] = []
var _overlay_mat := D.vertex_color_mat()
var _shots: Array = []        # pending screenshots: [path, camera]
var _shot_frames := 0
var _shot_restore := -1


func _ready() -> void:
	add_child(world)
	world.add_child(_object_labels)
	for m in [_path_mesh, _trail_mesh, _traj_mesh, _pred_mesh, _objects_mesh]:
		var mi := MeshInstance3D.new()
		mi.mesh = m
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		world.add_child(mi)
	add_child(rig)
	add_child(hud)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_G:
				if world.grid != null:
					world.grid.visible = not world.grid.visible
			KEY_H:
				hud.toggle_help()


## Reset: rebuild the world and the car model for the new scene.
func build_scene(scene: Dictionary, vp: Dictionary, vehicle: Node3D) -> void:
	world.build(scene, vp, vehicle.global_transform)
	# the car model lives under the physics body, which main.gd frees on every reset:
	# build a new one for the new body
	car = VehicleView.new()
	vehicle.add_child(car)
	car.build(vp)
	_trail.clear()
	info = {}
	for m in [_path_mesh, _traj_mesh, _pred_mesh, _trail_mesh, _objects_mesh]:
		m.clear_surfaces()
	for l in _object_labels.get_children():
		l.free()
	var bounds: Array = scene.get("bounds", [-20.0, 20.0, -20.0, 20.0])
	var x0 := float(bounds[0])
	var x1 := float(bounds[1])
	var z0 := float(bounds[2])
	var z1 := float(bounds[3])
	rig.set_target(vehicle, Vector3((x0 + x1) / 2.0, 0, (z0 + z1) / 2.0), Vector2(x1 - x0, z1 - z0))
	hud.set_limits(float(vp.get("max_acceleration", 1.5)), float(vp.get("max_deceleration", 3.0)))


## `viz` request: overlays and HUD values (map data already in the Godot frame).
func update_overlay(msg: Dictionary) -> void:
	if msg.has("path"):
		_draw_path(msg.path, msg.get("path_gear", []))
	if msg.has("objects"):
		_draw_objects(msg.objects)
	_set_ribbon(_traj_mesh, msg.get("planned", []), 0.22, 0.03, TRAJ)
	_set_ribbon(_pred_mesh, msg.get("predicted", []), 0.07, 0.045, PRED)
	var scenario := str(msg.get("info", {}).get("scenario", ""))
	if scenario != "" and scenario != str(info.get("scenario", "")):
		get_window().title = "doraware · %s" % scenario
	info = msg.get("info", {})


func add_trail_point(x: float, z: float) -> void:
	var v := Vector2(x, z)
	if _trail.is_empty() or _trail[_trail.size() - 1].distance_to(v) > 0.05:
		_trail.append(v)
		_set_ribbon(_trail_mesh, _trail, 0.05, 0.02, TRAIL)


func update_vehicle(vehicle: Node, s: Dictionary) -> void:
	var dt := get_process_delta_time()
	if is_instance_valid(car) and car.get_parent() == vehicle:
		car.update_from(vehicle, s, dt)
	hud.update_from(vehicle, s, info, dt)


## Save the next rendered frame(s) from `camera` (-1 = current) to `path`.
func queue_screenshot(path: String, camera: int) -> void:
	if path != "":
		_shots.append([path, camera])


func _process(_delta: float) -> void:
	if _shots.is_empty():
		return
	var shot: Array = _shots[0]
	if _shot_frames == 0:
		_shot_restore = rig.mode
		if int(shot[1]) >= 0:
			rig.set_mode(int(shot[1]))
		_shot_frames = 3          # let the camera, lights and HUD settle
		return
	_shot_frames -= 1
	if _shot_frames == 0:
		get_viewport().get_texture().get_image().save_png(str(shot[0]))
		_shots.pop_front()
		if _shot_restore >= 0:
			rig.set_mode(_shot_restore)


## Perceived objects (a perception component wired into the sim bridge): box
## outlines at the ground and the top, coloured by class, with "#id class" labels.
func _draw_objects(objects: Array) -> void:
	_objects_mesh.clear_surfaces()
	for l in _object_labels.get_children():
		l.free()
	var b := D.Batch.new()
	for o in objects:
		var cls := clampi(int(o.get("cls", 0)), 0, CLASS_COLORS.size() - 1)
		var c: Color = CLASS_COLORS[cls]
		var center := Vector3(float(o.x), 0.0, float(o.z))
		var h := float(o.get("height", 1.5))
		var bottom := float(o.get("y", h / 2.0)) - h / 2.0
		for y in [bottom + 0.03, bottom + h + 0.02]:
			b.rect_outline(center, float(o.rot_y), float(o.length), float(o.width), 0.06, y, c)
		var l := Label3D.new()
		var oid := int(o.get("id", -1))
		l.text = ("#%d " % oid if oid >= 0 else "") + str(CLASS_NAMES[cls])
		l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		l.fixed_size = true
		l.pixel_size = 0.0008
		l.font_size = 24
		l.outline_size = 6
		l.modulate = c
		l.position = center + Vector3(0, bottom + h + 0.4, 0)
		_object_labels.add_child(l)
	b.commit(_objects_mesh, _overlay_mat)


func _set_ribbon(mesh: ArrayMesh, pts: Array, width: float, y: float, color: Color) -> void:
	mesh.clear_surfaces()
	var b := D.Batch.new()
	b.ribbon(pts, width, y, color)
	b.commit(mesh, _overlay_mat)


## Planned path: one ribbon per direction segment, coloured by gear, with a
## ring at every cusp (gear change).
func _draw_path(pts: Array, gears: Array) -> void:
	_path_mesh.clear_surfaces()
	var b := D.Batch.new()
	var seg: Array = []
	var seg_gear := -1
	for i in pts.size():
		var g := int(gears[i]) if i < gears.size() else 4
		if seg_gear != -1 and g != seg_gear:
			b.ribbon(seg, 0.16, 0.012, PATH_REV if seg_gear == GEAR_REVERSE else PATH_FWD)
			var c := D.to_xz(pts[i - 1])
			var ring: Array = []
			for k in 17:
				ring.append(c + Vector2(cos(TAU * k / 16.0), sin(TAU * k / 16.0)) * 0.22)
			b.ribbon(ring, 0.07, 0.014, Color(1, 1, 1, 0.9))
			seg = [pts[i - 1]]
		seg.append(pts[i])
		seg_gear = g
	b.ribbon(seg, 0.16, 0.012, PATH_REV if seg_gear == GEAR_REVERSE else PATH_FWD)
	b.commit(_path_mesh, _overlay_mat)
