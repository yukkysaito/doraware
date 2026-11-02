## Cameras (GUI only):
##   [1] top: orthographic, over the map (wheel zooms)
##   [2] chase: behind the vehicle, smoothed (wheel changes the distance)
##   [3] free fly: WASD + Q/E, hold the right mouse button to look (wheel = speed)
##   [4] overview: perspective view of the whole map, following the vehicle
extends Node3D

enum Mode { TOP, CHASE, FREE, OVERVIEW }

var mode := Mode.CHASE
var cam := Camera3D.new()
var _target: Node3D
var _center := Vector3.ZERO
var _extent := 30.0
var _map_size := Vector2(30.0, 30.0)   # map extent along X and Z
var _zoom := 1.0
var _chase_dist := 8.0
var _yaw := 0.0
var _pitch := -0.5
var _free_speed := 8.0
var _snap := true          # jump to the next pose instead of easing (mode change, screenshots)


func _ready() -> void:
	cam.far = 2000.0
	cam.near = 0.05
	add_child(cam)
	cam.current = true


func set_target(target: Node3D, center: Vector3, map_size: Vector2) -> void:
	_target = target
	_center = center
	_map_size = Vector2(maxf(map_size.x, 5.0), maxf(map_size.y, 5.0))
	_extent = maxf(_map_size.x, _map_size.y)
	_zoom = 1.0
	_snap = true


## Screen band left free by the HUD in the top view (fractions of the height).
const TOP_BAND := Vector2(0.20, 0.82)


## Orthographic size (vertical world extent of the screen) that fits the map
## into the band between the HUD panels, with a margin.
func _top_fit() -> float:
	var vp := get_viewport().get_visible_rect().size
	var aspect := vp.x / maxf(vp.y, 1.0)
	var band := TOP_BAND.y - TOP_BAND.x
	return maxf(_map_size.y * 1.06 / band, _map_size.x * 1.04 / aspect)


## Camera position that puts the map centre in the middle of the band
## (screen down = world +Z in the top view).
func _top_center(size: float) -> Vector3:
	var offset := ((TOP_BAND.x + TOP_BAND.y) / 2.0 - 0.5) * size
	return _center - Vector3(0, 0, offset)


func set_mode(m: int) -> void:
	if m == Mode.FREE:
		_yaw = cam.global_rotation.y
		_pitch = cam.global_rotation.x
	mode = m
	_snap = true


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_1: set_mode(Mode.TOP)
			KEY_2: set_mode(Mode.CHASE)
			KEY_3: set_mode(Mode.FREE)
			KEY_4: set_mode(Mode.OVERVIEW)
	elif event is InputEventMouseMotion and mode == Mode.FREE \
			and Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
		_yaw -= event.relative.x * 0.004
		_pitch = clampf(_pitch - event.relative.y * 0.004, -1.5, 1.5)
	elif event is InputEventMouseButton and event.pressed:
		var f := 0.9 if event.button_index == MOUSE_BUTTON_WHEEL_UP else (1.1 if event.button_index == MOUSE_BUTTON_WHEEL_DOWN else 1.0)
		_zoom = clampf(_zoom * f, 0.1, 5.0)
		_chase_dist = clampf(_chase_dist * f, 3.0, 60.0)
		_free_speed /= f


func _process(delta: float) -> void:
	var goal := cam.global_transform
	var has_target := _target != null and is_instance_valid(_target)
	match mode:
		Mode.TOP:
			cam.projection = Camera3D.PROJECTION_ORTHOGONAL
			cam.size = _top_fit() * _zoom
			goal = Transform3D(Basis.from_euler(Vector3(-PI / 2.0, 0, 0)), _top_center(cam.size) + Vector3(0, 120, 0))
		Mode.CHASE:
			cam.projection = Camera3D.PROJECTION_PERSPECTIVE
			cam.fov = 55.0
			if has_target:
				var xf := _target.global_transform
				var fwd := xf.basis.x
				var eye := xf.origin - fwd * _chase_dist + Vector3(0, _chase_dist * 0.55, 0)
				goal = Transform3D(Basis(), eye).looking_at(xf.origin + fwd * 2.5 + Vector3(0, 0.5, 0), Vector3.UP)
		Mode.OVERVIEW:
			cam.projection = Camera3D.PROJECTION_PERSPECTIVE
			cam.fov = 45.0
			var focus := _center.lerp(_target.global_position, 0.35) if has_target else _center
			var eye := focus + Vector3(-_extent * 0.08, _extent * 0.55, _extent * 0.62) * _zoom
			goal = Transform3D(Basis(), eye).looking_at(focus, Vector3.UP)
		Mode.FREE:
			cam.projection = Camera3D.PROJECTION_PERSPECTIVE
			cam.fov = 60.0
			var basis := Basis.from_euler(Vector3(_pitch, _yaw, 0))
			var mv := Vector3.ZERO
			if Input.is_key_pressed(KEY_W): mv -= basis.z
			if Input.is_key_pressed(KEY_S): mv += basis.z
			if Input.is_key_pressed(KEY_A): mv -= basis.x
			if Input.is_key_pressed(KEY_D): mv += basis.x
			if Input.is_key_pressed(KEY_E): mv += Vector3.UP
			if Input.is_key_pressed(KEY_Q): mv -= Vector3.UP
			goal = Transform3D(basis, cam.global_position + mv * _free_speed * delta)
			_snap = true
	if _snap:
		cam.global_transform = goal
		_snap = false
	else:
		var k := 1.0 - exp(-delta * 6.0)
		cam.global_transform = cam.global_transform.interpolate_with(goal, k)
