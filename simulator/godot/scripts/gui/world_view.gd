## Static scene of the GUI: sky and light, ground, 1 m / 5 m grid, map bounds,
## obstacles, the goal bay and the start marker. Rebuilt on every reset.
## Display only: the physics bodies are created by main.gd.
extends Node3D

const D = preload("res://scripts/gui/draw3d.gd")

const ASPHALT := Color(0.20, 0.21, 0.23)
const SURROUND := Color(0.13, 0.14, 0.15)
const GRID_MINOR := Color(1, 1, 1, 0.045)
const GRID_MAJOR := Color(1, 1, 1, 0.11)
const BOUNDS := Color(0.98, 0.80, 0.25)
const OBSTACLE := Color(0.52, 0.52, 0.54)
const OBSTACLE_EDGE := Color(0.25, 0.24, 0.23)
const GOAL := Color(0.30, 0.95, 0.55)
const PAINT := Color(0.93, 0.94, 0.95)

var grid: MeshInstance3D
var _root: Node3D


func _ready() -> void:
	var env := WorldEnvironment.new()
	var e := Environment.new()
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.32, 0.42, 0.58)
	sky_mat.sky_horizon_color = Color(0.66, 0.70, 0.76)
	sky_mat.ground_horizon_color = Color(0.40, 0.42, 0.45)
	sky_mat.ground_bottom_color = Color(0.12, 0.13, 0.14)
	sky_mat.sun_angle_max = 20.0
	var sky := Sky.new()
	sky.sky_material = sky_mat
	e.background_mode = Environment.BG_SKY
	e.sky = sky
	e.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	e.ambient_light_energy = 0.75
	e.tonemap_mode = Environment.TONE_MAPPER_LINEAR   # overlay colours must match the legend
	e.glow_enabled = true
	e.glow_intensity = 0.6
	e.glow_bloom = 0.05
	e.glow_hdr_threshold = 1.2
	e.ssao_enabled = true            # Forward+ only; ignored by the compatibility renderer
	e.ssao_radius = 1.2
	e.ssao_intensity = 1.6
	e.fog_enabled = true
	e.fog_light_color = Color(0.62, 0.66, 0.72)
	e.fog_density = 0.0007
	env.environment = e
	add_child(env)
	var sun := DirectionalLight3D.new()
	sun.rotation = Vector3(deg_to_rad(-52), deg_to_rad(35), 0)
	sun.light_energy = 1.0
	sun.light_color = Color(1.0, 0.97, 0.92)
	sun.shadow_enabled = true
	sun.shadow_blur = 1.5
	sun.directional_shadow_max_distance = 80.0
	add_child(sun)


func build(scene: Dictionary, vp: Dictionary, start: Transform3D) -> void:
	if _root != null:
		_root.free()
	_root = Node3D.new()
	add_child(_root)
	var bounds: Array = scene.get("bounds", [-20.0, 20.0, -20.0, 20.0])
	var x0 := float(bounds[0])
	var x1 := float(bounds[1])
	var z0 := float(bounds[2])
	var z1 := float(bounds[3])
	var cx := (x0 + x1) / 2.0
	var cz := (z0 + z1) / 2.0

	# ground: darker surroundings, asphalt inside the map bounds
	var outer := D.box(Vector3(x1 - x0 + 200.0, 0.02, z1 - z0 + 200.0), _ground_mat(SURROUND))
	outer.position = Vector3(cx, -0.03, cz)
	_root.add_child(outer)
	var inner := D.box(Vector3(x1 - x0, 0.02, z1 - z0), _ground_mat(ASPHALT))
	inner.position = Vector3(cx, -0.015, cz)
	_root.add_child(inner)

	# grid (1 m minor, 5 m major) and the map bounds as a painted double line
	var gb := D.Batch.new()
	for gx in range(int(ceil(x0)), int(floor(x1)) + 1):
		gb.ribbon([Vector2(gx, z0), Vector2(gx, z1)], 0.035 if gx % 5 == 0 else 0.02, 0.004,
			GRID_MAJOR if gx % 5 == 0 else GRID_MINOR)
	for gz in range(int(ceil(z0)), int(floor(z1)) + 1):
		gb.ribbon([Vector2(x0, gz), Vector2(x1, gz)], 0.035 if gz % 5 == 0 else 0.02, 0.004,
			GRID_MAJOR if gz % 5 == 0 else GRID_MINOR)
	grid = _commit(gb)
	var bb := D.Batch.new()
	for off in [0.0, 0.25]:
		bb.ribbon([Vector2(x0 - off, z0 - off), Vector2(x1 + off, z0 - off), Vector2(x1 + off, z1 + off),
			Vector2(x0 - off, z1 + off), Vector2(x0 - off, z0 - off)], 0.1, 0.006, BOUNDS)
	_commit(bb)

	# obstacles: concrete blocks with a dark top outline for readability from above
	var ob_mat := StandardMaterial3D.new()
	ob_mat.albedo_color = OBSTACLE
	ob_mat.roughness = 0.92
	var eb := D.Batch.new()
	for ob in scene.get("obstacles", []):
		var h := float(ob.get("height", 1.5))
		var rot := float(ob.get("rot_y", 0.0))
		var size := Vector3(float(ob.length), h, float(ob.width))
		var mi := D.box(size, ob_mat)
		mi.transform = Transform3D(Basis(Vector3.UP, rot), Vector3(float(ob.x), h / 2.0, float(ob.z)))
		_root.add_child(mi)
		eb.rect_outline(Vector3(float(ob.x), 0.0, float(ob.z)), rot, size.x, size.z, 0.05, h + 0.004, OBSTACLE_EDGE)
	_commit(eb)

	# goal bay: painted footprint, translucent fill, heading chevrons and a label
	var length: float = vp.get("front_overhang", 0.9) + vp.get("wheel_base", 2.79) + vp.get("rear_overhang", 1.0)
	var width: float = vp.get("wheel_tread", 1.6) + vp.get("left_overhang", 0.15) + vp.get("right_overhang", 0.15)
	var fx: float = (vp.get("wheel_base", 2.79) + vp.get("front_overhang", 0.9) - vp.get("rear_overhang", 1.0)) / 2.0
	var fz: float = -(vp.get("left_overhang", 0.15) - vp.get("right_overhang", 0.15)) / 2.0
	if scene.has("goal") and scene.goal != null:
		var g: Dictionary = scene.goal
		var gxf := Transform3D(Basis(Vector3.UP, float(g.get("rot_y", 0.0))), Vector3(float(g.x), 0.0, float(g.z)))
		_add_footprint(gxf, length, width, fx, fz, GOAL, true, "GOAL")
	_add_footprint(start, length, width, fx, fz, Color(0.75, 0.78, 0.82, 0.8), false, "START")


func _add_footprint(xf: Transform3D, length: float, width: float, fx: float, fz: float, color: Color,
		goal: bool, label: String) -> void:
	var center := xf * Vector3(fx, 0.0, fz)
	var rot := xf.basis.get_euler().y
	var b := D.Batch.new()
	b.rect_outline(center, rot, length + 0.3, width + 0.3, 0.1, 0.008, PAINT if goal else color,
		0.0 if goal else 0.25)
	# chevrons along the heading
	var fwd := xf.basis.x
	var left := -xf.basis.z
	for k in 2:
		var tip := center + fwd * (0.3 + 0.55 * k)
		b.ribbon([tip - fwd * 0.45 + left * 0.45, tip, tip - fwd * 0.45 - left * 0.45], 0.12, 0.009,
			Color(color.r, color.g, color.b, 0.9))
	_commit(b)
	if goal:
		var fill := D.box(Vector3(length + 0.2, 0.004, width + 0.2), D.mat(Color(color.r, color.g, color.b, 0.16), true))
		fill.transform = Transform3D(Basis(Vector3.UP, rot), center + Vector3(0, 0.003, 0))
		fill.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		_root.add_child(fill)
	var l := Label3D.new()
	l.text = label
	l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	l.fixed_size = true
	l.pixel_size = 0.0011
	l.font_size = 30
	l.outline_size = 8
	l.modulate = Color(color.r, color.g, color.b, 1.0)
	l.outline_modulate = Color(0, 0, 0, 0.75)
	l.position = center + Vector3(0, 1.9, 0)
	l.no_depth_test = goal            # the goal label stays visible; the start label is occluded
	_root.add_child(l)


func _commit(b) -> MeshInstance3D:
	var m := ArrayMesh.new()
	b.commit(m, D.vertex_color_mat())
	var mi := MeshInstance3D.new()
	mi.mesh = m
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_root.add_child(mi)
	return mi


static func _ground_mat(color: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = 0.95
	var noise := FastNoiseLite.new()
	noise.frequency = 0.9
	var tex := NoiseTexture2D.new()
	tex.noise = noise
	tex.seamless = true
	tex.width = 256
	tex.height = 256
	m.detail_enabled = true
	m.detail_albedo = tex
	m.detail_blend_mode = BaseMaterial3D.BLEND_MODE_MUL
	m.uv1_triplanar = true
	m.uv1_scale = Vector3(0.5, 0.5, 0.5)
	return m
