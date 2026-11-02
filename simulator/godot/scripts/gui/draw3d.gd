## 3D drawing helpers for the GUI (display only): materials, boxes and
## `Batch`, which collects flat ground ribbons (thick lines with per-point
## colours) and outlines into a single mesh surface.
extends RefCounted


static func mat(color: Color, unshaded := false, emission := 0.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	if unshaded:
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	if color.a < 1.0:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	if emission > 0.0:
		m.emission_enabled = true
		m.emission = Color(color.r, color.g, color.b)
		m.emission_energy_multiplier = emission
	return m


## Unshaded material that takes its colour (and alpha) from the vertices.
static func vertex_color_mat() -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.vertex_color_use_as_albedo = true
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	return m


static func box(size: Vector3, material: Material) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var b := BoxMesh.new()
	b.size = size
	b.material = material
	mi.mesh = b
	return mi


static func to_xz(q) -> Vector2:
	return Batch.to_xz(q)


class Batch:
	var verts := PackedVector3Array()
	var cols := PackedColorArray()

	## A point given as Vector2 (x, z), Vector3 or [x, z] -> Vector2 (x, z).
	static func to_xz(q) -> Vector2:
		if q is Vector2:
			return q
		if q is Vector3:
			return Vector2(q.x, q.z)
		return Vector2(float(q[0]), float(q[1]))

	## Flat ribbon along `pts` ([x, z] / Vector2 / Vector3) at height `y`.
	## `colors`: one colour per point (or empty: `color`). At a reversal (cusp)
	## the ribbon keeps the incoming side, so it never folds over itself.
	func ribbon(pts: Array, width: float, y: float, color: Color, colors: Array = []) -> void:
		var p: Array[Vector2] = []
		var c: Array[Color] = []
		for i in pts.size():
			var v := to_xz(pts[i])
			if p.is_empty() or p[p.size() - 1].distance_to(v) > 1e-4:
				p.append(v)
				c.append(colors[i] if i < colors.size() else color)
		if p.size() < 2:
			return
		var h := width / 2.0
		var left: Array[Vector3] = []
		var right: Array[Vector3] = []
		for i in p.size():
			var t_in := (p[i] - p[i - 1]).normalized() if i > 0 else (p[1] - p[0]).normalized()
			var t_out := (p[i + 1] - p[i]).normalized() if i < p.size() - 1 else t_in
			var t := t_in + t_out
			var k := 1.0
			if t_in.dot(t_out) < 0.0 or t.length() < 1e-6:
				t = t_in
			else:
				t = t.normalized()
				k = clampf(1.0 / maxf(t.dot(t_in), 1e-3), 1.0, 2.0)   # limited miter
			var n := Vector2(-t.y, t.x) * h * k
			left.append(Vector3(p[i].x + n.x, y, p[i].y + n.y))
			right.append(Vector3(p[i].x - n.x, y, p[i].y - n.y))
		for i in p.size() - 1:
			_quad(left[i], right[i], left[i + 1], right[i + 1], c[i], c[i + 1])

	## Rectangle outline (length x width, centred at `center`, rotated by
	## `rot_y`); dashed when `dash` > 0.
	func rect_outline(center: Vector3, rot_y: float, length: float, w: float, line: float, y: float,
			color: Color, dash := 0.0) -> void:
		var b := Basis(Vector3.UP, rot_y)
		var hl := length / 2.0
		var hw := w / 2.0
		var corners := [Vector3(-hl, 0, -hw), Vector3(hl, 0, -hw), Vector3(hl, 0, hw), Vector3(-hl, 0, hw)]
		for k in 4:
			var a: Vector3 = center + b * corners[k]
			var e: Vector3 = center + b * corners[(k + 1) % 4]
			if dash <= 0.0:
				ribbon([a - (e - a).normalized() * line / 2.0, e + (e - a).normalized() * line / 2.0], line, y, color)
				continue
			var d := e - a
			var n := int(maxf(1.0, floor(d.length() / (2.0 * dash))))
			for j in n:
				ribbon([a + d * (float(j) / n), a + d * ((float(j) + 0.5) / n)], line, y, color)

	## Filled band between two polylines with the same number of points (e.g. the
	## left and right boundary of a lane) at height `y`.
	func band(left: Array, right: Array, y: float, color: Color) -> void:
		for i in mini(left.size(), right.size()) - 1:
			var l0 := to_xz(left[i])
			var l1 := to_xz(left[i + 1])
			var r0 := to_xz(right[i])
			var r1 := to_xz(right[i + 1])
			_quad(Vector3(l0.x, y, l0.y), Vector3(r0.x, y, r0.y), Vector3(l1.x, y, l1.y), Vector3(r1.x, y, r1.y),
				color, color)

	## Dashed polyline: `dash` metres painted, `gap` metres free.
	func dashed(pts: Array, width: float, y: float, color: Color, dash := 1.5, gap := 1.5) -> void:
		var seg: Array = []
		var acc := 0.0
		var painting := true
		for i in pts.size():
			var p := to_xz(pts[i])
			if i > 0:
				acc += p.distance_to(to_xz(pts[i - 1]))
			if painting:
				seg.append(p)
			if painting and acc >= dash:
				ribbon(seg, width, y, color)
				seg = []
				painting = false
				acc = 0.0
			elif not painting and acc >= gap:
				seg = [p]
				painting = true
				acc = 0.0
		if painting:
			ribbon(seg, width, y, color)

	func _quad(a: Vector3, b: Vector3, c: Vector3, d: Vector3, c0: Color, c1: Color) -> void:
		verts.append_array(PackedVector3Array([a, b, c, b, d, c]))
		cols.append_array(PackedColorArray([c0, c0, c1, c0, c1, c1]))

	## Add the collected triangles to `mesh` as one surface.
	func commit(mesh: ArrayMesh, material: Material) -> void:
		if verts.is_empty():
			return
		var arrays := []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = verts
		arrays[Mesh.ARRAY_COLOR] = cols
		var normals := PackedVector3Array()
		normals.resize(verts.size())
		normals.fill(Vector3.UP)
		arrays[Mesh.ARRAY_NORMAL] = normals
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
		mesh.surface_set_material(mesh.get_surface_count() - 1, material)
