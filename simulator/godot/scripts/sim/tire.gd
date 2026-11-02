## Planar tire / brake force solver for one physics step.
##
## Works in the vehicle frame at the center of gravity with map-like axes
## (x forward, y left, r = yaw rate CCW). Generalised velocity u = (ux, uy, r),
## mass matrix M = diag(m, m, Iz). For wheel i at (px, py) (relative to the CoG)
## with steer angle d_i:
##   v_lon_i = J_lon_i . u,  v_lat_i = J_lat_i . u
##   J_lon = [ cos d,  sin d,  -cos d * py + sin d * px ]
##   J_lat = [-sin d,  cos d,   sin d * py + cos d * px ]
##
## Lateral: F = -C * alpha, alpha = atan2(v_lat, max(|v_lon|, v_min)) (low-speed
##   regularised). It is applied as a (secant) damper k_i = F / v_lat and solved
##   *implicitly* together with the brakes, so a tire force can never reverse the
##   lateral velocity within one step (the force opposes the *resulting*
##   velocity; stable at any speed and dt).
## Longitudinal: explicit drive force per wheel; brakes are Coulomb-type: an
##   implicit stiff damper (stop the wheel this step) unless that would need more
##   than the brake capacity B_i, in which case the force saturates at B_i
##   (small active-set iteration). => brakes oppose motion and hold at standstill
##   but never push the vehicle the other way.
## Finally each wheel's (lon, lat) force is limited by the friction circle mu*Fz.
extends RefCounted

const V_MIN := 0.1            # [m/s] slip angle regularisation speed
const BRAKE_STIFFNESS := 1.0e7  # [N s/m] implicit "stop the wheel" damper


static func _solve3(a: Array, b: Array) -> Array:
	# Gaussian elimination with partial pivoting on a 3x3 system (doubles).
	var m := []
	for i in 3:
		m.append([a[i][0], a[i][1], a[i][2], b[i]])
	for c in 3:
		var p := c
		for r in range(c + 1, 3):
			if absf(m[r][c]) > absf(m[p][c]):
				p = r
		if p != c:
			var t = m[c]
			m[c] = m[p]
			m[p] = t
		for r in range(c + 1, 3):
			var f: float = m[r][c] / m[c][c]
			for k in range(c, 4):
				m[r][k] -= f * m[c][k]
	var x := [0.0, 0.0, 0.0]
	for i in range(2, -1, -1):
		var s: float = m[i][3]
		for k in range(i + 1, 3):
			s -= m[i][k] * x[k]
		x[i] = s / m[i][i]
	return x


## wheels: Array of Dictionary {px, py, angle, fz, drive, brake}
## Returns {fx, fy, mz, wheel_forces: [[f_lon, f_lat], ...]} in the CoG frame.
static func solve(u: Array, wheels: Array, mass: float, iz: float, mu: float,
		cornering: float, dt: float) -> Dictionary:
	var n := wheels.size()
	var jl := []   # J_lon rows
	var jt := []   # J_lat rows
	var k_lat := []
	var saturated := []
	var f_brake := []
	for w in wheels:
		var c := cos(w.angle)
		var s := sin(w.angle)
		var lon := [c, s, -c * w.py + s * w.px]
		var lat := [-s, c, s * w.py + c * w.px]
		jl.append(lon)
		jt.append(lat)
		var v_lon: float = lon[0] * u[0] + lon[1] * u[1] + lon[2] * u[2]
		var v_lat: float = lat[0] * u[0] + lat[1] * u[1] + lat[2] * u[2]
		var v_ref := maxf(absf(v_lon), V_MIN)
		var alpha := atan2(v_lat, v_ref)
		k_lat.append(cornering * alpha / v_lat if absf(v_lat) > 1e-9 else cornering / v_ref)
		saturated.append(w.brake <= 0.0)
		f_brake.append(0.0)

	var u_new := u
	for _iter in n + 1:
		var a := [[mass, 0.0, 0.0], [0.0, mass, 0.0], [0.0, 0.0, iz]]
		var b := [mass * u[0], mass * u[1], iz * u[2]]
		for i in n:
			var w: Dictionary = wheels[i]
			var f_lon_explicit: float = w.drive + (f_brake[i] if saturated[i] else 0.0)
			for r in 3:
				b[r] += dt * jl[i][r] * f_lon_explicit
				for c in 3:
					a[r][c] += dt * k_lat[i] * jt[i][r] * jt[i][c]
					if not saturated[i]:
						a[r][c] += dt * BRAKE_STIFFNESS * jl[i][r] * jl[i][c]
		u_new = _solve3(a, b)
		var changed := false
		for i in n:
			if saturated[i]:
				continue
			var v: float = jl[i][0] * u_new[0] + jl[i][1] * u_new[1] + jl[i][2] * u_new[2]
			var fb := -BRAKE_STIFFNESS * v
			var cap: float = wheels[i].brake
			if absf(fb) > cap:
				saturated[i] = true
				f_brake[i] = signf(fb) * cap
				changed = true
			else:
				f_brake[i] = fb
		if not changed:
			break

	var fx := 0.0
	var fy := 0.0
	var mz := 0.0
	var forces := []
	for i in n:
		var w: Dictionary = wheels[i]
		var v_lat: float = jt[i][0] * u_new[0] + jt[i][1] * u_new[1] + jt[i][2] * u_new[2]
		var f_lat: float = -k_lat[i] * v_lat
		var f_lon: float = w.drive + f_brake[i]
		var limit: float = mu * w.fz
		var mag := sqrt(f_lon * f_lon + f_lat * f_lat)
		if mag > limit and mag > 0.0:
			f_lon *= limit / mag
			f_lat *= limit / mag
		forces.append([f_lon, f_lat])
		for r in 3:
			var q: float = jl[i][r] * f_lon + jt[i][r] * f_lat
			if r == 0:
				fx += q
			elif r == 1:
				fy += q
			else:
				mz += q
	return {"fx": fx, "fy": fy, "mz": mz, "wheel_forces": forces}
