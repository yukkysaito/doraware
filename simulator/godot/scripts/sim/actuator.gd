## Steering and acceleration actuator dynamics, advanced once per physics step.
##   steering:     dead time (FIFO) -> noise -> first-order lag -> rate limit -> angle limit
##   acceleration: dead time (FIFO) -> noise -> accel/decel limit -> first-order lag -> jerk limit
## All parameters come from the `actuator` dict of the reset request.
extends RefCounted

var steer := 0.0          # actual bicycle-equivalent front-wheel angle [rad], +left
var accel := 0.0          # actual signed longitudinal acceleration request [m/s^2]

var _dt := 0.01
var _steer_fifo: Array[float] = []
var _accel_fifo: Array[float] = []
var _steer_alpha := 1.0
var _accel_alpha := 1.0
var _steer_rate := INF
var _steer_limit := INF
var _jerk := INF
var _acc_limit := INF
var _dec_limit := INF
var _steer_noise := 0.0
var _accel_noise := 0.0
var _rng := RandomNumberGenerator.new()


static func _num(d: Dictionary, key: String, fallback: float) -> float:
	var v = d.get(key, null)
	return fallback if v == null else float(v)


static func _lag_alpha(tau: float, dt: float) -> float:
	# exact discretisation of x' = (u - x) / tau with zero-order hold
	return 1.0 if tau <= 0.0 else 1.0 - exp(-dt / tau)


func configure(cfg: Dictionary, max_steer_angle: float, dt: float, seed: int) -> void:
	_dt = dt
	_steer_alpha = _lag_alpha(_num(cfg, "steering_time_constant", 0.0), dt)
	_accel_alpha = _lag_alpha(_num(cfg, "acceleration_time_constant", 0.0), dt)
	_steer_rate = _num(cfg, "steering_rate_limit", INF)
	_steer_limit = _num(cfg, "steering_angle_limit", max_steer_angle)
	_jerk = _num(cfg, "jerk_limit", INF)
	_acc_limit = _num(cfg, "acceleration_limit", INF)
	_dec_limit = _num(cfg, "deceleration_limit", INF)
	_steer_noise = _num(cfg, "steering_noise_std", 0.0)
	_accel_noise = _num(cfg, "acceleration_noise_std", 0.0)
	_rng.seed = int(_num(cfg, "noise_seed", 0.0)) * 1000003 + seed
	var n_steer := int(round(_num(cfg, "steering_dead_time", 0.0) / dt))
	var n_accel := int(round(_num(cfg, "acceleration_dead_time", 0.0) / dt))
	_steer_fifo.clear()
	_accel_fifo.clear()
	for i in n_steer:
		_steer_fifo.append(0.0)
	for i in n_accel:
		_accel_fifo.append(0.0)
	steer = 0.0
	accel = 0.0


static func _delay(fifo: Array[float], value: float) -> float:
	if fifo.is_empty():
		return value
	fifo.append(value)
	return fifo.pop_front()


## Advance one physics step.
## `direction`: +1 DRIVE, -1 REVERSE, 0 otherwise (selects which limit applies).
## `emergency`: bypass the dead time and brake at the deceleration limit.
func update(steer_cmd: float, accel_cmd: float, direction: int, emergency: bool, speed: float) -> void:
	# ---- steering
	var s_target := _delay(_steer_fifo, steer_cmd)
	if _steer_noise > 0.0:
		s_target += _rng.randfn(0.0, _steer_noise)
	var ds := (s_target - steer) * _steer_alpha
	ds = clampf(ds, -_steer_rate * _dt, _steer_rate * _dt)
	steer = clampf(steer + ds, -_steer_limit, _steer_limit)

	# ---- acceleration
	var a_target := _delay(_accel_fifo, accel_cmd)
	if emergency:
		var d := direction
		if d == 0:
			d = 1 if speed >= 0.0 else -1
		a_target = -d * _dec_limit   # against the motion direction = braking
	elif _accel_noise > 0.0:
		a_target += _rng.randfn(0.0, _accel_noise)
	if direction != 0 and a_target * direction > 0.0:
		a_target = clampf(a_target, -_acc_limit, _acc_limit)   # propulsion
	else:
		a_target = clampf(a_target, -_dec_limit, _dec_limit)   # braking
	var da := (a_target - accel) * _accel_alpha
	da = clampf(da, -_jerk * _dt, _jerk * _dt)
	accel += da
