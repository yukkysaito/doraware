## Heads-up display (display only).
##   top left:      scenario, sim time, planning status
##   top right:     controller mode, solver, tracking errors, command
##   bottom centre: instrument cluster (indicators, speed, gear, acceleration,
##                  steering angle, hazard lights)
##   bottom left:   legend and key help ([H] toggles)
##   top centre:    warning banner (emergency brake, collision)
extends CanvasLayer

const BG := Color(0.06, 0.07, 0.09, 0.82)
const BORDER := Color(1, 1, 1, 0.08)
const TEXT := Color(0.93, 0.95, 0.97)
const MUTED := Color(0.58, 0.63, 0.70)
const DIM := Color(0.24, 0.27, 0.32)
const OK := Color(0.24, 0.86, 0.52)
const WARN := Color(1.0, 0.69, 0.13)
const ERR := Color(1.0, 0.33, 0.33)
const ACCENT := Color(0.25, 0.80, 1.0)

const LEGEND := [
	["planned path (forward)", Color(0.35, 0.65, 1.0)],
	["planned path (reverse)", Color(1.0, 0.62, 0.22)],
	["trajectory window", Color(0.25, 0.92, 1.0)],
	["controller prediction", Color(1.0, 0.35, 0.85)],
	["driven path", Color(1.0, 0.85, 0.35)],
]
const MODE_COLORS := {"track": OK, "track_reuse": WARN, "hold": ACCENT, "hold_release": ACCENT,
	"gear_change": WARN, "fallback": ERR}


# ------------------------------------------------------------------ widgets
class Chip extends PanelContainer:
	var label := Label.new()
	var _style := StyleBoxFlat.new()

	func _init() -> void:
		_style.set_corner_radius_all(6)
		_style.content_margin_left = 8
		_style.content_margin_right = 8
		_style.content_margin_top = 1
		_style.content_margin_bottom = 1
		add_theme_stylebox_override("panel", _style)
		label.add_theme_font_size_override("font_size", 13)
		add_child(label)

	func show_value(text: String, color: Color) -> void:
		label.text = text
		label.add_theme_color_override("font_color", color.lightened(0.1))
		_style.bg_color = Color(color.r, color.g, color.b, 0.18)
		_style.border_color = Color(color.r, color.g, color.b, 0.6)
		_style.set_border_width_all(1)


class TurnArrow extends Control:
	var dir := 1
	var lit := false
	var active := false

	func _draw() -> void:
		var w := size.x
		var h := size.y
		var pts := PackedVector2Array([Vector2(0.08, 0.36), Vector2(0.52, 0.36), Vector2(0.52, 0.12),
			Vector2(0.94, 0.5), Vector2(0.52, 0.88), Vector2(0.52, 0.64), Vector2(0.08, 0.64)])
		for i in pts.size():
			var x := pts[i].x if dir > 0 else 1.0 - pts[i].x
			pts[i] = Vector2(x * w, pts[i].y * h)
		var c := OK if lit else (OK.darkened(0.72) if active else DIM)
		draw_colored_polygon(pts, c)


class HazardIcon extends Control:
	var lit := false
	var active := false

	func _draw() -> void:
		var c := ERR if lit else (ERR.darkened(0.7) if active else DIM)
		var w := size.x
		var h := size.y
		var outer := PackedVector2Array([Vector2(w * 0.5, h * 0.1), Vector2(w * 0.95, h * 0.88),
			Vector2(w * 0.05, h * 0.88), Vector2(w * 0.5, h * 0.1)])
		draw_polyline(outer, c, 3.0, true)
		var inner := PackedVector2Array([Vector2(w * 0.5, h * 0.42), Vector2(w * 0.68, h * 0.74),
			Vector2(w * 0.32, h * 0.74), Vector2(w * 0.5, h * 0.42)])
		draw_polyline(inner, c, 2.0, true)


class SteeringIcon extends Control:
	var angle := 0.0     # front-wheel angle [rad], +left

	func _draw() -> void:
		var c := size / 2.0
		var r := minf(size.x, size.y) * 0.42
		draw_arc(c, r, 0.0, TAU, 48, TEXT, 3.0, true)
		var a := -angle                       # screen y points down: left = counter-clockwise
		for d in [Vector2(-1, 0), Vector2(1, 0), Vector2(0, 1)]:
			draw_line(c, c + d.rotated(a) * r, TEXT, 2.5, true)
		draw_circle(c, r * 0.18, TEXT)
		draw_circle(c + Vector2(0, -r).rotated(a), 3.0, ACCENT)


class AccelBar extends Control:
	var value := 0.0      # effective acceleration in the gear direction (+ drive, - brake)
	var command := NAN
	var max_drive := 1.5
	var max_brake := 3.0

	func _draw() -> void:
		var w := size.x
		var h := size.y
		draw_rect(Rect2(0, h * 0.3, w, h * 0.4), DIM)
		var x0 := w * max_brake / (max_brake + max_drive)
		var x := clampf(x0 + value / (max_brake + max_drive) * w, 0.0, w)
		draw_rect(Rect2(minf(x0, x), h * 0.3, absf(x - x0), h * 0.4), OK if value >= 0.0 else ERR)
		draw_line(Vector2(x0, h * 0.1), Vector2(x0, h * 0.9), MUTED, 1.0)
		if not is_nan(command):
			var xc := clampf(x0 + command / (max_brake + max_drive) * w, 0.0, w)
			draw_line(Vector2(xc, 0), Vector2(xc, h), TEXT, 2.0)


class GearSelector extends Control:
	const NAMES := ["P", "R", "N", "D"]
	const GEARS := [1, 2, 3, 4]
	var gear := 1
	var target := -1
	var blink := false

	func _draw() -> void:
		var font := get_theme_default_font()
		var cell := size.x / 4.0
		for i in 4:
			var rect := Rect2(i * cell + 2, 0, cell - 4, size.y)
			var current: bool = GEARS[i] == gear
			var pending: bool = GEARS[i] == target and blink
			if current:
				draw_rect(rect, Color(ACCENT.r, ACCENT.g, ACCENT.b, 0.9))
			elif pending:
				draw_rect(rect, Color(WARN.r, WARN.g, WARN.b, 0.55))
			var col := Color(0.03, 0.05, 0.07) if current else (TEXT if pending else MUTED)
			draw_string(font, Vector2(i * cell, size.y * 0.74), NAMES[i], HORIZONTAL_ALIGNMENT_CENTER, cell, 20, col)


# ------------------------------------------------------------------ state
var _scenario := Label.new()
var _time := Label.new()
var _planning := Chip.new()
var _mode := Chip.new()
var _rows := {}
var _speed := Label.new()
var _speed_kmh := Label.new()
var _steer_deg := Label.new()
var _left := TurnArrow.new()
var _right := TurnArrow.new()
var _hazard := HazardIcon.new()
var _steer := SteeringIcon.new()
var _accel := AccelBar.new()
var _gear := GearSelector.new()
var _banner := PanelContainer.new()
var _banner_label := Label.new()
var _help: Control
var _blink_t := 0.0


func _ready() -> void:
	_build_scenario_panel()
	_build_control_panel()
	_build_cluster()
	_build_help()
	_build_banner()


func set_limits(max_drive: float, max_brake: float) -> void:
	_accel.max_drive = maxf(max_drive, 0.1)
	_accel.max_brake = maxf(max_brake, 0.1)


func toggle_help() -> void:
	_help.visible = not _help.visible


## Called every rendered frame.
func update_from(vehicle: Node, s: Dictionary, info: Dictionary, delta: float) -> void:
	_blink_t = fmod(_blink_t + delta, 0.8)
	_scenario.text = str(info.get("scenario", "-"))
	_time.text = "t  %7.2f s" % float(s.time)
	if info.has("planning"):
		var st := str(info.planning)
		_planning.show_value(st, OK if st == "SUCCESS" else ERR)
	else:
		_planning.show_value("WAITING", MUTED)
	var mode := str(info.get("mode", "-"))
	_mode.show_value(mode.to_upper(), MODE_COLORS.get(mode, MUTED))
	_set_row("solver", "%s  %.2f ms" % [str(info.get("solver", "-")), float(info.get("solve_ms", 0.0))])
	_set_row("lateral error", "%+.3f m" % float(info.get("lateral_error", 0.0)))
	_set_row("heading error", "%+.2f°" % rad_to_deg(float(info.get("heading_error", 0.0))))
	_set_row("cmd accel", "%+.2f m/s²" % float(info.get("cmd_accel", 0.0)))
	_set_row("cmd steer", "%+.1f°" % rad_to_deg(float(info.get("cmd_steer", 0.0))))

	var v: float = s.v_local[0]
	_speed.text = "%.2f" % absf(v)
	_speed_kmh.text = "m/s   %.1f km/h" % (absf(v) * 3.6)
	var gear := int(s.gear)
	var dir := 1.0 if gear == 4 else (-1.0 if gear == 2 else 1.0)
	_gear.gear = gear
	_gear.target = -1
	_gear.blink = _blink_t < 0.4
	if bool(s.shifting) and info.has("gear_request"):
		_gear.target = int(info.gear_request)
	_gear.queue_redraw()
	_accel.value = float(s.accel_actual) * dir
	_accel.command = float(info.cmd_accel) * dir if info.has("cmd_accel") else NAN
	_accel.queue_redraw()
	_steer.angle = float(s.steer)
	_steer.queue_redraw()
	_steer_deg.text = "%+.1f°" % rad_to_deg(float(s.steer))
	var turn := int(s.turn_indicator)
	_left.active = turn == 1 or bool(s.hazard_lights)
	_right.active = turn == 2 or bool(s.hazard_lights)
	_left.lit = vehicle.left_lamp_on
	_right.lit = vehicle.right_lamp_on
	_hazard.active = bool(s.hazard_lights)
	_hazard.lit = bool(s.hazard_lights) and (vehicle.left_lamp_on or vehicle.right_lamp_on)
	for w in [_left, _right, _hazard]:
		w.queue_redraw()

	var warn := ""
	if int(s.get("contacts", 0)) > 0:
		warn = "COLLISION  ×%d" % int(s.contacts)
	elif bool(info.get("emergency", false)):
		warn = "EMERGENCY BRAKE"
	elif mode == "fallback":
		warn = "SAFE STOP (fallback)"
	_banner.visible = warn != ""
	_banner_label.text = warn


# ------------------------------------------------------------------ layout
static func _panel(radius := 12) -> PanelContainer:
	var p := PanelContainer.new()
	var st := StyleBoxFlat.new()
	st.bg_color = BG
	st.border_color = BORDER
	st.set_border_width_all(1)
	st.set_corner_radius_all(radius)
	st.content_margin_left = 16
	st.content_margin_right = 16
	st.content_margin_top = 12
	st.content_margin_bottom = 12
	st.shadow_color = Color(0, 0, 0, 0.35)
	st.shadow_size = 8
	p.add_theme_stylebox_override("panel", st)
	return p


static func _style_label(l: Label, size: int, color: Color) -> Label:
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", color)
	return l


func _anchor(c: Control, preset: int, grow_h: int, grow_v: int) -> void:
	add_child(c)
	c.set_anchors_and_offsets_preset(preset, Control.PRESET_MODE_MINSIZE, 16)
	c.grow_horizontal = grow_h
	c.grow_vertical = grow_v


func _build_scenario_panel() -> void:
	var p := _panel()
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 4)
	p.add_child(v)
	v.add_child(_style_label(Label.new(), 12, MUTED))
	v.get_child(0).text = "DORAWARE  ·  SCENARIO"
	v.add_child(_style_label(_scenario, 26, TEXT))
	v.add_child(_style_label(_time, 15, MUTED))
	var row := HBoxContainer.new()
	row.add_child(_style_label(Label.new(), 13, MUTED))
	row.get_child(0).text = "planning  "
	row.add_child(_planning)
	v.add_child(row)
	_anchor(p, Control.PRESET_TOP_LEFT, Control.GROW_DIRECTION_END, Control.GROW_DIRECTION_END)


func _build_control_panel() -> void:
	var p := _panel()
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 4)
	p.add_child(v)
	var head := HBoxContainer.new()
	head.add_child(_style_label(Label.new(), 12, MUTED))
	head.get_child(0).text = "CONTROLLER  "
	head.add_child(_mode)
	v.add_child(head)
	var g := GridContainer.new()
	g.columns = 2
	g.add_theme_constant_override("h_separation", 18)
	for key in ["solver", "lateral error", "heading error", "cmd accel", "cmd steer"]:
		g.add_child(_style_label(Label.new(), 14, MUTED))
		g.get_child(g.get_child_count() - 1).text = key
		var val := _style_label(Label.new(), 14, TEXT)
		val.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		val.custom_minimum_size = Vector2(120, 0)
		g.add_child(val)
		_rows[key] = val
	v.add_child(g)
	_anchor(p, Control.PRESET_TOP_RIGHT, Control.GROW_DIRECTION_BEGIN, Control.GROW_DIRECTION_END)


func _set_row(key: String, text: String) -> void:
	_rows[key].text = text


func _build_cluster() -> void:
	var p := _panel(18)
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 22)
	h.alignment = BoxContainer.ALIGNMENT_CENTER
	p.add_child(h)
	_left.dir = -1
	for w in [_left, _right]:
		w.custom_minimum_size = Vector2(44, 44)
		w.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	h.add_child(_left)
	var sp := VBoxContainer.new()
	sp.add_theme_constant_override("separation", -6)
	_speed.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_speed.custom_minimum_size = Vector2(130, 0)
	sp.add_child(_style_label(_speed, 46, TEXT))
	_speed_kmh.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	sp.add_child(_style_label(_speed_kmh, 13, MUTED))
	h.add_child(sp)
	var mid := VBoxContainer.new()
	mid.add_theme_constant_override("separation", 10)
	mid.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	_gear.custom_minimum_size = Vector2(176, 30)
	mid.add_child(_gear)
	_accel.custom_minimum_size = Vector2(176, 16)
	mid.add_child(_accel)
	var al := _style_label(Label.new(), 11, MUTED)
	al.text = "brake  ◂  acceleration  ▸  drive"
	al.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	mid.add_child(al)
	h.add_child(mid)
	var st := VBoxContainer.new()
	st.add_theme_constant_override("separation", 2)
	_steer.custom_minimum_size = Vector2(58, 58)
	st.add_child(_steer)
	_steer_deg.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	st.add_child(_style_label(_steer_deg, 13, MUTED))
	h.add_child(st)
	_hazard.custom_minimum_size = Vector2(42, 38)
	_hazard.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	h.add_child(_hazard)
	h.add_child(_right)
	_anchor(p, Control.PRESET_CENTER_BOTTOM, Control.GROW_DIRECTION_BOTH, Control.GROW_DIRECTION_BEGIN)


func _build_help() -> void:
	var p := _panel()
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 3)
	p.add_child(v)
	for item in LEGEND:
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		var sw := ColorRect.new()
		sw.color = item[1]
		sw.custom_minimum_size = Vector2(22, 5)
		sw.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		row.add_child(sw)
		var l := _style_label(Label.new(), 12, MUTED)
		l.text = item[0]
		row.add_child(l)
		v.add_child(row)
	var keys := _style_label(Label.new(), 12, MUTED)
	keys.text = "\n[1] top   [2] chase   [3] free   [4] overview\n[G] grid   [H] help   wheel: zoom   RMB: look (free)"
	v.add_child(keys)
	_help = p
	_anchor(p, Control.PRESET_BOTTOM_LEFT, Control.GROW_DIRECTION_END, Control.GROW_DIRECTION_BEGIN)


func _build_banner() -> void:
	var st := StyleBoxFlat.new()
	st.bg_color = Color(ERR.r, ERR.g, ERR.b, 0.85)
	st.set_corner_radius_all(10)
	st.content_margin_left = 22
	st.content_margin_right = 22
	st.content_margin_top = 6
	st.content_margin_bottom = 6
	_banner.add_theme_stylebox_override("panel", st)
	_banner.add_child(_style_label(_banner_label, 20, Color(1, 1, 1)))
	_banner.visible = false
	_anchor(_banner, Control.PRESET_CENTER_TOP, Control.GROW_DIRECTION_BOTH, Control.GROW_DIRECTION_END)
