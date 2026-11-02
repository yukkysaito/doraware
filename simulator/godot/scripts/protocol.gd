## Newline-delimited JSON over TCP (client side). See docs/architecture.md §7.
## The Python bridge listens; this side connects and exchanges one JSON object
## per line. Purely transport: no simulation logic here.
extends RefCounted

var _tcp := StreamPeerTCP.new()
var _buf := PackedByteArray()
var connected := false


func connect_to(host: String, port: int, timeout_s: float = 30.0) -> bool:
	if _tcp.connect_to_host(host, port) != OK:
		return false
	var t0 := Time.get_ticks_msec()
	while Time.get_ticks_msec() - t0 < int(timeout_s * 1000.0):
		_tcp.poll()
		var st := _tcp.get_status()
		if st == StreamPeerTCP.STATUS_CONNECTED:
			_tcp.set_no_delay(true)
			connected = true
			return true
		if st == StreamPeerTCP.STATUS_ERROR:
			return false
		OS.delay_usec(1000)
	return false


## Returns the next complete message (Dictionary) or null if none is buffered.
## Sets `connected = false` when the peer closed the connection.
func poll_message():
	var msg = _pop_line()
	if msg != null:
		return msg
	_tcp.poll()
	if _tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		connected = false
		return null
	var n := _tcp.get_available_bytes()
	if n > 0:
		var res := _tcp.get_partial_data(n)
		if res[0] == OK:
			_buf.append_array(res[1])
	return _pop_line()


func _pop_line():
	var idx := _buf.find(10)  # '\n'
	if idx < 0:
		return null
	var line := _buf.slice(0, idx).get_string_from_utf8()
	_buf = _buf.slice(idx + 1)
	var data = JSON.parse_string(line)
	if typeof(data) != TYPE_DICTIONARY:
		push_error("protocol: ignoring malformed line: %s" % line.left(200))
		return {}
	return data


func send(msg: Dictionary) -> void:
	# full_precision: doubles round-trip exactly (determinism checks compare bits)
	var line := JSON.stringify(msg, "", false, true) + "\n"
	_tcp.put_data(line.to_utf8_buffer())


func close() -> void:
	_tcp.disconnect_from_host()
	connected = false
