"""Godot simulator process + TCP client protocol (docs/architecture.md §7).

:class:`GodotSimulator` listens on a TCP port, launches Godot (headless or
GUI) which connects back, and exchanges newline-delimited JSON. All public
methods take/return **map-frame** data except the raw ``state`` reply dicts
(Godot frame; convert with :func:`doraware.simulation.plugins.godot.frame.state_to_vehicle_state`).
No Dora import.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from doraware.core.config import REPO_ROOT
from doraware.interfaces import SimulationScene, VehicleSignalCommand

import doraware.simulation.plugins.godot.frame as gf


class SimulatorError(RuntimeError):
    """Base class of simulator failures."""


class SimulatorDisconnected(SimulatorError, ConnectionError):
    """Godot closed the connection or exited."""


class SimulatorTimeout(SimulatorError, TimeoutError):
    """No connection / reply within the wall-clock timeout."""


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


class GodotSimulator:
    """Owns one Godot process.

    ``sim_cfg``: the ``simulation.godot`` config section (``binary``,
    ``project_dir`` repo-relative, ``headless``, ``host``, ``port``,
    ``connect_timeout``). ``log_path``: Godot stdout/stderr go there.
    ``physics_hz`` only selects ``--fixed-fps`` (one physics tick per engine
    frame, no real-time sync); the effective physics rate is set by ``reset``.
    """

    def __init__(self, sim_cfg: Dict, log_path: Optional[str] = None, *,
                 physics_hz: float = 100.0, headless: Optional[bool] = None,
                 reply_timeout: float = 60.0):
        self.cfg = dict(sim_cfg)
        self.headless = bool(self.cfg.get("headless", True)) if headless is None else bool(headless)
        self.reply_timeout = float(reply_timeout)
        self.physics_dt = 1.0 / float(physics_hz)
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._proc: Optional[subprocess.Popen] = None
        self._log = None
        binary = _resolve(self.cfg.get("binary", "tools/godot"))
        project = _resolve(self.cfg.get("project_dir", "simulator/godot"))
        if not binary.exists():
            raise SimulatorError(f"Godot binary not found: {binary}")
        if not (project / "project.godot").exists():
            raise SimulatorError(f"Godot project not found: {project}")
        host = str(self.cfg.get("host", "127.0.0.1"))
        timeout = float(self.cfg.get("connect_timeout", 30.0))

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, int(self.cfg.get("port", 0) or 0)))
        server.listen(1)
        port = server.getsockname()[1]
        cmd = [str(binary), "--path", str(project), "--fixed-fps", str(int(round(physics_hz)))]
        if self.headless:
            cmd.insert(1, "--headless")
        cmd += ["--", f"--host={host}", f"--port={port}", f"--connect_timeout={timeout}"]
        self.command_line = cmd
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "w")
        out = self._log if self._log else subprocess.DEVNULL
        env = dict(os.environ)
        try:
            self._proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
            server.settimeout(0.2)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    self._sock, _ = server.accept()
                    break
                except socket.timeout:
                    if self._proc.poll() is not None:
                        raise SimulatorDisconnected(
                            f"Godot exited with code {self._proc.returncode} before connecting (log: {log_path})") from None
                    if time.monotonic() > deadline:
                        raise SimulatorTimeout(f"Godot did not connect within {timeout} s (log: {log_path})") from None
        except BaseException:
            server.close()
            self._kill()
            raise
        server.close()
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(self.reply_timeout)

    # ------------------------------------------------------------ transport
    def _send(self, msg: Dict) -> None:
        if self._sock is None:
            raise SimulatorDisconnected("simulator is closed")
        try:
            self._sock.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        except OSError as e:
            raise SimulatorDisconnected(f"send to Godot failed: {e}") from e

    def _recv(self) -> Dict:
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout as e:
                raise SimulatorTimeout(f"no reply from Godot within {self.reply_timeout} s") from e
            except OSError as e:
                raise SimulatorDisconnected(f"receive from Godot failed: {e}") from e
            if not chunk:
                raise SimulatorDisconnected("Godot closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        msg = json.loads(line)
        if msg.get("type") != "state":
            raise SimulatorError(f"unexpected reply from Godot: {msg.get('type')!r}")
        return msg

    def request(self, msg: Dict) -> Dict:
        self._send(msg)
        return self._recv()

    # ------------------------------------------------------------ API
    def reset(self, scene: SimulationScene, actuator: Dict, physics_hz: float, seed: int,
              initial_gear: Optional[int] = None) -> Dict:
        """Rebuild the scene; returns the raw ``state`` reply (step 0)."""
        self.physics_dt = 1.0 / float(physics_hz)
        return self.request(gf.reset_request(scene, actuator, physics_hz, seed, initial_gear))

    def step(self, n: int, accel: float, steer: float, gear: int, emergency: bool = False,
             signals: Optional[VehicleSignalCommand] = None) -> Dict:
        """Apply the command for ``n`` physics steps; returns the raw ``state`` after them.
        ``signals`` (turn indicator / hazard lights) is latched; None keeps the current one."""
        return self.request(gf.step_request(n, accel, steer, gear, emergency, signals))

    def send_viz(self, planned=(), predicted=(), path=None, info=None, path_gear=()) -> None:
        """Map-frame (x, y) polylines and HUD values; no reply."""
        self._send(gf.viz_request(planned, predicted, path, info, path_gear))

    def screenshot(self, path: str, camera: int = 0) -> None:
        """GUI only: save an upcoming rendered frame (camera 0 top, 1 chase, 2 free); no reply."""
        self._send({"type": "screenshot", "path": str(path), "camera": int(camera)})

    def close(self, timeout: float = 5.0) -> None:
        if self._sock is not None:
            try:
                self._send({"type": "quit"})
            except SimulatorError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._kill()
        if self._log:
            self._log.close()
            self._log = None

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=5.0)

    @property
    def returncode(self) -> Optional[int]:
        return None if self._proc is None else self._proc.poll()

    def __enter__(self) -> "GodotSimulator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close(timeout=1.0)
        except Exception:
            pass
