"""Closed-loop test harness for the controller: a small Python plant and
synthetic time-based trajectories (imported by the other controller tests).

Plant: rear-axle kinematic bicycle + actuators (dead time FIFO, first-order
lag, steering-rate / jerk limit, saturation) + gear semantics identical to the
simulator contract (docs/architecture.md §8): propulsion in the gear
direction, a request against the gear is braking that never reverses the
motion, PARK holds, gear shifts only below gear_shift_max_speed and take
gear_shift_time of standstill.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from doraware.core.config import load_config
from doraware.control.plugins.mpc.follower import TrajectoryFollower
from doraware.interfaces import (ControlCommand, Gear, Header, Pose, Trajectory, Twist,
                                          Vector3, VehicleParameters, VehicleState, to_ns)
from doraware.core.trajectory import sample

VP = VehicleParameters()


# ====================================================================== plant
@dataclass
class PlantConfig:
    steering_dead_time: float = 0.0
    steering_time_constant: float = 0.05
    steering_rate_limit: float = 0.6
    acceleration_dead_time: float = 0.0
    acceleration_time_constant: float = 0.05
    jerk_limit: float = 10.0
    gear_shift_time: float = 0.3
    gear_shift_max_speed: float = 0.05


class Plant:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, gear=Gear.PARK, cfg: PlantConfig = PlantConfig(),
                 vp: VehicleParameters = VP, dt: float = 0.01):
        self.x, self.y, self.yaw, self.v = x, y, yaw, 0.0
        self.delta, self.acc = 0.0, 0.0
        self.gear = gear
        self.cfg, self.vp, self.dt = cfg, vp, dt
        self.cmd = ControlCommand()
        self.fifo_d = deque([0.0] * max(int(round(cfg.steering_dead_time / dt)), 0))
        self.fifo_a = deque([0.0] * max(int(round(cfg.acceleration_dead_time / dt)), 0))
        self.shift_target: Optional[Gear] = None
        self.shift_timer = 0.0
        self.gear_changes: List[tuple] = []     # (t, old, new, |v| when requested, x, y)
        self.t = 0.0

    def state(self) -> VehicleState:
        return VehicleState(header=Header(stamp=to_ns(self.t)), pose=Pose.from_xyyaw(self.x, self.y, self.yaw),
                            twist=Twist(linear=Vector3(self.v, 0.0, 0.0),
                                        angular=Vector3(0, 0, self.v * math.tan(self.delta) / self.vp.wheel_base)),
                            acceleration=Vector3(self.acc, 0, 0), steering_angle=self.delta, gear=self.gear)

    def apply(self, cmd: ControlCommand) -> None:
        self.cmd = cmd
        g = cmd.gear_request
        if g != Gear.NONE and g != self.gear and self.shift_target != g and abs(self.v) < self.cfg.gear_shift_max_speed:
            self.shift_target, self.shift_timer = g, 0.0
            self.shift_request_speed = abs(self.v)

    def step(self) -> None:
        c, dt, vp = self.cfg, self.dt, self.vp
        # dead time
        self.fifo_d.append(self.cmd.target_steering_angle)
        self.fifo_a.append(self.cmd.target_acceleration)
        d_in = self.fifo_d.popleft()
        a_in = self.fifo_a.popleft()
        # first-order lag + rate limits
        d_in = float(np.clip(d_in, -vp.max_steer_angle, vp.max_steer_angle))
        dd = (d_in - self.delta) * (dt / max(c.steering_time_constant, dt))
        self.delta += float(np.clip(dd, -c.steering_rate_limit * dt, c.steering_rate_limit * dt))
        a_in = float(np.clip(a_in, -vp.max_deceleration, vp.max_deceleration if self.gear.direction < 0 else vp.max_acceleration))
        da = (a_in - self.acc) * (dt / max(c.acceleration_time_constant, dt))
        self.acc += float(np.clip(da, -c.jerk_limit * dt, c.jerk_limit * dt))
        # gear semantics
        v0 = self.v
        gdir = self.gear.direction
        if self.shift_target is not None:
            self.v = math.copysign(max(abs(v0) - 3.0 * dt, 0.0), v0)        # held during the shift
            if abs(self.v) < 1e-6:
                self.shift_timer += dt
            if self.shift_timer >= c.gear_shift_time - 1e-9:
                self.gear_changes.append((self.t, self.gear, self.shift_target, self.shift_request_speed, self.x, self.y))
                self.gear, self.shift_target = self.shift_target, None
        elif gdir == 0:
            self.v = math.copysign(max(abs(v0) - 5.0 * dt, 0.0), v0)        # PARK / NEUTRAL hold
        elif self.acc * gdir > 0:
            self.v = v0 + self.acc * dt                                      # propulsion
        else:
            mag = abs(self.acc) * dt                                         # braking
            self.v = math.copysign(max(abs(v0) - mag, 0.0), v0) if v0 != 0.0 else 0.0
        vm = 0.5 * (v0 + self.v)
        w = vm * math.tan(self.delta) / vp.wheel_base
        ym = self.yaw + 0.5 * dt * w
        self.x += dt * vm * math.cos(ym)
        self.y += dt * vm * math.sin(ym)
        self.yaw = math.atan2(math.sin(self.yaw + dt * w), math.cos(self.yaw + dt * w))
        self.t += dt


# ============================================================ trajectories
@dataclass
class Seg:
    """Motion segment (from standstill to standstill) or a dwell (length 0)."""
    direction: int = 1
    length: float = 0.0
    vmax: float = 1.0
    curvature: Callable[[float], float] = lambda s: 0.0   # kappa(s), s = travelled distance
    ramp_time: float = 3.0                                # cosine speed ramp
    dwell: float = 0.0


@dataclass
class Master:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    a: np.ndarray
    kappa: np.ndarray
    delta: np.ndarray
    gear: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.t[-1])


def build_master(segs: List[Seg], x0=0.0, y0=0.0, yaw0=0.0, start_dwell=1.5, h=0.002,
                 steer_ramp_time=1.0) -> Master:
    """Fine-grained time profile. Dwells ramp the steering to the next segment's
    kappa(0) (cosine) and keep the previous gear label (the gear manager must
    look ahead to the next moving point)."""
    rows = []
    x, y, yaw, t = x0, y0, yaw0, 0.0
    gear = Gear.from_direction(next(s.direction for s in segs if s.length > 0))
    kappa = 0.0
    segs = [Seg(dwell=start_dwell)] + list(segs)
    for idx, s in enumerate(segs):
        if s.length <= 0.0:
            nxt = next((q for q in segs[idx + 1:] if q.length > 0), None)
            k_to = nxt.curvature(0.0) if nxt is not None else kappa
            k_from = kappa
            n = int(round(s.dwell / h))
            for i in range(n):
                f = min(i * h / steer_ramp_time, 1.0)
                kappa = k_from + (k_to - k_from) * 0.5 * (1 - math.cos(math.pi * f))
                rows.append((t, x, y, yaw, 0.0, 0.0, kappa, gear))
                t += h
            kappa = k_to if n > 0 else kappa
            continue
        gear = Gear.from_direction(s.direction)
        tr = s.ramp_time
        cruise = (s.length - s.vmax * tr) / s.vmax
        assert cruise >= 0.0, "segment too short for its speed ramp"
        T = 2 * tr + cruise
        dist = 0.0
        n = int(math.ceil(T / h))
        for i in range(n + 1):
            tau = min(i * h, T)

            def speed(tt):
                if tt < tr:
                    return s.vmax * 0.5 * (1 - math.cos(math.pi * tt / tr))
                if tt <= tr + cruise:
                    return s.vmax
                return s.vmax * 0.5 * (1 - math.cos(math.pi * (T - tt) / tr))

            def accel(tt):
                if tt < tr:
                    return s.vmax * 0.5 * math.pi / tr * math.sin(math.pi * tt / tr)
                if tt <= tr + cruise:
                    return 0.0
                return -s.vmax * 0.5 * math.pi / tr * math.sin(math.pi * (T - tt) / tr)

            sp, ac = speed(tau), accel(tau)
            kappa = s.curvature(dist)
            rows.append((t, x, y, yaw, s.direction * sp, s.direction * ac, kappa, gear))
            if i == n:
                break
            # integrate one step (midpoint)
            sp2 = speed(min(tau + h, T))
            vm = s.direction * 0.5 * (sp + sp2)
            km = s.curvature(dist + 0.5 * abs(vm) * h)
            ym = yaw + 0.5 * h * vm * km
            x += h * vm * math.cos(ym)
            y += h * vm * math.sin(ym)
            yaw += h * vm * km
            dist += abs(vm) * h
            t += h
        t += h
    arr = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], int(r[7])) for r in rows])
    kap = arr[:, 6]
    return Master(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], kap,
                  np.arctan(VP.wheel_base * kap), arr[:, 7].astype(np.int8))


def window(m: Master, t_pub: float, time_step: float, horizon: float, validity: float = 1.0,
           seq: int = 0, stamp: Optional[float] = None) -> Trajectory:
    """Trajectory window [t_pub, t_pub + horizon] sampled from the master profile.
    Beyond the master end the final (stationary) state is held."""
    n = int(round(horizon / time_step)) + 1
    ts = t_pub + time_step * np.arange(n)
    tc = np.clip(ts, m.t[0], m.t[-1])
    yaw = np.interp(tc, m.t, m.yaw)
    idx = np.clip(np.searchsorted(m.t, tc), 0, len(m.t) - 1)
    beyond = ts > m.t[-1]
    v = np.where(beyond, 0.0, np.interp(tc, m.t, m.v))
    a = np.where(beyond, 0.0, np.interp(tc, m.t, m.a))
    st = t_pub if stamp is None else stamp
    return Trajectory(header=Header(stamp=to_ns(st), sequence_id=seq), start_time=to_ns(t_pub),
                      time_step=time_step, valid_until=to_ns(st + validity),
                      x=np.interp(tc, m.t, m.x), y=np.interp(tc, m.t, m.y),
                      yaw=np.arctan2(np.sin(yaw), np.cos(yaw)), longitudinal_velocity=v,
                      longitudinal_acceleration=a, curvature=np.interp(tc, m.t, m.kappa),
                      front_wheel_angle=np.interp(tc, m.t, m.delta), gear=m.gear[idx].astype(np.int8))


# ============================================================== closed loop
def make_config(**overrides) -> dict:
    """``overrides``: dotted keys with '__' instead of '.', e.g. timing__controller_hz=20."""
    cfg = copy.deepcopy(load_config(path=None))
    for k, v in overrides.items():
        node = cfg
        parts = k.split("__")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {k!r}")   # a typo must not silently do nothing
        node[parts[-1]] = v
    return cfg


@dataclass
class Log:
    t: List[float] = field(default_factory=list)
    acc: List[float] = field(default_factory=list)
    steer: List[float] = field(default_factory=list)
    gear_cmd: List[int] = field(default_factory=list)
    mode: List[str] = field(default_factory=list)
    lat: List[float] = field(default_factory=list)
    lon: List[float] = field(default_factory=list)
    hdg: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    vgear: List[int] = field(default_factory=list)
    solve: List[float] = field(default_factory=list)
    status: List[str] = field(default_factory=list)

    def np(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name))


@dataclass
class Result:
    log: Log
    plant: Plant
    follower: TrajectoryFollower
    master: Master
    cfg: dict

    def final_errors(self):
        p = self.plant
        dx, dy = p.x - self.master.x[-1], p.y - self.master.y[-1]
        dyaw = math.atan2(math.sin(p.yaw - self.master.yaw[-1]), math.cos(p.yaw - self.master.yaw[-1]))
        return math.hypot(dx, dy), abs(math.degrees(dyaw)), abs(p.v)


def run_closed_loop(master: Master, cfg: Optional[dict] = None, plant_cfg: PlantConfig = PlantConfig(),
                    t_end: Optional[float] = None, physics_hz: int = 100,
                    publish: Optional[Callable[[float], Optional[Trajectory]]] = None,
                    x0: Optional[tuple] = None) -> Result:
    cfg = cfg or make_config()
    tm = cfg["timing"]
    follower = TrajectoryFollower(cfg, VP)
    sx, sy, syaw = x0 if x0 is not None else (master.x[0], master.y[0], master.yaw[0])
    plant = Plant(sx, sy, syaw, cfg=plant_cfg, dt=1.0 / physics_hz)
    ctrl_every = int(round(physics_hz / tm["controller_hz"]))
    plan_every = int(round(physics_hz / tm["planner_hz"]))
    t_end = t_end if t_end is not None else master.duration + 4.0
    seq = 0
    log = Log()
    for i in range(int(round(t_end * physics_hz)) + 1):
        t = i / physics_hz
        if i % plan_every == 0:
            if publish is not None:
                tr = publish(t)
            else:
                seq += 1
                tr = window(master, t, tm["trajectory_time_step"], tm["trajectory_horizon"],
                            tm["trajectory_validity"], seq)
            if tr is not None:
                follower.on_trajectory(tr)
        if i % ctrl_every == 0:
            st = plant.state()
            cmd, dbg = follower.step(st, to_ns(t))
            plant.apply(cmd)
            log.t.append(t)
            log.acc.append(cmd.target_acceleration)
            log.steer.append(cmd.target_steering_angle)
            log.gear_cmd.append(int(cmd.gear_request))
            log.mode.append(dbg.mode)
            log.lat.append(dbg.lateral_error)
            log.lon.append(dbg.longitudinal_error)
            log.hdg.append(dbg.heading_error)
            log.v.append(st.speed)
            log.vgear.append(int(st.gear))
            log.solve.append(dbg.solve_time if dbg.mode.startswith("track") or dbg.solver_status not in ("", "no_valid_trajectory") else float("nan"))
            log.status.append(dbg.solver_status)
        plant.step()
    return Result(log, plant, follower, master, cfg)


def check_constraints(res: Result, tol: float = 1e-6) -> None:
    """Command-level constraints: steering angle/rate, gear-mapped accel bounds, jerk."""
    log, vp = res.log, VP
    T = 1.0 / res.cfg["timing"]["controller_hz"]
    steer, acc = log.np("steer"), log.np("acc")
    assert np.all(np.abs(steer) <= vp.max_steer_angle + tol)
    assert np.all(np.abs(np.diff(steer)) <= vp.max_steer_rate * T + tol), np.max(np.abs(np.diff(steer))) / T
    assert np.all(np.abs(np.diff(acc)) <= vp.max_jerk * T + tol), np.max(np.abs(np.diff(acc))) / T
    gd = np.array([Gear(g).direction for g in log.vgear])
    hi = np.where(gd < 0, vp.max_deceleration, np.where(gd > 0, vp.max_acceleration, vp.max_deceleration))
    lo = np.where(gd < 0, -vp.max_acceleration, -vp.max_deceleration)
    assert np.all(acc <= hi + tol) and np.all(acc >= lo - tol)


def check_gear_changes(res: Result) -> None:
    gcs = res.cfg["control"]["controller"]["mpc"]["gear_change_speed"]
    log = res.log
    for i, g in enumerate(log.gear_cmd):
        if g != int(Gear.NONE):
            assert abs(log.v[i]) < gcs, f"gear requested at |v|={abs(log.v[i])}"
    for (_, _, _, vreq, _, _) in res.plant.gear_changes:
        assert vreq < gcs


# a plant-only sanity test so this helper module is also exercised on its own
def test_plant_gear_semantics():
    p = Plant(gear=Gear.DRIVE)
    p.apply(ControlCommand(target_acceleration=1.0))
    for _ in range(100):
        p.step()
    assert p.v > 0.5
    p.apply(ControlCommand(target_acceleration=-3.0))
    for _ in range(200):
        p.step()
    assert p.v == 0.0                     # braking never reverses
    p.apply(ControlCommand(target_acceleration=0.0, gear_request=Gear.REVERSE))
    for _ in range(40):
        p.step()
    assert p.gear == Gear.REVERSE
    p.apply(ControlCommand(target_acceleration=-1.0))
    for _ in range(50):
        p.step()
    assert p.v < -0.2


def test_master_profile_consistent():
    m = build_master([Seg(1, 6.0, 1.0)])
    assert abs(m.x[-1] - 6.0) < 0.02 and abs(m.v[-1]) < 1e-9
    tr = window(m, 2.0, 0.05, 5.0)
    p = sample(tr, 3.0)
    assert abs(p.x - np.interp(3.0, m.t, m.x)) < 1e-3
