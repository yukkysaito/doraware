from doraware.core.clock import TickSchedule, ns_to_step, step_to_ns
from doraware.simulation.coordinator import StepCoordinator


def test_tick_schedule_integer_ratio():
    s = TickSchedule(100, 50)
    assert [i for i in range(9) if s.is_tick(i)] == [0, 2, 4, 6, 8]
    assert s.last_tick(5) == 4 and s.next_tick(4) == 6


def test_tick_schedule_non_integer_ratio():
    s = TickSchedule(100, 30)
    ticks = [i for i in range(101) if s.is_tick(i)]
    assert len(ticks) == 31 and ticks[:4] == [0, 4, 7, 10]
    for i in range(1, 100):
        assert s.last_tick(i) <= i < s.next_tick(i)


def test_time_conversion():
    assert step_to_ns(3, 100) == 30_000_000 and ns_to_step(30_000_000, 100) == 3


def test_coordinator_chunks_cover_all_events():
    c = StepCoordinator(dict(physics_hz=100, controller_hz=50, planner_hz=10, state_hz=50, visualization_hz=20))
    step, visited = 0, [0]
    while step < 100:
        step += c.chunk(step)
        visited.append(step)
    for i in range(100):
        if c.needs_command(i) or c.publishes_state(i) or c.sends_viz(i):
            assert i in visited
