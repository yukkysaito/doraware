"""Turn indicator / hazard light usage metrics (evaluation/metrics.py: SignalStats)."""
import pytest

from doraware.evaluation.metrics import SignalStats
from doraware.interfaces import Header, TurnIndicator, VehicleSignalStatus, to_ns


def _st(t, turn=TurnIndicator.OFF, hazard=False, left=False, right=False):
    return VehicleSignalStatus(header=Header(stamp=to_ns(t)), turn_indicator=turn, hazard_lights=hazard,
                               left_lamp_on=left, right_lamp_on=right)


def test_no_status_gives_no_keys():
    assert SignalStats().summary() == {}


def test_durations_activations_and_final_state():
    s = SignalStats()
    s.add(_st(0.0))
    s.add(_st(1.0, TurnIndicator.LEFT, left=True))
    s.add(_st(3.0))
    s.add(_st(4.0, hazard=True, left=True, right=True))
    s.add(_st(6.5, hazard=True))
    out = s.summary()
    assert out["signal_left_time"] == pytest.approx(2.0)
    assert out["signal_hazard_time"] == pytest.approx(2.5)
    assert out["signal_right_time"] == 0.0
    assert (out["signal_left_activations"], out["signal_hazard_activations"]) == (1, 1)
    assert out["signal_first_hazard_time"] == pytest.approx(4.0)
    assert out["signal_final_turn_indicator"] == "OFF" and out["signal_final_hazard_lights"] is True
    assert out["signal_lamp_inconsistencies"] == 0


def test_lamp_without_function_is_counted():
    s = SignalStats()
    s.add(_st(0.0, TurnIndicator.LEFT, right=True))
    s.add(_st(0.1, left=True))
    assert s.summary()["signal_lamp_inconsistencies"] == 2
