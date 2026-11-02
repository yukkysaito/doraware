"""Control component: follows the Trajectory and outputs ControlCommand.

The Dora entry point is ``node.py``; the control law is a plugin selected by
``control.controller.type`` (``registry.py``, ``plugins/<name>/``). Shared,
controller-independent helpers: ``gear.py`` (gear manager) and
``prediction.py`` (actuator delay compensation).
"""
