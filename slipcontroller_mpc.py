"""
slip_controller_simulation.py

Example usage of the kinematic BicycleXYModel together with a very simple
body‑slip controller.  The controller keeps the vehicle at a user‑defined
constant body‑slip angle β while cruising at a constant speed.

*   Builds and configures the model and the controller
*   Runs a forward simulation for 10 s (100 steps @ 0.1 s)
*   Visualises the resulting trajectory with the model’s built‑in
    `animateSimulation` method

Adjust the import path of `BicycleXYModel` if your project structure is
different.
"""

from __future__ import annotations

import math
from typing import Tuple

import casadi as ca
import numpy as np

# ↳ EDIT THIS IMPORT TO MATCH YOUR PROJECT LAYOUT
# If BicycleXYModel lives in, e.g., models/bicycleXYModel.py, change to
# `from models.bicycleXYModel import BicycleXYModel`
from models.bicyleXYModel import BicycleXYModel  # noqa: F401  (lint: imported but unused – we use it!)


class SlipController:
    """Feed‑forward body‑slip controller for the kinematic bicycle model.

    The body‑slip angle is defined (for the kinematic model) as::

        β = arctan( l_r * tan(δ) / (l_r + l_f) )

    Solving for the steering angle δ that produces a desired β gives::

        δ = arctan( tan(β) * (l_r + l_f) / l_r )

    Because the analytic relation is exact for the kinematic model,
    a simple feed‑forward steering command is enough; however\, steering
    saturation is respected via *delta_limits*.
    """

    def __init__(
        self,
        model: BicycleXYModel,
        beta_target: float,
        delta_limits: Tuple[float, float] = (-0.6, 0.6),
    ) -> None:
        """Create a slip controller.

        Args:
            model: Instance of :class:`BicycleXYModel`.
            beta_target: Desired body‑slip angle *β* in **radians**.
            delta_limits: Min/Max steering angle *δ* (radians).
        """
        self.model = model
        self.beta_target = float(beta_target)
        self.delta_min, self.delta_max = delta_limits

        # Feed‑forward steering that yields the requested slip
        lf = self.model.model_config.lf
        lr = self.model.model_config.lr
        self._delta_cmd: float = math.atan(math.tan(self.beta_target) * (lr + lf) / lr)

        # Saturate to physical steering limits
        self._delta_cmd = float(max(min(self._delta_cmd, self.delta_max), self.delta_min))

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def control(self, v_ref: float) -> ca.DM:
        """Return the control vector *u* = [δ, v_ref]^T (CasADi DM)."""
        return ca.DM([self._delta_cmd, v_ref])


# -------------------------------------------------------------------------
# Simulation driver / example script
# -------------------------------------------------------------------------

def run_simulation() -> None:
    """Simulate constant‑β cruising and visualise the result."""
    dt = 0.1              # [s] integration step
    horizon = 100         # [‑] number of simulation steps (→ 10 s)
    v_ref = 1.0           # [m/s] target forward velocity
    beta_target_deg = 10.0 # [°] desired body‑slip angle

    # Model and controller -------------------------------------------------
    model = BicycleXYModel(dt)
    controller = SlipController(model, math.radians(beta_target_deg))

    # Storage for trajectories
    nx = model.model_config.nx
    nu = model.model_config.nu
    x_traj = np.zeros((nx, horizon + 1))
    u_traj = np.zeros((nu, horizon))

    # Initial pose (px, py, yaw)
    x = np.array([0.0, 0.0, 0.0])
    x_traj[:, 0] = x

    # Main simulation loop -------------------------------------------------
    for k in range(horizon):
        u = controller.control(v_ref)
        xf = model.I(x0=ca.DM(x), p=u)["xf"]
        x = np.asarray(xf).flatten()

        # store ------------------------------------------------------------
        x_traj[:, k + 1] = x
        u_traj[:, k] = np.asarray(u).flatten()

    # Visualise ------------------------------------------------------------
    model.animateSimulation(x_traj, u_traj)


if __name__ == "__main__":
    run_simulation()
