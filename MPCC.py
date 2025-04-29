from __future__ import annotations

import math
from typing import Tuple
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np

from models.bicyleXYTireModelMPC import BicycleXYTireModelMPC

class MPCC:
    def __init__(
        self,
        model: BicycleXYTireModelMPC,
        beta_target: float,
        vx_target: float = 10.0,
        delta_limits: Tuple[float, float] = (-0.6, 0.6),
        Fx_max: float = 4000,  # Maximum longitudinal force (N)
        N: int = 20  # Prediction horizon
    ) -> None:
        self.model = model
        self.beta_target = beta_target
        self.delta_min, self.delta_max = delta_limits
        self.Fx_max = Fx_max
        self.N = N
        
        self.opti = ca.Opti()  # Create an Opti object to define the optimization problem

        # Decision variables
        X = self.opti.variable(self.model.model_config.nx, self.N + 1)  # State trajectory (nx, N+1)
        U = self.opti.variable(self.model.model_config.nu, self.N)  # Control inputs (delta, Fx)

        self.X = X
        self.U = U

        # Parameters for initial conditions
        X0_param = self.opti.parameter(self.model.model_config.nx)
        self.X0_param = X0_param

        # Initial state constraint (vehicle state at time k=0)
        self.opti.subject_to(X[:, 0] == X0_param)

        # Cost function
        cost = 0
        for k in range(self.N):
            vx = X[3, k]
            vy = X[4, k]
            yawrate = X[5, k]
            delta_s = U[0, k]
            Fx = U[1, k]

            # Calculate the body slip angle (beta)
            BETA_SMOOTHMAX = 15.0
            VX_MIN_SMOOTHMAX = 1.0
            vx_n = (
                VX_MIN_SMOOTHMAX
                + ca.log(ca.exp(BETA_SMOOTHMAX * (vx - VX_MIN_SMOOTHMAX)) + VX_MIN_SMOOTHMAX)
                / BETA_SMOOTHMAX
            )
            beta = ca.atan2(vy, vx_n)

            cost += 10 * (beta - self.beta_target) ** 2
            cost += 0.2 * (vx - vx_target) ** 2
            cost += 0.001 * U[1, k] ** 2

        # terminal cost only on the last beta
        vx_n = (
                VX_MIN_SMOOTHMAX
                + ca.log(ca.exp(BETA_SMOOTHMAX * (X[3, self.N] - VX_MIN_SMOOTHMAX)) + VX_MIN_SMOOTHMAX)
                / BETA_SMOOTHMAX
            )
        beta_last = ca.atan2(X[4, self.N], vx_n)
        cost += 100 * (beta_last - self.beta_target) ** 2
        cost += 0.2 * (X[3, self.N] - vx_target) ** 2

        # Minimize cost
        self.opti.minimize(cost)

        # Dynamics constraints: Use the model's integrator to compute the next state
        for k in range(self.N):
            # Use the integrator to compute the next state
            x_next = self.model.I(x0=X[:, k], p=U[:, k])["xf"]

            # State propagation constraint
            self.opti.subject_to(X[:, k + 1] == x_next)

            # Input constraints (handle as symbolic expressions)
            self.opti.subject_to(U[0, k] >= self.delta_min)  # steering angle lower bound
            self.opti.subject_to(U[0, k] <= self.delta_max)  # steering angle upper bound
            self.opti.subject_to(U[1, k] >= -self.Fx_max)  # longitudinal force lower bound
            self.opti.subject_to(U[1, k] <= self.Fx_max)   # longitudinal force upper bound

        # Solver settings
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    def solve(self, x_now: np.ndarray) -> np.ndarray:
        """Solve the MPC problem to find optimal control input."""
        self.opti.set_value(self.X0_param, x_now)
        try:
            sol = self.opti.solve()
        except RuntimeError as exc:
            raise RuntimeError("MPC failed to solve: " + str(exc)) from exc

        # Get the first control input from the solution
        u0 = np.asarray(sol.value(self.U)[:, 0]).squeeze()

        return u0
