from __future__ import annotations

import math
from typing import Tuple
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np

from models.bicyleXYTireModel import BicycleXYTireModel


class SlipController:
    def __init__(
        self,
        model: BicycleXYTireModel,
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


class ControlPath:
    def __init__(self) -> None:
        self.path = np.zeros((2, 0))  # Initialize an empty path
        self
    
def calculate_slip_angles(
    vx: float,
    vy: float,
    yawrate: float,
    delta: float,
    lf: float,
    lr: float,) -> Tuple[float, float]:
    vx = max(vx, 1e-6)  # Avoid division by zero
    alpha_f = math.atan2(vy + lf * yawrate, vx) - delta
    alpha_r = math.atan2(vy - lr * yawrate, vx)
    return alpha_f, alpha_r


def run_simulation() -> None:
    dt = 0.1
    n_sim = 50
    vx_target = 10.0
    beta_target_deg = 10.0

    # Model and controller -------------------------------------------------
    model = BicycleXYTireModel(dt)
    controller = SlipController(model, math.radians(beta_target_deg), vx_target=vx_target)

    # Storage for trajectories
    nx = model.model_config.nx
    nu = model.model_config.nu
    x_traj = np.zeros((nx, n_sim + 1))
    u_traj = np.zeros((nu, n_sim))

    # Initial pose (px, py, yaw)
    x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    x_traj[:, 0] = x

    # Main simulation loop -------------------------------------------------
    for k in range(n_sim):
        print(f"Step {k + 1}/{n_sim}")
        u = controller.solve(x)
        xf = model.I(x0=ca.DM(x), p=u)["xf"]
        x = np.asarray(xf).flatten()

        # store ------------------------------------------------------------
        x_traj[:, k + 1] = x
        u_traj[:, k] = np.asarray(u).flatten()

    # model.animateSimulation(x_traj, u_traj)

    plt.figure(figsize=(10, 5))
    plt.subplot(3, 3, 1)
    plt.plot(np.arange(n_sim + 1) * dt, x_traj[0, :], label='px')
    plt.xlabel('Time [s]')
    plt.ylabel('px [m]')
    plt.subplot(3, 3, 2)
    plt.plot(np.arange(n_sim + 1) * dt, x_traj[1, :], label='py')
    plt.xlabel('Time [s]')
    plt.ylabel('py [m]')
    plt.subplot(3, 3, 3)
    plt.plot(np.arange(n_sim + 1) * dt, x_traj[2, :], label='yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('yaw [rad]')
    plt.subplot(3, 3, 4)
    plt.plot(np.arange(n_sim + 1) * dt, x_traj[3, :], label='vx')
    plt.xlabel('Time [s]')
    plt.ylabel('vx [m/s]')
    plt.subplot(3, 3, 5)
    plt.plot(np.arange(n_sim + 1) * dt, x_traj[4, :], label='vy')
    plt.xlabel('Time [s]')
    plt.ylabel('vy [m/s]')
    plt.subplot(3, 3, 6)
    plt.plot(np.arange(n_sim + 1) * dt, x_traj[5, :], label='yaw_rate')
    plt.xlabel('Time [s]')
    plt.ylabel('yaw_rate [rad/s]')

    plt.subplot(3, 3, 7)
    plt.plot(np.arange(n_sim) * dt, u_traj[0, :], label='delta')
    plt.xlabel('Time [s]')
    plt.ylabel('delta [rad]')
    plt.subplot(3, 3, 8)
    plt.plot(np.arange(n_sim) * dt, u_traj[1, :], label='v_ref')
    plt.xlabel('Time [s]')
    plt.ylabel('v_ref [m/s]')
    plt.subplot(3, 3, 9)
    alpha_f, alpha_r = np.zeros(n_sim), np.zeros(n_sim)
    for k in range(n_sim):
        alpha_f[k], alpha_r[k] = calculate_slip_angles(
            x_traj[3, k], x_traj[4, k], x_traj[5, k], u_traj[0, k], model.model_config.lf, model.model_config.lr
        )
    plt.plot(np.arange(n_sim) * dt, alpha_r, label='alpha_r')
    plt.plot(np.arange(n_sim) * dt, alpha_f, label='alpha_f')
    plt.xlabel('Time [s]')
    plt.ylabel('alpha_f [rad]')
    plt.tight_layout()

    # plot desired and actual beta
    plt.figure(figsize=(10, 5))
    beta_target = np.full(n_sim, math.radians(beta_target_deg))
    beta_actual = np.zeros(n_sim)
    for k in range(n_sim):
        beta_actual[k] = math.atan2(x_traj[4, k], x_traj[3, k])
    plt.plot(np.arange(n_sim) * dt, beta_target, label='beta_target')
    plt.plot(np.arange(n_sim) * dt, beta_actual, label='beta_actual')
    plt.xlabel('Time [s]')
    plt.ylabel('beta [rad]')
    plt.legend()
    plt.title('Body Slip Angle')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_simulation()
