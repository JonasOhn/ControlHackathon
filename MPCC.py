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
        N: int = 20,  # Prediction horizon
        q_theta: float = 0.1,  # Weight for the progress theta
        Q: np.array[float, float] = np.array([0.2, 0.2]),
        R: np.array[float, float] = np.array([0.001, 0.001]),
    ) -> None:
        self.model = model
        self.beta_target = beta_target
        self.delta_min, self.delta_max = delta_limits
        self.Fx_max = Fx_max
        self.N = N

        self.path = np.loadtxt("circular_path.csv", delimiter=",", skiprows=1)  # Skip header
        self.x_path = self.path[:, 0]  # Extract x-coordinates
        self.y_path = self.path[:, 1]  # Extract y-coordinates
        self.path_length = np.sum(np.sqrt(np.diff(self.x_path)**2 + np.diff(self.y_path)**2))
        print(f"Path length: {self.path_length:.2f} m")

        self.theta_controlpoints = np.linspace(0, self.path_length, len(self.x_path))

        # calculate the angle between contour and the x-axis
        self.phi_target = np.arctan2(
            np.gradient(self.y_path), np.gradient(self.x_path)
        )
        self.phi_target = np.unwrap(self.phi_target)  # Unwrap the angle to avoid discontinuities

        self.spline_x = ca.interpolant('LUT', 'bspline', [self.theta_controlpoints], self.x_path)
        self.spline_y = ca.interpolant('LUT', 'bspline', [self.theta_controlpoints], self.y_path)
        self.phi_target = ca.interpolant('LUT', 'bspline', [self.theta_controlpoints], self.phi_target)
        
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

        # initialize errors
        eps_c = [0 for _ in range(self.N)] # contouring error
        eps_l = [0 for _ in range(self.N)] # lag 

        # Cost function
        cost = 0
        for k in range(self.N):
            progress = X[6, k]  # progress variable
            progress_x = self.spline_x(progress)
            progress_y = self.spline_y(progress)
            x = X[0, k]
            y = X[1, k]
            eps_c[k] = np.sin(progress)*(x - progress_x) - np.cos(progress)*(y - progress_y)
            eps_l[k] = -np.cos(progress)*(x -progress_x) - np.sin(progress)*(y - progress_y)
            # vx = X[3, k]
            # vy = X[4, k]
            # yawrate = X[5, k]
            # delta_s = U[0, k]
            # Fx = U[1, k]

            delta_u = self.U[:2, k] - self.U[:2, k-1] if k > 0 else 0
            delta_v = self.U[2, k] - self.U[2, k-1] if k > 0 else 0

            # contouring cost
            cost += [eps_c[k], eps_l[k]].T @ Q @ [eps_c[k], eps_l[k]] - q_theta * progress + [delta_u, delta_v].T @ R @ [delta_u, delta_v]

        # terminal cost only on the 
        

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
        u0 = np.asarray(sol.value(self.U)[:, 0]).squeeze()
        return u0


    def get_path_coordinates(self, progress):
        x = self.spline_x(progress)
        y = self.spline_y(progress)
        return np.array([x, y]).flatten()


    def plot_path(self):
        """
        Plot the path from the CSV file.
        """
        plt.figure(figsize=(8, 8))
        plt.plot(self.x_path, self.y_path, label="Path", color="b", marker="o", markersize=3)
        x_path_interp = self.spline_x(self.theta_controlpoints)
        y_path_interp = self.spline_y(self.theta_controlpoints)
        plt.plot(x_path_interp, y_path_interp, label="Interpolated Path", color="r")
        plt.title("Path for Vehicle to Follow")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)

        plt.figure(figsize=(8, 8))
        # plot the target angle depending on progress
        plt.plot(self.theta_controlpoints, self.phi_target(self.theta_controlpoints), label="Target Angle", color="g")
        plt.title("Target Angle vs Progress")
        plt.xlabel("Progress [m]")
        plt.ylabel("Target Angle [rad]")
        plt.legend()
        plt.grid(True)

        plt.show()