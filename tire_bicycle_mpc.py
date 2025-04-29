from __future__ import annotations

import math
from typing import Tuple
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np

from MPCC import MPCC
from models.bicyleXYTireModel import BicycleXYTireModel

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

    model = BicycleXYTireModel(dt)
    controller = MPCC(model, math.radians(beta_target_deg), vx_target=vx_target)
    controller.plot_path()

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

    model.animateSimulation(x_traj, u_traj)

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
