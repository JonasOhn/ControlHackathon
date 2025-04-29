import casadi as ca
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.baseModel import BaseModel


"""
Consider the kinematic model only:
x = (px, py, yaw, vx, vy, yawrate, theta)
a = (delta, Fx, progvel)
"""


@dataclass
class BicycleXYTireMPCConfig:
    nx: int = 7
    nu: int = 3
    lf: float = 0.5
    lr: float = 0.5
    Iz: float = 200
    m: float = 150
    B_tire: float = 9.0
    C_tire: float = 1.3
    D_tire: float = 1.0
    safety_radius: float = 0.8


class BicycleXYTireModelMPC(BaseModel):
    def __init__(self, sampling_time):
        super().__init__(sampling_time)
        self.model_name = "BicycleXYTireModelMPC"
        self.model_config = BicycleXYTireMPCConfig()

        x = ca.MX.sym('x', self.model_config.nx)
        u = ca.MX.sym('u', self.model_config.nu)

        l_f = self.model_config.lf
        l_r = self.model_config.lr
        m = self.model_config.m
        Iz = self.model_config.Iz
        g = 9.81
        B_tire = self.model_config.B_tire
        C_tire = self.model_config.C_tire
        D_tire = self.model_config.D_tire

        px = x[0]
        py = x[1]
        yaw = x[2]
        vx = x[3]
        vy = x[4]
        yawrate = x[5]
        theta = x[6]

        delta_s = u[0]
        Fx = u[1] * m
        Fx_f = Fx/2
        Fx_r = Fx/2
        progvel = u[2]

        BETA_SMOOTHMAX = 15.0
        VX_MIN_SMOOTHMAX = 1.0
        vx_n = (
            VX_MIN_SMOOTHMAX
            + ca.log(ca.exp(BETA_SMOOTHMAX * (vx - VX_MIN_SMOOTHMAX)) + VX_MIN_SMOOTHMAX)
            / BETA_SMOOTHMAX
        )
        alpha_f = ca.atan2(vy + l_f * yawrate, vx_n) - delta_s
        alpha_r = ca.atan2(vy - l_r * yawrate, vx_n)

        Fz_f = m * g * l_r / (l_f + l_r)
        Fz_r = m * g * l_f / (l_f + l_r)

        Fy_f = -Fz_f * D_tire * ca.sin(C_tire * ca.atan(B_tire * alpha_f))
        Fy_r = -Fz_r * D_tire * ca.sin(C_tire * ca.atan(B_tire * alpha_r))

        ax_v = (Fx_f * ca.cos(delta_s) + Fx_r - Fy_f * ca.sin(delta_s)) / m
        ay_v = (Fx_f * ca.sin(delta_s) + Fy_f * ca.cos(delta_s) + Fy_r) / m

        x_dot = ca.vertcat(
            vx * ca.cos(yaw) - vy * ca.sin(yaw),  # \dot{px}
            vx * ca.sin(yaw) + vy * ca.cos(yaw),  # \dot{py}
            yawrate,  # \dot{yaw}
            ax_v + vy * yawrate,  # \dot{vx}
            ay_v - vx * yawrate,  # \dot{vy}
            (Fx_f * ca.sin(delta_s) * l_f + Fy_f * ca.cos(delta_s) * l_f - Fy_r * l_r) / Iz,  # \dot{yawrate}
            progvel  # \dot{theta}
        )
        
        
        dae = {'x': x, 'p': u, 'ode': x_dot}
        opts = {'tf': self._sampling_time}
        self.I = ca.integrator('I', 'rk', dae, opts)


    def animateSimulation(self, x_trajectory, u_trajectory, num_agents:int=1, additional_lines_or_scatters=None):
        wheel_long_axis = 0.4
        wheel_short_axis = 0.1

        sim_length = u_trajectory.shape[1]
        _, ax = plt.subplots()

        nx = self.model_config.nx
        nu = self.model_config.nu

        for i in range(sim_length+1):
            ax.set_aspect('equal')
            # ax.set_xlim(-1, 5)
            # ax.set_ylim(-1, 5)
            ax.set_xlabel('px(m)', fontsize=14)
            ax.set_ylabel('py(m)', fontsize=14)

            for i_agent in range(num_agents):
                front_x = x_trajectory[i_agent*nx, i] + self.model_config.lf * ca.cos(x_trajectory[i_agent*nx+2, i])
                front_y = x_trajectory[i_agent*nx+1, i] + self.model_config.lf * ca.sin(x_trajectory[i_agent*nx+2, i])
                rear_x = x_trajectory[i_agent*nx, i] - self.model_config.lr * ca.cos(x_trajectory[i_agent*nx+2, i])
                rear_y = x_trajectory[i_agent*nx+1, i] - self.model_config.lr * ca.sin(x_trajectory[i_agent*nx+2, i])
                ax.plot(x_trajectory[i_agent*nx, :i+1], x_trajectory[i_agent*nx+1, :i+1], color="tab:gray", linewidth=2, zorder=0)
                ax.scatter(x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i], color="tab:gray", s=50, zorder=2)
                ax.plot([front_x, rear_x], [front_y, rear_y], color="tab:blue", linewidth=3, zorder=1)
                if i < sim_length:
                    wheel_f = patches.Ellipse((front_x, front_y), wheel_long_axis, wheel_short_axis, angle=math.degrees(x_trajectory[i_agent*nx+2, i] + u_trajectory[0, i]), color="tab:green", label="Wheels" if i_agent == 0 else None)
                    wheel_r = patches.Ellipse((rear_x, rear_y), wheel_long_axis, wheel_short_axis, angle=math.degrees(x_trajectory[i_agent*nx+2, i]), color="tab:green")
                    ax.add_patch(wheel_f)
                    ax.add_patch(wheel_r)
                safety_circle = patches.Circle((x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i]), self.model_config.safety_radius, color="tab:orange", alpha=0.3)
                ax.add_patch(safety_circle)
            if additional_lines_or_scatters is not None:
                for key, value in additional_lines_or_scatters.items():
                    if value["type"] == "scatter":
                        ax.scatter(value["data"][0], value["data"][1], color=value["color"], s=value["s"], label=key, marker=value["marker"])
                    elif value["type"] == "line":
                        ax.plot(value["data"][0], value["data"][1], color=value["color"], linewidth=2, label=key)
            ax.set_title(f"Bicycle Simulation: Step {i+1}")
            ax.legend()
            plt.show(block=False)
            plt.pause(1.0 if i == sim_length else 0.3)
            ax.clear()
        return