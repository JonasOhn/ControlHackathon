import numpy as np
import matplotlib.pyplot as plt

def generate_circular_path(radius: float, total_time: float, dt: float) -> np.ndarray:
    # Time vector
    t = np.arange(0, total_time, dt)

    # Angular velocity (constant for a uniform circular motion)
    angular_velocity = 2 * np.pi / total_time  # Full circle in total_time seconds

    # Angle theta as a function of time
    theta = angular_velocity * t

    # Calculate x and y positions based on circular motion equations
    x = radius * np.sin(theta)
    y = radius * (1 - np.cos(theta))

    # Stack x and y coordinates into a single array
    path = np.vstack((x, y)).T

    return path

# Parameters for the circular path
radius = 10.0  # meters
total_time = 20.0  # seconds (time to complete one full circle)
dt = 0.1  # time step in seconds

# Generate the circular path
path = generate_circular_path(radius, total_time, dt)

# Plot the circular path
plt.figure(figsize=(8, 8))
plt.plot(path[:, 0], path[:, 1], label="Circular Path", color="b")
plt.scatter(path[0, 0], path[0, 1], color="g", label="Start Point")
plt.scatter(path[-1, 0], path[-1, 1], color="r", label="End Point")
plt.title("Circular Path for Car to Follow")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.axis("equal")  # Ensure equal scaling of x and y axes
plt.grid(True)
plt.show()

# Save the path as a CSV file
np.savetxt("circular_path.csv", path, delimiter=",", header="x,y", comments="")
print("Path saved to circular_path.csv")
