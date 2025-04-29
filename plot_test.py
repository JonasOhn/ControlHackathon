import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy.ndimage import binary_erosion

def generate_text_path(text: str, font_path: str, font_size: int, image_size: tuple[int, int] = (800, 400)) -> np.ndarray:
    """
    Generate a smooth path following a given text. The path includes rounded corners.
    
    Args:
    - text: The text to trace.
    - font_path: Path to the font file.
    - font_size: Size of the text font.
    - image_size: Size of the image (width, height).
    
    Returns:
    - path: A list of (x, y) coordinates representing the text path.
    """
    # Create a blank image to render the text
    image = Image.new('L', image_size, color=0)  # 'L' mode for grayscale (black background)
    draw = ImageDraw.Draw(image)
    
    # Load a font (you can use a .ttf file on your system)
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate text size and position it in the center
    bbox = draw.textbbox((0, 0), text, font=font)  # Get the bounding box of the text
    text_width = bbox[2] - bbox[0]  # Right - Left
    text_height = bbox[3] - bbox[1]  # Bottom - Top
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    
    # Render the text onto the image
    draw.text(position, text, font=font, fill=255)  # White text on black background
    
    # Convert the image to a numpy array
    text_array = np.array(image)
    
    # Apply erosion to get a cleaner path and avoid jagged edges (rounding corners)
    eroded = binary_erosion(text_array, structure=np.ones((3, 3)))  # Erode by 1 pixel for smoothing
    
    # Find contours using OpenCV (find paths)
    contours, _ = cv2.findContours(eroded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert the contours to a list of points (x, y)
    path_points = []
    for contour in contours:
        for point in contour:
            path_points.append((point[0][0], point[0][1]))
    
    # Merge disjoint contours to create a fully connected path
    path_points = np.array(path_points)

    # shift everything such that the first point is at (0, 0)
    path_points -= path_points[0]

    for point in path_points:
        # delete the next point if it is too close to the current point
        if np.linalg.norm(point - path_points[0]) < 15 and (point[0] != path_points[0][0] and point[1] != path_points[0][1]):
            path_points = np.delete(path_points, np.where(path_points == point)[0][0], axis=0)

    path_points = np.array(path_points)
    path_points = path_points[1:, :]

    # iterate over the array and insert a point in between two points if
    # they are too far apart
    for i in range(len(path_points) - 1):
        # calculate the distance between the two points
        dist = np.linalg.norm(path_points[i] - path_points[i + 1])
        if dist > 10:
            # insert a point in between
            new_point = (path_points[i] + path_points[i + 1]) / 2
            path_points = np.insert(path_points, i + 1, new_point, axis=0)
            # increment the index to skip the new point
            i += 1

    # smoothen the path using a bspline
    import scipy.interpolate
    tck, u = scipy.interpolate.splprep(path_points.T, s=100)  # Increase smoothing with s=5
    unew = np.arange(0, 1.0, 0.001)  # Finer interpolation for smoother path
    out = scipy.interpolate.splev(unew, tck)
    path_points = np.array(out).T
    
    # remove every n-th point to reduce the number of points
    n = 10
    path_points = path_points[::n, :]

    return path_points

def animate_car_on_path(path: np.ndarray, speed: float = 0.05, car_marker_size: float = 10.0) -> None:
    """
    Animate the car following the path.
    
    Args:
    - path: Array of (x, y) coordinates representing the text path.
    - speed: Speed of the car (how fast it moves along the path).
    - car_marker_size: Size of the car marker in the plot.
    """
    # Flip y-coordinates to account for Pillow's origin being at the top-left
    path[:, 1] = max(path[:, 1]) - path[:, 1]
    
    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(min(path[:, 0]) - 20, max(path[:, 0]) + 20)
    ax.set_ylim(min(path[:, 1]) - 20, max(path[:, 1]) + 20)
    ax.set_title("Car Following the Text Path")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    
    # Plot the text path
    ax.plot(path[:, 0], path[:, 1], color="blue", label="Text Path", marker="o", markersize=3)
    
    # Initialize a point for the car
    car, = ax.plot([], [], 'ro', markersize=car_marker_size, label="Car")
    
    # Function to update the car's position in the animation
    def update(frame):
        car.set_data([path[frame, 0]], [path[frame, 1]])  # Pass as sequences
        return car,
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(path), interval=speed * 1000, repeat=False)
    
    # Show the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the font file (use a valid path to a .ttf font on your system)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Example font path on Linux
    text = "PhD"
    font_size = 100  # Size of the font

    # Generate the text path
    path = generate_text_path(text, font_path, font_size)

    path = path / 10

    # Animate the car following the text path
    animate_car_on_path(path, speed=0.5, car_marker_size=10)

    # Optionally, save the path to a CSV
    np.savetxt("text_path.csv", path, delimiter=",", header="x,y", comments="")
    print("Path saved to text_path.csv")
