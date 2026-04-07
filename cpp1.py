import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Parameters for the matrix
width = 100    # Number of columns
length = 100   # Number of rows
base_height = 0    # Minimum height of the slope
max_height = 100   # Maximum height of the slope
camrot=np.zeros((width,length))
cam_min_foc_len=5
cam_max_foc_len=100

# Generate a sigmoid array for altitude values with a steeper incline at the top
x = np.linspace(-6, 3, length)  # Adjusted range for sharp decline at top

sigmoid_heights = base_height + (max_height - base_height) * (1 / (1 + np.exp(-x)))

# Create the slope matrix with the same altitude for each cell in a row
slope_matrix = np.tile(sigmoid_heights[:, np.newaxis], (1, width))

# # Display the generated slope matrix
# print("Slope Matrix (Sigmoid Pattern):")
# print(slope_matrix)

# Define movement costs
horizontal_cost = 1
vertical_cost = 5
yaw_cost = 2

# Calculate yaw between two points for rotation cost
def calculate_yaw(point1, point2):
    delta_x, delta_y = point2[0] - point1[0], point2[1] - point1[1]
    return np.degrees(np.arctan2(delta_y, delta_x))

# 3D Path cost function including yaw, horizontal, vertical movements
def compute_path_cost(path_x, path_y, path_z, horizontal_cost, vertical_cost, yaw_cost):
    total_cost = 0
    if len(path_x) < 2:
        return total_cost  # No movement if only one point

    current_yaw = calculate_yaw((path_x[0], path_y[0]), (path_x[1], path_y[1]))  # Initial orientation

    for i in range(1, len(path_x)):
        dx, dy, dz = path_x[i] - path_x[i - 1], path_y[i] - path_y[i - 1], path_z[i] - path_z[i - 1]

        # Horizontal and vertical movement costs
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        total_cost += horizontal_distance * horizontal_cost
        total_cost += abs(dz) * vertical_cost

        # Yaw change cost
        new_yaw = calculate_yaw((path_x[i - 1], path_y[i - 1]), (path_x[i], path_y[i]))
        yaw_change = abs(new_yaw - current_yaw)
        total_cost += yaw_change * yaw_cost

        current_yaw = new_yaw  # Update yaw for the next step

    return total_cost

# Initialize lists for the optimal path coordinates
optimal_path_x, optimal_path_y, optimal_path_z = [], [], []
previous_end = 0  # Starting on the left side for the first row

# Generate the optimized lawnmower search pattern with altitude constraints
for row in range(length):
    # Two options for each row: start left-to-right or right-to-left
    left_to_right_x = list(range(width))
    right_to_left_x = list(range(width - 1, -1, -1))
    
    # Define altitude bounds for the current row based on ground altitude
    ground_altitude = sigmoid_heights[row]
    min_altitude = ground_altitude + cam_min_foc_len
    max_altitude = ground_altitude + cam_max_foc_len
    target_altitude = (min_altitude + max_altitude) / 2  # Midpoint altitude within the focal length range
    
    # Candidate paths
    left_to_right_path_x = optimal_path_x + left_to_right_x
    left_to_right_path_y = optimal_path_y + [row] * width
    left_to_right_path_z = optimal_path_z + [target_altitude] * width

    right_to_left_path_x = optimal_path_x + right_to_left_x
    right_to_left_path_y = optimal_path_y + [row] * width
    right_to_left_path_z = optimal_path_z + [target_altitude] * width

    # Calculate cost for both path choices
    cost_left_to_right = compute_path_cost(left_to_right_path_x, left_to_right_path_y, left_to_right_path_z, horizontal_cost, vertical_cost, yaw_cost)
    cost_right_to_left = compute_path_cost(right_to_left_path_x, right_to_left_path_y, right_to_left_path_z, horizontal_cost, vertical_cost, yaw_cost)

    # Choose the path with the lower cost
    if cost_left_to_right <= cost_right_to_left:
        optimal_path_x.extend(left_to_right_x)
        optimal_path_y.extend([row] * width)
        optimal_path_z.extend([target_altitude] * width)
        previous_end = width - 1  # End at right side
    else:
        optimal_path_x.extend(right_to_left_x)
        optimal_path_y.extend([row] * width)
        optimal_path_z.extend([target_altitude] * width)
        previous_end = 0  # End at left side

# Output the optimal path
optimal_path = list(zip(optimal_path_x, optimal_path_y, optimal_path_z))
print("Optimal Path Coordinates (x, y, z):", optimal_path)

drone_altitude = 20  # Altitude above the slope surface
step_size = 50       # Distance between passes in the lawnmower pattern
prev=0
rotrang=[]
for i in range(0,length):
    a= sigmoid_heights[i]-sigmoid_heights[prev]
    if a>2:
        rotrang.append(i)
        camrot[i]= sigmoid_heights[i]+drone_altitude
        # print(camrot[i])
    # print(a)
    prev=i

# print(camrot[67])

for i in range(59,75):
    # a=math.sqrt((67-i)**2 + (72.27116333-camrot[i][1])**2)
    a=(67-i)**2 + (72.27116333-camrot[i][1])**2
    b=math.sqrt(a)
    print(b)
    


# Create a grid based on the slope matrix dimensions
x = np.arange(width)
y = np.arange(length)
X, Y = np.meshgrid(x, y)
Z = slope_matrix

# Initialize lists to store the drone path
drone_path_x = []
drone_path_y = []
drone_path_z = []

# # Generate the lawnmower pattern
# for i in range(length):
#     # Alternating direction for lawnmower pattern
#     if i % 2 == 0:
#         x_path = np.arange(0, width, step_size)
#     else:
#         x_path = np.arange(width - 1, -1, -step_size)
    
#     for x_pos in x_path:
#         drone_path_x.append(x_pos)
#         drone_path_y.append(i)
#         # Drone flies at a fixed altitude above the slope surface
#         drone_path_z.append(slope_matrix[i, x_pos] + drone_altitude)

# Plot the slope and the drone path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.plot_surface(X, Y, camrot, cmap='magma', alpha=0.6)
ax.plot(optimal_path_x, optimal_path_y,optimal_path_z, color='red', marker='o', markersize=1, label='Drone Path')
ax.set_xlabel('Width')
ax.set_ylabel('Length')
ax.set_zlabel('Height')
ax.set_title("Drone Survey of Sigmoid Slope with Lawn Mower Pattern")
plt.legend()
plt.show()
