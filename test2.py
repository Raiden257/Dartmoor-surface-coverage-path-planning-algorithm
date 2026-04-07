import numpy as np
import matplotlib.pyplot as plt
from terrain import TerrainTileCollection,  TerrainTile
from scipy.spatial import KDTree
from sklearn.tree import DecisionTreeClassifier
import random,math

# tile_con = TerrainTileCollection('dataset\pen_y_fan_2m')

# sorted_tiles = sorted(tile_con.tiles, key=lambda x: (x.yllcorner, x.xllcorner))
# tilearray2d = [[]]
# buffer = []

# for j in range(500):
#     # print("row: ", j)
#     for tile in sorted_tiles:
#         # print("| ", tile.ncols," | ", tile.nrows," |")
#         for i in range(tile.ncols):
#             buffer.append(tile.Z[j, i])
#     tilearray2d.append(buffer.copy())
#     buffer.clear()
# # print(tilearray2d)
# tilearray2d.pop(0)
# with open("pen y fan 2d array4.txt", "w") as f:
#     for line in tilearray2d:
#         f.write(f"{line}\n")

# tilearr=[[]]
# t1=TerrainTile('so0122_dtm_2m.asc')
# for j in range(t1.nrows):
#     for i in range(t1.ncols):
#         buffer.append(t1.Z[j,i])
#     tilearr.append(buffer.copy())
#     buffer.clear()
# tilearr.pop(0)

# with open ("testset.txt",'w') as f:
#     for line in tilearr:
#         f.write(f"{line}\n")


# with open("pen y fan 2d array3.txt", "r") as f:
#     arr2dtile = [[]]
#     for line in f:
#         line_bits = [float(item.strip("[]")) for item in line.strip().split(
#             ',') if item != '']  # same line read and split code as tile read section
#         arr2dtile.append(line_bits.copy())  # avoid line reset issues
#     arr2dtile.pop(0)  # remove the first empty array element


with open("testset.txt",'r') as f:
    arr2dtile=[[]]
    for line in f:
        line_bits=[float(item.strip("[]")) for item in line.strip().split(',') if item != '']
        arr2dtile.append(line_bits.copy())
    arr2dtile.pop(0)

a = np.array(arr2dtile, dtype=float)

def sort_to_lawnmower_pattern(path, height_map):
    rows, cols = height_map.shape

    # Group points by their row index
    points_by_row = {row: [] for row in range(rows)}
    for x, y in path:
        points_by_row[x].append((x, y))

    # Sort points within each row by their column index
    sorted_path = []
    for row in range(rows):
        if row in points_by_row:
            if row % 2 == 0:  # Left-to-right for even rows
                sorted_path.extend(sorted(points_by_row[row], key=lambda point: point[1]))
            else:  # Right-to-left for odd rows
                sorted_path.extend(sorted(points_by_row[row], key=lambda point: point[1], reverse=True))

    return sorted_path

def poisson_disk_path(height_map, min_distance=10, max_height_diff=1, k=30):
    rows, cols = height_map.shape
    path = []  # The points selected for the path
    kd_tree = None  # KDTree for efficient nearest neighbor searches

    # Helper function to check if a point is within bounds and height constraints
    def is_valid_point(new_point):
        x, y = new_point
        if not (0 <= x < rows and 0 <= y < cols):
            return False
        if kd_tree is not None:
            distances, _ = kd_tree.query(new_point, k=1)
            if distances < min_distance:
                return False
        return True

    # Helper function to generate candidate points in a circular region
    def generate_candidates(center_point):
        candidates = []
        cx, cy = center_point
        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_distance, 2 * min_distance)
            new_x = cx + radius * math.sin(angle)
            new_y = cy + radius * math.cos(angle)
            candidates.append((int(round(new_x)), int(round(new_y))))
        return candidates

    # Initialize with a random starting point
    start_point = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    path.append(start_point)
    active_list = [start_point]
    kd_tree = KDTree(np.array(path))

    # Poisson Disk Sampling
    while active_list:
        idx = random.randint(0, len(active_list) - 1)  # Randomly pick an active point
        center_point = active_list[idx]
        candidates = generate_candidates(center_point)

        point_added = False
        for candidate in candidates:
            if is_valid_point(candidate):
                path.append(candidate)
                active_list.append(candidate)
                kd_tree = KDTree(np.array(path))  # Update KDTree
                point_added = True
                break  # Stop after adding a valid candidate

        if not point_added:  # If no valid candidates, remove the point from the active list
            active_list.pop(idx)

    path=sort_to_lawnmower_pattern(path,height_map)
    return path


def generate_smooth_path_with_clearance(height_map, min_distance=10, max_height_diff=1):
    rows, cols = height_map.shape
    path = []
    kd_tree = None
    # Simple decision tree with a max depth of 1
    classifier = DecisionTreeClassifier(max_depth=2)

    def is_far_enough(new_point):
        """Check if new_point is at least min_distance away from all points in path."""
        # for p in path:
        #     if euclidean(new_point, p) < min_distance:
        #         return False
        # return True
        if kd_tree is None:
            return True
        distances, _ = kd_tree.query(new_point, k=1)
        return distances >= min_distance

    last_position = None

    for i in range(rows):
        # Calculate slope in the current row to decide movement direction
        slopes = [height_map[i, j + 1] - height_map[i, j]
                  for j in range(cols - 1)]
        avg_slope = sum(slopes) / len(slopes)

        # Train the decision tree based on slope direction
        X_train = np.array(slopes).reshape(-1, 1)
        # 1 for positive slope, 0 for negative or flat
        y_train = [1 if slope > 0 else 0 for slope in slopes]
        classifier.fit(X_train, y_train)

        # Determine movement direction for this row
        # Starting at the left
        if last_position is None or last_position[1] == 0:
            path_direction = 1  # Left-to-right
        else:  # Starting at the right
            path_direction = 0  # Right-to-left

        # Traverse the row in the current direction
        if path_direction == 1:  # Left-to-right
            for j in range(cols):
                if (j == 0 or abs(height_map[i, j] - height_map[i, j - 1]) <= max_height_diff):
                    new_point = (i, j)
                    if is_far_enough(new_point):
                        path.append(new_point)
                        last_position = new_point
        else:  # Right-to-left
            for j in range(cols - 1, -1, -1):
                if (j == cols - 1 or abs(height_map[i, j] - height_map[i, j + 1]) <= max_height_diff):
                    new_point = (i, j)
                    if is_far_enough(new_point):
                        path.append(new_point)
                        last_position = new_point

        # Update KDTree for efficient distance checks
        kd_tree = KDTree(np.array(path))

    return path


def plot_surface_with_path(height_map, path):
    rows, cols = height_map.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x, y = np.meshgrid(x, y)
    z = np.array(height_map)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.3)

    # Overlay path
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    path_z = [height_map[p[0], p[1]]+50 for p in path]

    ax.plot(path_x, path_y, path_z, color='red', linewidth=1, marker='o',
            markersize=2, label='Decision Tree Path with Clearance')
    # ax.scatter(path_x, path_y, path_z, color='blue', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title('Path on Height Map with Clearance Constraints')
    ax.legend()
    plt.show()


def generate_mitchell_points_large_space(
        num_points, x_range, y_range, z_range, threshold_func, num_candidates=10):
    """
    Generate a 3D point distribution using Mitchell's Best Candidate Algorithm with optimization for large spaces.

    Parameters:
        num_points (int): Number of points to generate.
        x_range (tuple): Range for x-coordinate as (min_x, max_x).
        y_range (tuple): Range for y-coordinate as (min_y, max_y).
        z_range (tuple): Range for z-coordinate as (min_z, max_z).
        threshold_func (callable): A function that takes (x, y) and returns the minimum z-value for a point to be valid.
        num_candidates (int): Number of candidates to evaluate at each step.

    Returns:
        points (numpy.ndarray): Array of shape (num_points, 3) with generated 3D points.
    """
    def random_point_above_threshold():
        """Generate a random point above the threshold."""
        while True:
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            # Determine the minimum z-value for this (x, y)
            min_z, max_z = threshold_func(x, y)
            z = np.random.uniform(max(z_range[0], min_z), min(
                max_z, z_range[1]))  # Ensure z is valid
            return np.array([x, y, z])

    # Initialize with an empty list of points
    points = np.empty((0, 3))

    # Generate the first point randomly
    points = np.vstack([points, random_point_above_threshold()])

    # Add the remaining points
    for _ in range(1, num_points):
        best_candidate = None

        # Generate candidates and find the one with the maximum minimum distance
        candidates = np.array([random_point_above_threshold()
                              for _ in range(num_candidates)])
        distances = np.min(
            np.linalg.norm(candidates[:, np.newaxis, :] -
                           points[np.newaxis, :, :], axis=2),
            axis=1,
        )

        # Select the candidate with the maximum minimum distance
        best_candidate = candidates[np.argmax(distances)]
        points = np.vstack([points, best_candidate])

    return points


def plot_3d_points(points, x_range, y_range, z_range):
    """
    Plot 3D points using Matplotlib.

    Parameters:
        points (list of tuples): List of 3D points to plot.
        x_range (tuple): Range for the x-axis.
        y_range (tuple): Range for the y-axis.
        z_range (tuple): Range for the z-axis.
    """
    # Extract x, y, z coordinates
    x_coords, y_coords, z_coords = zip(*points)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', s=20)

    # Set axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set axis limits
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(*z_range)

    # Display the plot
    plt.title('Mitchell\'s Best Candidate 3D Points')
    plt.show()


def theshval(x, y):
    x = int(x)
    y = int(y)
    return a[y, x] + 10, a[y, x] + 20


def generate_lawnmower_pattern2(points, min_distance=0, max_height_diff=10):
    """
    Generate a lawnmower path from points produced by Mitchell's Best Candidate Algorithm.

    Parameters:
        points (numpy.ndarray): Array of generated 3D points (x, y, z).
        x_range (tuple): Range of x-coordinates as (min_x, max_x).
        y_range (tuple): Range of y-coordinates as (min_y, max_y).
        z_range (tuple): Range of z-coordinates as (min_z, max_z).
        min_distance (float): Minimum allowable distance between consecutive path points.
        max_height_diff (float): Maximum height difference between points.

    Returns:
        path (list of tuples): List of points forming the lawnmower path.
    """
    # Ensure unique (x, y) pairs
    unique_xy = np.unique(points[:, :2], axis=0)  # Unique (x, y) pairs
    unique_points = []
    for x, y in unique_xy:
        # Find all points matching the (x, y) pair and keep the one with the minimum z-coordinate
        matching_points = points[(points[:, 0] == x) & (points[:, 1] == y)]
        if len(matching_points) > 0:
            min_z_point = matching_points[np.argmin(matching_points[:, 2])]

            if (a[int(min_z_point[1]), int(min_z_point[0])]) < min_z_point[2]:

                unique_points.append(min_z_point)
    # Convert to NumPy array and sort for lawnmower traversal
    points_sorted = np.array(unique_points)
    points_sorted = points_sorted[np.lexsort(
        (points_sorted[:, 0], points_sorted[:, 1]))]  # Sort by y, then x
    path = []
    kd_tree = None  # KDTree for efficient clearance checks

    # Organize points row by row
    rows = np.unique(points_sorted[:, :2], axis=0)  # Unique y-values

    for row_index, y_value in enumerate(rows):
        # Filter points in the current row
        row_points = points_sorted[np.isclose(
            points_sorted[:, 1], y_value[1], atol=0.05)]
        # print(len(row_points))
        # Sort points left-to-right or right-to-left depending on the row index
        if row_index % 2 == 0:  # Left-to-right
            row_points = row_points[np.argsort(row_points[:, 0])]
        else:  # Right-to-left
            row_points = row_points[np.argsort(row_points[:, 0])[::-1]]

        # Traverse the row
        for point in row_points:
            # Check clearance and height constraints
            if kd_tree is None or is_far_enough(point, kd_tree, min_distance):
                if len(path) == 0 or abs(point[2] - path[-1][2]) <= max_height_diff:
                    path.append(tuple(point))

        # Update KDTree for efficient clearance checks
        if len(path) > 0:
            kd_tree = KDTree(np.array(path))

    path = sort_points_by_path(path, 7)
    return path


def sort_points_by_path(points, z_tolerance=0.1):
    """
    Sort points into a viable path that starts at the lowest Z value, 
    connects points based on distance, and moves up to the next Z range.

    Parameters:
    - points: np.array, shape (n, 3), list of 3D points.
    - z_tolerance: float, range of Z values to consider as part of the same slice.

    Returns:
    - sorted_path: np.array, points ordered to form the path.
    """
    # Step 1: Sort points by Z value
    points = sorted(points, key=lambda p: p[2])  # Sort tuples by z-coordinate

    # Step 2: Group points into Z slices
    z_min = points[0][2]
    slices = []
    current_slice = []

    for point in points:
        if len(current_slice) == 0 or abs(point[2] - current_slice[-1][2]) <= z_tolerance:
            current_slice.append(point)
        else:
            slices.append(current_slice)
            current_slice = [point]
    if current_slice:
        slices.append(current_slice)

    # Step 3: Sort points within each Z slice by nearest neighbor
    def sort_slice(slice_points):
        if len(slice_points) == 1:
            return slice_points  # Single point in the slice, no sorting needed
        # Convert to NumPy array for KDTree
        tree = KDTree(np.array(slice_points))
        visited = set()
        # Start at the first point in the slice
        sorted_slice = [slice_points[0]]
        visited.add(0)

        for _ in range(1, len(slice_points)):
            last_point = np.array(sorted_slice[-1])  # Get the last point
            _, nearest_idx = tree.query(last_point, k=len(slice_points))
            for idx in nearest_idx:
                if idx not in visited:
                    sorted_slice.append(slice_points[idx])
                    visited.add(idx)
                    break
        return sorted_slice

    sorted_slices = [sort_slice(slice_points) for slice_points in slices]

    # Step 4: Combine slices into a single path
    sorted_path = [sorted_slices[0][0]]  # Start at the lowest Z point
    for i in range(len(sorted_slices)):
        sorted_path.extend(sorted_slices[i])
        # If there is another slice, connect the last point in the current slice to the first in the next slice
        if i + 1 < len(sorted_slices):
            current_last = sorted_slices[i][-1]
            next_first = sorted_slices[i + 1][0]
            sorted_path.append(next_first)

    return sorted_path


def is_far_enough(new_point, kd_tree, min_distance):
    """
    Check if new_point is at least min_distance away from all points in path.

    Parameters:
        new_point (numpy.ndarray): The point to check.
        kd_tree (scipy.spatial.KDTree): KDTree of existing points in the path.
        min_distance (float): Minimum allowable distance.

    Returns:
        bool: True if the point is far enough, False otherwise.
    """
    distances, _ = kd_tree.query(new_point, k=1)
    return distances >= min_distance


def plot_surface_with_path2(height_map, path):
    rows, cols = height_map.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x, y = np.meshgrid(x, y)
    z = np.array(height_map)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.3)

    # Overlay the lawnmower path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = [p[2] for p in path]

    ax.plot(path_x, path_y, path_z, color='red', linewidth=0.3, marker='o',
            markersize=0.9, label='surface coverage path plan', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title('Path on Height Map with Clearance Constraints')
    ax.legend()
    plt.show()


def plot_surface_with_path3(height_map, path, path2):
    rows, cols = height_map.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x, y = np.meshgrid(x, y)
    z = np.array(height_map)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.3)

    # Overlay the lawnmower path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = [p[2] for p in path]

    x2 = [p[0] for p in path2]
    y2 = [p[1] for p in path2]
    z2 = [height_map[p[1], p[0]]+30 for p in path2]

    ax.plot(path_x, path_y, path_z, color='red', linewidth=0.5, marker='o',
            markersize=1.4, label='surface coverage path plan', alpha=0.7)
    ax.plot(x2, y2, z2, color='blue', linewidth=0.2, marker='o',
            markersize=0.9, label='map coverage path plan', alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title('Path on Height Map with Clearance Constraints')
    ax.legend()
    plt.show()


def plot_surface_with_path4(height_map, path, path2):
    rows, cols = height_map.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x, y = np.meshgrid(x, y)
    z = np.array(height_map)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.3)

    # Overlay the lawnmower path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = [p[2] for p in path]

    x2 = [p[0] for p in path2]
    y2 = [p[1] for p in path2]
    z2 = [p[2] for p in path2]

    ax.plot(path_x, path_y, path_z, color='red', linewidth=0.3, marker='o',
            markersize=0.9, label='surface coverage path plan', alpha=0.5)
    ax.plot(x2, y2, z2, color='blue', linewidth=0.3, marker='o',
            markersize=0.9, label='map coverage path plan', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title('Path on Height Map with Clearance Constraints')
    ax.legend()
    plt.show()


def save_to_file(path1, path2):

    p3d = []
    for p in path1:
        p3d.append([p[0], p[1], a[p[1], p[0]]+30])
    with open("2d_lwnmwr_scld_to_3d22.txt", 'w') as f:
        for line in p3d:
            f.write(f"{line}\n")

    with open("mitchells_best_path22.txt", 'w') as f:
        for line in path2:
            f.write(f"{line}\n")
    print('done')


def read_from_file(name1, name2):
    with open(name1, 'r') as f:
        p1 = [[]]
        for line in f:
            line_bits = [float(item.strip("[]"))
                         for item in line.strip().split(',') if item != '']
            p1.append(line_bits.copy())
        p1.pop(0)

    with open(name2, 'r') as f2:
        p2 = [[]]
        for line in f2:
            line_bits = [float(item.strip("()"))
                         for item in line.strip().split(',') if item != '']
            p2.append(line_bits.copy())
        p2.pop(0)
    return p1, p2


if __name__ == "__main__":
    xran = (0, 500)
    yran = (0, 500)
    minz = 9999
    maxz = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i, j] > maxz:
                maxz = a[i, j]
            if a[i, j] < minz:
                minz = a[i, j]
    zran = (0, int(maxz))
    zran1 = (int(minz), int(maxz)+50)
    zdist = maxz-minz

    points = generate_mitchell_points_large_space(10000, xran, yran, zran1, theshval, 20)
    path = generate_lawnmower_pattern2(points, 10, 10)
    # path2 = generate_smooth_path_with_clearance(a, 10, 10)
    path2=poisson_disk_path(a,10,10,50)
    
    
    # plot_surface_with_path2(a,path)
    plot_surface_with_path3(a, path, path2)
    
    # yeaj=[]
    # for jes in range (20):
    #     points = generate_mitchell_points_large_space(10000, xran, yran, zran1, theshval, 20)
    #     path = generate_lawnmower_pattern2(points, 10, 10)
    #     z1=[0,0]
    #     z2=[0,0]
        
        # for p in path2:
        #     if 0 <= p[0] <= 139 and 0<=p[1]<=206:
        #         z1[0]+=1
        #     if 252 <= p[0] <= 372 and 179<=p[1]<=319:
        #         z2[0]+=1
                

        # for p in path:
        #     if 0 <= p[0] <= 139 and 0<=p[1]<=206:
        #         z1[1]+=1
                
        #     if 252 <= p[0] <= 372 and 179<=p[1]<=319:
        #         z2[1]+=1
                
    #     for p in path2:
    #         if 200 <= p[0] <= 320 and 372<=p[1]<=499:
    #             z1[0]+=1
                
    #         if 343 <= p[0] <= 452 and 360<=p[1]<=499:
    #             z2[0]+=1
                

    #     for p in path:
    #         if 200 <= p[0] <= 320 and 372<=p[1]<=499:
    #             z1[1]+=1
                
    #         if 343 <= p[0] <= 452 and 360<=p[1]<=499:
    #             z2[1]+=1
                
                
                
    #     yeaj.append([z1.copy(),z2.copy()])
    #     print(jes)
        
    # with open("yeag2.txt",'w') as f2:
    #     for line in yeaj:
    #         f2.write(f"{line}\n")
        
    #     print(yeaj)
    #     print("done jone")

    # save_to_file(path2,path)

    # path2, path=read_from_file("2d_lwnmwr_scld_to_3d.txt","Mitchells_best_path.txt")
    # plot_surface_with_path4(a,path,path2)
    
    # z1=[0,0]
    # p1z1=[]
    # p2z1=[]
    # z2=[0,0]
    # p1z2=[]
    # p2z2=[]
    # for p in path2:
    #     if 0 <= p[0] <= 130 and 0<=p[1]<=139:
    #         z1[0]+=1
    #         p1z1.append(p)
    #     if 252 <= p[0] <= 372 and 179<=p[1]<=319:
    #         z2[0]+=1
    #         p2z1.append(p)

    # for p in path:
    #     if 0 <= p[0] <= 130 and 0<=p[1]<=139:
    #         z1[1]+=1
    #         p1z2.append(p)
    #     if 252 <= p[0] <= 372 and 179<=p[1]<=319:
    #         z2[1]+=1
    #         p2z2.append(p)
    
    # for p in path2:
    #     if 200 <= p[0] <= 320 and 372<=p[1]<=499:
    #         z1[0]+=1
    #         p1z1.append(p)
    #     if 343 <= p[0] <= 452 and 360<=p[1]<=499:
    #         z2[0]+=1
    #         p2z1.append(p)

    # for p in path:
    #     if 200 <= p[0] <= 320 and 372<=p[1]<=499:
    #         z1[1]+=1
    #         p1z2.append(p)
    #     if 343 <= p[0] <= 452 and 360<=p[1]<=499:
    #         z2[1]+=1
    #         p2z2.append(p)
            
    

