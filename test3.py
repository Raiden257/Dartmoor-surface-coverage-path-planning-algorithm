import numpy as np
from scipy.ndimage import label
import test2, statistics as stats

def calculate_slope(grid):
    """
    Calculates the slopes between neighboring points in a 2D grid.
    
    Parameters:
    grid (2D list or np.ndarray): 2D array where each element is a z-value.

    Returns:
    slope_map (2D np.ndarray): 2D array with slope magnitudes.
    """
    grid = np.array(grid)
    rows, cols = grid.shape

    # Initialize slope_map to store maximum slope for each point
    slope_map = np.zeros_like(grid, dtype=float)

    # Directions for neighbors: (dy, dx, distance)
    directions = [
        (-1, 0, 1),  # Up
        (1, 0, 1),   # Down
        (0, -1, 1),  # Left
        (0, 1, 1),   # Right
        (-1, -1, np.sqrt(2)),  # Up-Left (diagonal)
        (-1, 1, np.sqrt(2)),   # Up-Right (diagonal)
        (1, -1, np.sqrt(2)),   # Down-Left (diagonal)
        (1, 1, np.sqrt(2))     # Down-Right (diagonal)
    ]

    for y in range(rows):
        for x in range(cols):
            max_slope = 0  # Initialize maximum slope for this point

            for dy, dx, distance in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:  # Check bounds
                    dz = grid[ny, nx] - grid[y, x]  # Elevation difference
                    slope = abs(dz) / distance  # Slope = elevation change / distance
                    max_slope = max(max_slope, slope)  # Update max slope

            slope_map[y, x] = max_slope

    return slope_map

def designate_areas(slope_map, segment_size):
    """
    Automatically designates areas based on slope values and segment size.
    
    Parameters:
    slope_map (2D np.ndarray): 2D array with slope magnitudes.
    segment_size (float): Size of each slope segment.
    
    Returns:
    designation_map (2D np.ndarray): 2D array with designations.
    """
    # Calculate the maximum slope to determine the number of segments
    max_slope = np.max(slope_map)
    num_segments = int(np.ceil(max_slope / segment_size))

    # Assign designations based on slope ranges
    designation_map = np.zeros_like(slope_map, dtype=int)
    for i in range(num_segments):
        lower_bound = i * segment_size
        upper_bound = (i + 1) * segment_size
        designation_map[(slope_map >= lower_bound) & (slope_map < upper_bound)] = i

    return designation_map


import matplotlib.pyplot as plt

def plot_designation1(designation_map,p1z1=None,p1z2=None,p2z1=None,p2z2=None):
    """
    Plots the designated areas on a grid.
    
    Parameters:
    designation_map (2D np.ndarray): 2D array with area designations.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(designation_map, cmap='viridis', origin='upper')
    x11,y11,z11=zip(*p1z1)
    x12,y12,z12=zip(*p1z2)
    x21,y21,z21=zip(*p2z1)
    x22,y22,z22=zip(*p2z2)
    plt.scatter(x11,y11,c='red',s=2)
    plt.scatter(x12,y12,c='orange',s=2)
    plt.scatter(x21,y21,c='white',s=2)
    plt.scatter(x22,y22,c='yellow',s=2)
    plt.colorbar(label="magnitude of slope")
    plt.title("Pen y fan terrain so0023")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_designation(designation_map):
    """
    Plots the designated areas on a grid.
    
    Parameters:
    designation_map (2D np.ndarray): 2D array with area designations.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(designation_map, cmap='viridis', origin='upper')
    plt.colorbar(label="Designation")
    plt.title("Area Designations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    

def segment_map(slope_map, segment_size):
    """
    Segments the map into areas based on slope ranges and identifies their bounds.

    Parameters:
    slope_map (2D np.ndarray): 2D array with slope magnitudes.
    segment_size (float): Size of each slope segment.

    Returns:
    segments (list of dict): List of segments, each containing:
        - 'range': (min_slope, max_slope)
        - 'bounds': [(x_min, x_max, y_min, y_max)] for each area in the segment
    """
    max_slope = np.max(slope_map)
    num_segments = int(np.ceil(max_slope / segment_size))
    labeled_segments = np.zeros_like(slope_map, dtype=int)

    label_id = 1
    label_colors = {}

    for i in range(num_segments):
        lower_bound = i * segment_size
        upper_bound = (i + 1) * segment_size

        # Mask for the current slope range
        mask = (slope_map >= lower_bound) & (slope_map < upper_bound)

        # Find connected components in the mask
        labeled_array, num_features = label(mask)

        for feature_id in range(1, num_features + 1):
            region_mask = labeled_array == feature_id
            labeled_segments[region_mask] = label_id
            label_colors[label_id] = (lower_bound, upper_bound)
            label_id += 1

    return labeled_segments, label_colors

def print_segments(segments):
    """
    Prints the slope segments and their bounds.

    Parameters:
    segments (list of dict): List of segments with slope ranges and bounds.
    """
    for segment in segments:
        slope_range = segment['range']
        bounds = segment['bounds']
        print(f"Slope Range: {slope_range}")
        print(f"Bounds: X ({bounds[0]} to {bounds[1]}), Y ({bounds[2]} to {bounds[3]})\n")

def plot_segments(labeled_segments, label_colors):
    """
    Plots the segmented areas.

    Parameters:
    labeled_segments (2D np.ndarray): Array where each region is assigned a unique label.
    label_colors (dict): Dictionary mapping labels to slope ranges.
    """
    # Create a discrete colormap for the labeled regions
    num_labels = np.max(labeled_segments)
    cmap = plt.get_cmap("tab20", num_labels)

    plt.figure(figsize=(8, 8))
    plt.imshow(labeled_segments, cmap=cmap, origin="upper")
    plt.colorbar(ticks=range(1, num_labels + 1), label="Segment ID")
    plt.title("Segmented Areas by Slope")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Add slope range annotations for each label
    for label_id, slope_range in label_colors.items():
        print(f"Label {label_id}: Slope range {slope_range}")

    plt.show()


# Example usage
if __name__ == "__main__":
    with open("pen y fan 2d array3.txt",'r') as f: #  pen y fan 2d array3
        arr2dtile=[[]]
        for line in f:
            line_bits=[float(item.strip("[]")) for item in line.strip().split(',') if item != '']
            arr2dtile.append(line_bits.copy())
        arr2dtile.pop(0)

    a=np.array(arr2dtile,dtype=float)

    slope_map = calculate_slope(arr2dtile)
    designation_map = designate_areas(slope_map,0.5)
    
    path2, path=test2.read_from_file("2d_lwnmwr_scld_to_3d22.txt","Mitchells_best_path22.txt")
    
    
    # seg,col=segment_map(slope_map,0.5)
    # plot_segments(seg,col)

    # print("Slope Map:")
    # plot_designation(slope_map)
    # print(slope_map)
    # print("\nDesignation Map:")
    # print(designation_map)
    
    z1=[0,0]
    p1z1=[]
    p2z1=[]
    z2=[0,0]
    p1z2=[]
    p2z2=[]
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
    
    for p in path2:
        if 200 <= p[0] <= 320 and 372<=p[1]<=499:
            z1[0]+=1
            p1z1.append(p)
        if 343 <= p[0] <= 452 and 360<=p[1]<=499:
            z2[0]+=1
            p2z1.append(p)

    for p in path:
        if 200 <= p[0] <= 320 and 372<=p[1]<=499:
            z1[1]+=1
            p1z2.append(p)
        if 343 <= p[0] <= 452 and 360<=p[1]<=499:
            z2[1]+=1
            p2z2.append(p)
    
    plot_designation1(slope_map,p1z1,p2z1,p1z2,p2z2)
    # print(p1z1)
    # print(z1,z2)
    
    
    with open("yeag.txt",'r') as f2:
        data=[[]]
        for line in f2:
            line_bits=[int(item.strip().strip('[]')) for item in line.strip().split(',') if item != '']
            data.append(line_bits.copy())
        data.pop(0)
    print(data[0])
    
    # Extract the values for each zone
    pa1zo1=[entry[0]for entry in data]
    pa2zo1=[entry[1]for entry in data]
    pa1zo2=[entry[2]for entry in data]
    pa2zo2=[entry[3]for entry in data]
    # print(pa1zo1,pa1zo2,pa2zo1,pa2zo2)

    stdz1=np.std(pa2zo1)
    stdz2=np.std(pa2zo2)
    labels=['Path 1 Zone 1', 'Path 2 Zone 1','Path 1 Zone 2', 'Path 2 Zone 2']
    print(stdz1,stdz2)
    # x=np.array([1,2,3,4])
    x=np.array(labels)
    y=np.array([pa1zo1[0],stats.mean(pa2zo1),pa1zo2[0],stats.mean(pa2zo2)])
    print(stats.mean(pa2zo1),stats.mean(pa2zo2))
    e=np.array([0,stdz1,0,stdz2])
    plt.errorbar(x,y,e,linestyle='None',marker='o',capsize=20)
    plt.title("Standard Deviation of path points within Each Zone")
    plt.show()
    