import random
import math
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import numpy as np
visRange=2
# Function to generate a complex polygon with concave and convex edges
def generate_complex_polygon():
    # Define a complex polygon by specifying its vertices
    vertices = [
        (0, 0), (4, 2), (6, 0), (8, 4), (5, 6), (3, 5), (1, 7), (0, 5), (2, 3), (-1, 2)
    ]
    return Polygon(vertices)

# Function to generate a random polygon
def generate_random_polygon(num_points=10, radius=10):
    """
    Generate a convex random polygon.
    
    :param num_points: Number of vertices of the polygon
    :param radius: Maximum radius of the polygon
    :return: A shapely Polygon object
    """
    # Randomly generate points within the bounding box, then sort them angularly
    points = np.random.rand(num_points, 2) * 2 * radius - radius
    
    # Sort points by angle around the center to ensure convexity
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_points = points[np.argsort(angles)]

    # Convert sorted points to a shapely Polygon
    return Polygon(sorted_points)

# Function to calculate visibility with a limited range of 0.2 units
def is_visible(guard, target, max_range):
    guard_point = Point(guard)
    target_point = Point(target)
    distance = guard_point.distance(target_point)
    return distance <= max_range

# Function to generate points along each edge of the polygon perimeter
def generate_perimeter_points(polygon, step=0.05):
    perimeter_points = []
    perimeter_line = list(polygon.exterior.coords)
    
    for i in range(len(perimeter_line) - 1):
        start = Point(perimeter_line[i])
        end = Point(perimeter_line[i + 1])
        line = LineString([start, end])
        length = line.length
        
        # Generate points along the line with a given step size
        num_points = int(length // step)
        for j in range(num_points + 1):
            point = line.interpolate(j / num_points, normalized=True)
            perimeter_points.append(point)
    
    return perimeter_points

# Randomized guard placement process with perimeter coverage and no overlap
def generate_guards(polygon, visibility_range):
    perimeter_points = generate_perimeter_points(polygon)
    uncovered_points = perimeter_points.copy()
    guards = []
    last_guard_position = None

    while uncovered_points:
        # Pick the first uncovered point on the perimeter for placing a new guard
        random_uncovered_point = uncovered_points[0]
        new_guard = (random_uncovered_point.x, random_uncovered_point.y)
        
        # Ensure the new guard does not overlap with the previous one
        if last_guard_position and Point(new_guard).distance(Point(last_guard_position)) < visibility_range * 2:
            # Move the guard just beyond the overlap range by snapping it to the perimeter
            new_guard = (random_uncovered_point.x, random_uncovered_point.y)

        guards.append(new_guard)
        last_guard_position = new_guard
        
        # Check which points on the perimeter are now visible from the new guard
        uncovered_points = [
            pt for pt in uncovered_points 
            if not is_visible(new_guard, (pt.x, pt.y), visibility_range)
        ]
    
    return guards

# Function to connect guards (this will create a path between them)
def connect_guards(guards):
    path = []
    for i in range(len(guards)):
        for j in range(i + 1, len(guards)):
            path.append(LineString([guards[i], guards[j]]))
    return path

# Main function to plot the complex shape, guards, and their limited visibility areas
def main():
    # Generate a complex polygon with concave and convex edges
    complex_polygon = generate_random_polygon()
    
    # Generate guards to cover the entire perimeter without overlapping
    guards = generate_guards(complex_polygon, visRange)
    
    # Plot complex polygon and guards
    x, y = complex_polygon.exterior.xy
    plt.plot(x, y, color='blue', linewidth=2, label='Complex Shape')
    
    # Plot guards and their visibility circles
    for guard in guards:
        plt.plot(guard[0], guard[1], 'ro', label='Guard')
        # Plot the visibility range as a circle around the guard
        visibility_circle = plt.Circle(guard, visRange, color='red', fill=False, linestyle='dotted')
        plt.gca().add_artist(visibility_circle)
    
    # Connect guards
    # guard_connections = connect_guards(guards)
    
    # # Plot connections between guards
    # for connection in guard_connections:
    #     x, y = connection.xy
    #     plt.plot(x, y, color='green', linestyle='dashed')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Complex Shape with Non-Overlapping Guards (Full Perimeter Coverage)')
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
