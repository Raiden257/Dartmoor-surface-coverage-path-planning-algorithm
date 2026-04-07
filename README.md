# Terrain Path Planning and Sampling

A Python project for **terrain-based path planning and surface coverage** using digital elevation data.  
The system reads **ASCII terrain tiles**, constructs a height map, and generates coverage paths using spatial sampling and planning algorithms.

The project includes implementations of:

- **Poisson Disk Sampling** for spatially distributed path points
- **Mitchell’s Best Candidate Algorithm** for uniform 3D sampling
- **Lawnmower coverage path planning**
- **KD-Tree–based distance constraints**
- **Terrain visualization with 3D plotting**

These techniques are useful for applications such as:

- Autonomous drone mapping
- Surface inspection
- Terrain surveying
- Coverage path planning
- Environmental monitoring

---

# Features

## Terrain Tile Processing

- Reads `.asc` terrain files (Digital Elevation Model format)
- Supports loading multiple terrain tiles
- Handles missing data (`NODATA_value`)
- Provides terrain height interpolation
- Allows terrain visualization

---

## Sampling Algorithms

### Poisson Disk Sampling

Generates evenly spaced points across the terrain while enforcing a minimum distance constraint.

**Properties**

- Uniform spatial distribution
- Avoids clustering
- Uses KDTree for fast nearest neighbor checks

---

### Mitchell’s Best Candidate Algorithm

Generates candidate points and selects the point that maximizes the minimum distance to existing points.

**Benefits**

- Produces well distributed points
- Works efficiently for large spaces
- Good approximation of blue-noise sampling

---

# Path Planning Methods

## Lawnmower Coverage Pattern

Points are arranged in a **zig-zag scanning path** across the terrain.

This method is widely used in:

- UAV surveying
- robotic coverage planning
- agricultural scanning
- environmental monitoring

The algorithm alternates traversal direction row-by-row to reduce travel distance.

---

## Decision Tree Assisted Traversal

A simple **Decision Tree classifier** estimates terrain slope direction and assists in selecting traversal direction.

---

## Clearance and Height Constraints

Paths are generated while respecting:

- Minimum distance between points
- Maximum terrain height difference
- Safe altitude above terrain

KDTree structures are used to efficiently enforce spatial constraints.

---

# Visualization

The project includes 3D visualization tools using **Matplotlib**.

Features include:

- Terrain surface rendering
- Overlaying generated paths
- Comparing different path planning algorithms
- Visualizing sampling distributions

---

# Project Structure
├── terrain.py
│ Terrain tile loading and interpolation
│
├── test2.py
│ Sampling algorithms
│ Path planning algorithms
│ Visualization utilities
│
├── dataset/
│ Terrain tile files (.asc)
│
├── testset.txt
│ Example height map
│
└── README.md


---

# Installation

Install required dependencies:

```bash
pip install numpy matplotlib scipy scikit-learn
```
Usage

Run the main experiment script:

```python test2.py```

The script will:

Load terrain height data
Generate sampling points using Mitchell's algorithm
Generate a surface coverage path
Generate a Poisson disk coverage path
Visualize both paths on the terrain surface
Output

Generated paths can be exported to files:

2d_lwnmwr_scld_to_3d22.txt
mitchells_best_path22.txt

Each file contains 3D coordinates representing the generated coverage paths.

##Algorithms Used:
- Poisson Disk Sampling\n
- Mitchell’s Best Candidate Algorithm
- Lawnmower Coverage Path Planning
- KDTree nearest neighbor search
- Decision Tree classification
- 3D terrain interpolation
- Applications

## This project can be applied to:

- UAV terrain scanning
- autonomous inspection
- geographic surveying
- robotic coverage planning
- environmental mapping
- Future Improvements

## Potential enhancements include:

- Obstacle avoidance
- Adaptive sampling density
- GPU acceleration
- Real-time path planning
- Integration with ROS for robotics applications
