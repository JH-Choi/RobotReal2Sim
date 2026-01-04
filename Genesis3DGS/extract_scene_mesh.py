# https://github.com/aCodeDog/OmniPerception/blob/main/LidarSensor/LidarSensor/example/genesis/g1_lidar_taichi_visualization.py
import numpy as np

"""Extract mesh data from Genesis scene for ray casting"""

vertices = []
triangles = []
face_idx = 0

# Ground plane (large)
ground_size = 50.0
ground_verts = np.array([
    [-ground_size, -ground_size, 0], 
    [ground_size, -ground_size, 0], 
    [ground_size, ground_size, 0], 
    [-ground_size, ground_size, 0]
], dtype=np.float32)
vertices.extend(ground_verts)
triangles.extend([[0, 1, 2], [0, 2, 3]])
face_idx += 4



vertices, triangles = np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)