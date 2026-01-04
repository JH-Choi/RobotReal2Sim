import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
ply_path = "./assets/scene/gs/xarm6/xarm6.ply"
plydata = PlyData.read(ply_path)
print(plydata.elements[0].data.shape)
semantic_labels = np.load("./assets/scene/gs/xarm6/xarm6_semantics_gs.npy")
print(semantic_labels.shape)
print(np.unique(semantic_labels))
# [-1  1  2  3  4  5  6  7  8  9 10 11 12 13 15 16]
# check xarm_gs_semantics in constants.py
# -1 => background


crop_ply_path = "./assets/scene/gs/xarm6/xarm6_cropped_arm.ply"
crop_plydata = PlyData.read(crop_ply_path)
print(crop_plydata.elements[0].data.shape)