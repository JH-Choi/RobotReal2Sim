import numpy as np
import genesis as gs
import open3d as o3d
from gs_processor import GSProcessor


gs_path = '/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/tnt/playroom/point_cloud/iteration_30000/point_cloud.ply'
do_visualization = True
sp = GSProcessor()
params = sp.load_ply(gs_path)
pts = params['means3D']
tgt_raw = o3d.geometry.PointCloud()
tgt_raw.points = o3d.utility.Vector3dVector(pts)

# src_raw.paint_uniform_color([0, 1, 0])
tgt_raw.paint_uniform_color([1, 0, 0])


if do_visualization:
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([src_raw, tgt_raw, coordinate])  # type: ignore
    o3d.visualization.draw_geometries([tgt_raw, coordinate])  # type: ignore



