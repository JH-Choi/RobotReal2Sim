import os
import argparse
from pathlib import Path
import numpy as np
import torch
import copy
import open3d as o3d
import genesis as gs
from sklearn.neighbors import NearestNeighbors

import genesis.utils.geom as gu

def geom_verts_to_pc(geoms, pcd_dict, pcd_name=None, num_samples_per_geom=2000):
    """
    Extract point cloud by sampling points on geometry surfaces using the physics engine's transformations.
    """
    all_pc = []
    for geom in geoms:
        print(f'geom.link.name: {geom.link.name}')
        # Skip base links
        # if geom.link.name == 'Base_Mount' or geom.link.name == 'Base_R' or geom.link.name == 'Base_L':
        #     continue

        # Get the trimesh object for this geometry
        tmesh = geom.get_trimesh()

        # Sample points on the mesh surface
        sampled_points = tmesh.sample(count=num_samples_per_geom)

        # Get the geometry's transformation (position and quaternion)
        pos = geom.get_pos()
        quat = geom.get_quat()

        # Convert to numpy if tensors
        if hasattr(pos, 'cpu'):
            pos = pos.cpu().numpy()
        if hasattr(quat, 'cpu'):
            quat = quat.cpu().numpy()

        # Handle batched environments (take first environment if batched)
        if pos.ndim == 2:
            pos = pos[0]
        if quat.ndim == 2:
            quat = quat[0]

        # Transform sampled points to world frame
        transformed_points = gu.transform_by_trans_quat(sampled_points, pos, quat)

        all_pc.append(transformed_points)

    all_pc = np.concatenate(all_pc, axis=0)
    return all_pc




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--out_pcd_file", default="tmp_LR.ply")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of points to sample per geometry")
    args = parser.parse_args()


    urdf_file = "/mnt/hdd/code/Dongki_project/Genesis3DGS/data/marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
    initial_joint_angles_right = np.deg2rad([-90,-75,90,-90,-75,0,-20])
    initial_joint_angles_left = np.deg2rad([90,-75,-90,-90,75,0,20])

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file,
            fixed=True,
            merge_fixed_links=False,
        ),
        vis_mode="collision",
    )

    ########################## build ##########################
    scene.build()

    ########################## set joint positions ##########################
    left_joint_names = [
        'Joint1_L', "Joint2_L", "Joint3_L", "Joint4_L", "Joint5_L", "Joint6_L", "Joint7_L",
    ]
    right_joint_names = [
        'Joint1_R', "Joint2_R", "Joint3_R", "Joint4_R", "Joint5_R", "Joint6_R", "Joint7_R",
    ]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in left_joint_names + right_joint_names]

    q_arms = np.concatenate([initial_joint_angles_left,
                            initial_joint_angles_right])   # shape (14,)

    robot.set_dofs_position(q_arms, dofs_idx)

    ########################## step the simulation ##########################
    scene.step()  # This updates the geometry positions and vertices

    ########################## get geometry point cloud ##########################
    # Get all collision geometries from the robot
    geoms = robot.geoms
    pcd_dict = {}

    pcd_name = args.out_pcd_file
    num_samples_per_geom = args.num_samples
    pcd = geom_verts_to_pc(geoms, pcd_dict, pcd_name, num_samples_per_geom=num_samples_per_geom)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(pcd_name, pcd_o3d)


    if args.vis:
        import time
        time.sleep(1000)

if __name__ == "__main__":
    main()