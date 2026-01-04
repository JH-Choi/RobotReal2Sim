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
from genesis.ext import urdfpy

from gs_processor import GSProcessor

def trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    return o3d_mesh

def mesh_poses_to_pc(poses, meshes, offsets, num_pts, scales, pcd_dict, pcd_name=None):
    try:
        assert poses.shape[0] == len(meshes)
        assert poses.shape[0] == len(offsets)
        assert poses.shape[0] == len(num_pts)
        assert poses.shape[0] == len(scales)
    except:
        raise RuntimeError('poses and meshes must have the same length')

    N = poses.shape[0]
    all_pc = []
    for index in range(N):
        mat = poses[index]
        if pcd_name is None or pcd_name not in pcd_dict or len(pcd_dict[pcd_name]) <= index:
            mesh = copy.deepcopy(meshes[index])
            mesh.scale(scales[index], center=np.array([0, 0, 0]))
            sampled_cloud = mesh.sample_points_poisson_disk(number_of_points=num_pts[index])
            cloud_points = np.asarray(sampled_cloud.points)
            if pcd_name not in pcd_dict:
                pcd_dict[pcd_name] = []
            pcd_dict[pcd_name].append(cloud_points)
        else:
            cloud_points = pcd_dict[pcd_name][index]
        
        tf_obj_to_link = offsets[index]
        mat = mat @ tf_obj_to_link
        transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
        all_pc.append(transformed_points)

    all_pc = np.concatenate(all_pc, axis=0)
    return all_pc


def find_link_indices(entity, names):
    if not names:
        return []
    link_indices = list()
    for link in entity.links:
        print('link.name: ', link.name)
        flag = False
        for name in names:
            if name in link.name:
                flag = True
        if flag:
            link_indices.append(link.idx - entity.link_start)
    return link_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--out_pcd_file", default="tmp.ply")
    args = parser.parse_args()


    urdf_file = "/mnt/hdd/code/Dongki_project/Genesis3DGS/data/marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
    initial_joint_angles_right = np.deg2rad([-90,-75,90,-90,-75,0,-20])
    initial_joint_angles_left = np.deg2rad([90,-75,-90,-90,75,0,20])

    scene_gs_input_path = "/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/genesis/IMG_0701/point_cloud/iteration_30000/point_cloud.ply"
    sp = GSProcessor()
    params = sp.load(scene_gs_input_path)
    # params = sp.crop(params, robot_bbox)

    pts = params['means3D']
    if pts.device != 'cpu':
        pts = pts.cpu().numpy()
    tgt_raw = o3d.geometry.PointCloud()
    tgt_raw.points = o3d.utility.Vector3dVector(pts)
    # write to ply
    o3d.io.write_point_cloud("tmp.ply", tgt_raw)
    import pdb; pdb.set_trace()

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
    )

    robot_model = urdfpy.URDF.load(urdf_file)
    # for link in robot_model.links: print('link.name: ', link.name)
    # print('urdf_model.links: ', len(robot_model.links))
    # link_names = []
    # for link in robot_model.links:
    #     link_names.append(link.name)

    link_names = [
        'Base_Mount',
        'Base_R', 'Base_L',
        'Link1_L', 'Link1_R', 'Link2_L', 'Link2_R', 'Link3_L', 'Link3_R', 'Link4_L', 'Link4_R', 'Link5_L', 'Link5_R', 'Link6_L', 'Link6_R', 'Link7_L', 'Link7_R',
        'Link8_L', 'Link8_R', 'Link9_L', 'Link9_R', 
        'Custom_Adapter_L', 'Custom_Adapter_R',
        'Pika_Gripper_Base_L', 'Pika_Gripper_Base_R',
        # 'Gripper_Tip_L', 'Gripper_Tip_R',
    ]

    pcd_dict = {}
    meshes = {}
    scales = {}
    offsets = {}
    prev_offset = np.eye(4)
    for link in robot_model.links:
        if link_names is not None and link.name not in link_names:
            continue
        if len(link.collisions) > 0:
            collision = link.collisions[0]
            prev_offset = collision.origin
            if collision.geometry.mesh != None:
                if len(collision.geometry.mesh.meshes) > 0:
                    mesh = collision.geometry.mesh.meshes[0]
                    meshes[link.name] = trimesh_to_open3d(mesh)
                    scales[link.name] = collision.geometry.mesh.scale[0] if collision.geometry.mesh.scale is not None else 1.0
                    offsets[link.name] = prev_offset
        offsets[link.name] = prev_offset

    ########################## build ##########################
    scene.build()

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
    scene.step()

    link_idx_ls = find_link_indices(robot, link_names)
    qpos = robot.get_qpos()
    links_pos, links_quat = robot.forward_kinematics(qpos, links_idx_local=link_idx_ls)
    scene.step()

    rot_mat = gu.quat_to_R(links_quat)
    # concat rot_mat and links_pos
    link_pose_ls = torch.cat([rot_mat, links_pos.unsqueeze(-1)], dim=-1) 
    link_pose_ls = torch.cat([link_pose_ls, torch.zeros((link_pose_ls.shape[0], 1, 4))], dim=1) 
    link_pose_ls[:, 3, 3] = 1.0 
    link_pose_ls = link_pose_ls.cpu().numpy()

    meshes_ls = [meshes[link_name] for link_name in link_names]
    offsets_ls = [offsets[link_name] for link_name in link_names]
    scales_ls = [scales[link_name] for link_name in link_names]

    num_pts = [1000] * len(link_names)
    pcd_name = args.out_pcd_file
    pcd = mesh_poses_to_pc(poses=link_pose_ls, meshes=meshes_ls, offsets=offsets_ls, 
    num_pts=num_pts, scales=scales_ls, pcd_dict=pcd_dict, pcd_name=pcd_name)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(pcd_name, pcd_o3d)

if __name__ == "__main__":
    main()
