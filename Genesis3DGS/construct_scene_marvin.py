import os
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
import torch
import copy
import open3d as o3d
import genesis as gs
import genesis.utils.geom as gu
from sklearn.neighbors import NearestNeighbors
from gs_processor import GSProcessor
from icp_utils import preprocess_for_features, global_registration_ransac, \
    transform_copy, compute_ransac_inliers, refine_with_icp
from colormap import colormap


def visualize(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud) -> None:
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([src, tgt, coordinate])  # type: ignore

def visualize_list(pcd_list) -> None:
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(pcd_list + [coordinate])  # type: ignore

def get_bbox(pcd, eps=0.0):
    xmin, ymin, zmin = pcd.min(axis=0)
    xmax, ymax, zmax = pcd.max(axis=0)
    return [[xmin - eps, xmax + eps], [ymin - eps, ymax + eps], [zmin - eps, zmax + eps]]


def geom_verts_to_pc(geoms, sub_name, num_samples_per_geom=2000):
    """
    Extract point cloud by sampling points on geometry surfaces using the physics engine's transformations.
    """
    all_pc = []
    for geom in geoms:
        print(f'geom.link.name: {geom.link.name}')
        if sub_name not in geom.link.name:
            continue
        # skip base links
        if 'Base' in geom.link.name:
            continue
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


def ransac_icp():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    # parser.add_argument("--out_pcd_file", default="tmp.ply")
    # parser.add_argument("--num_samples", type=int, default=1000, help="Number of points to sample per geometry")
    args = parser.parse_args()

    urdf_file = "/mnt/hdd/code/Dongki_project/Genesis3DGS/data/marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
    initial_joint_angles_right = np.deg2rad([-90,-75,90,-90,-75,0,-20])
    initial_joint_angles_left = np.deg2rad([90,-75,-90,-90,75,0,20])

    voxel_size = 0.005
    icp_mode = "point_to_plane"
    max_ransac_iters = 100000
    ransac_n = 4
    confidence = 0.999
    robot_bbox = [[-0.06232, 0.05], [-0.05, 0.05], [0.0, 0.05]]

    do_visualization = True
    robot_file = 'tmp_LR.ply'
    # scene_gs_input_path = "/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/genesis/IMG_0701_aligned_v1/point_cloud/iteration_30000/point_cloud.ply"
    scene_gs_input_path = "/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/genesis/IMG_0701/point_cloud/iteration_30000/point_cloud.ply"

    ########################## init ##########################
    gs.init(backend=gs.gpu)
    ########################## create a scene ##########################
    scene = gs.Scene(
        show_viewer=True,
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

    # skip base links
    geoms = robot.geoms
    pcd_L = geom_verts_to_pc(geoms, '_L', num_samples_per_geom=2000)
    bbox_L = get_bbox(pcd_L, eps=0.04)
    pcd_R = geom_verts_to_pc(geoms, '_R', num_samples_per_geom=2000)
    bbox_R = get_bbox(pcd_R, eps=0.04)
    print('bbox_L: ', bbox_L, 'bbox_R: ', bbox_R)

    pcd = trimesh.load(robot_file)
    xmin, ymin, zmin = pcd.vertices.min(axis=0)
    xmax, ymax, zmax = pcd.vertices.max(axis=0)
    # print('xmin, ymin, zmin: ', xmin, ymin, zmin)
    # print('xmax, ymax, zmax: ', xmax, ymax, zmax)
    src_raw = o3d.geometry.PointCloud()
    src_raw.points = o3d.utility.Vector3dVector(pcd.vertices)

    sp = GSProcessor()
    params = sp.load(scene_gs_input_path)
    # params = sp.crop(params, [[[xmin, xmax], [ymin, ymax], [zmin, zmax]]])

    import pdb; pdb.set_trace()
    params = sp.crop(params, [bbox_L, bbox_R])

    pts = params['means3D']
    if pts.device != 'cpu':
        pts = pts.cpu().numpy()
    tgt_raw = o3d.geometry.PointCloud()
    tgt_raw.points = o3d.utility.Vector3dVector(pts)

    src_raw.paint_uniform_color([0, 1, 0]) # green
    tgt_raw.paint_uniform_color([1, 0, 0]) # red

    if do_visualization:
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([src_raw, tgt_raw, coordinate])  # type: ignore

    print("\n[2/6] Preprocessing (downsample + normals + FPFH)…")
    src_down, src_fpfh = preprocess_for_features(src_raw, voxel_size)
    tgt_down, tgt_fpfh = preprocess_for_features(tgt_raw, voxel_size)
    print(f"   Downsampled sizes: src={np.asarray(src_down.points).shape[0]}, tgt={np.asarray(tgt_down.points).shape[0]}")

    if do_visualization:
        visualize(src_down, tgt_down)

    print("\n[3/6] Global alignment via RANSAC…")
    ransac_result = global_registration_ransac(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size,
        ransac_n=ransac_n, max_iterations=max_ransac_iters, confidence=confidence,
    )
    print("   RANSAC fitness=%.4f, inlier_rmse=%.6f" % (ransac_result.fitness, ransac_result.inlier_rmse))
    print("   RANSAC transformation:\n", ransac_result.transformation)

    # Stage 3: post-RANSAC overlay on raw clouds
    src_ransac_raw = transform_copy(src_raw, ransac_result.transformation)

    # Stage 4: RANSAC inliers visualization on downsampled clouds
    dist_thresh = 1.5 * voxel_size
    src_down_T = transform_copy(src_down, ransac_result.transformation)
    inliers = compute_ransac_inliers(src_down_T, tgt_down, dist_thresh)
    src_in = src_down_T.select_by_index(np.where(inliers)[0].tolist())
    src_out = src_down_T.select_by_index(np.where(~inliers)[0].tolist())
    src_in.paint_uniform_color((0.2, 0.9, 0.2))
    src_out.paint_uniform_color((1.0, 0.2, 0.2))

    if do_visualization:
        visualize(src_in, tgt_down)
        visualize(src_ransac_raw, tgt_raw)

    print("\n[4/6] Refinement via two-stage ICP (%s)…" % icp_mode)
    icp_coarse, icp_fine = refine_with_icp(src_raw, tgt_raw, ransac_result.transformation, voxel_size, icp_mode=icp_mode)
    print("   ICP (coarse)  fitness=%.4f, inlier_rmse=%.6f" % (icp_coarse.fitness, icp_coarse.inlier_rmse))
    print("   ICP (fine)    fitness=%.4f, inlier_rmse=%.6f" % (icp_fine.fitness, icp_fine.inlier_rmse))
    print("   Final ICP transformation:\n", icp_fine.transformation)
    print("   Final ICP transformation (inverted):\n", np.linalg.inv(icp_fine.transformation))

    if do_visualization:
        # Stage 5: post-ICP coarse
        src_icp_coarse = transform_copy(src_raw, icp_coarse.transformation)
        visualize(src_icp_coarse, tgt_raw)

        # Stage 6: post-ICP fine (final)
        src_icp_final = transform_copy(src_raw, icp_fine.transformation)
        visualize(src_icp_final, tgt_raw)

    result = {
        'gs_to_robo': np.linalg.inv(icp_fine.transformation),
        'bboxs': [bbox_L, bbox_R],
    }
    return result



def segment_robot(x):
    gs_to_robo = x['gs_to_robo']
    bboxs = x['bboxs']

    do_visualization = True
    robot_file = 'tmp_LR.ply'
    scene_gs_input_path = "/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/genesis/IMG_0701/point_cloud/iteration_30000/point_cloud.ply"
    scene_mask_save_path = "scene_mask.npy"

    points = trimesh.load(robot_file)
    points = np.asarray(points.vertices)
    src_raw = o3d.geometry.PointCloud()
    src_raw.points = o3d.utility.Vector3dVector(points)

    sp = GSProcessor()
    params = sp.load(scene_gs_input_path)
    params = sp.rotate(params, gs_to_robo[:3, :3])
    params = sp.translate(params, gs_to_robo[:3, 3])
    pts = params['means3D'].cpu().numpy()

    tgt_raw = o3d.geometry.PointCloud()
    tgt_raw.points = o3d.utility.Vector3dVector(pts)

    src_raw.paint_uniform_color([0, 1, 0])
    tgt_raw.paint_uniform_color([1, 0, 0])

    if do_visualization:
        visualize(src_raw, tgt_raw)

    # robot_bbox = np.array([
    #     [np.min(points[:, 0]) - 0.10, np.max(points[:, 0]) + 0.10],
    #     [np.min(points[:, 1]) - 0.10, np.max(points[:, 1]) + 0.10],
    #     [np.min(points[:, 2]), np.max(points[:, 2]) + 0.10],  # hard stop at z min to leave robot base points to table params
    # ])
    # pts_is_robot_mask = (pts[:, 0] > robot_bbox[0, 0]) & (pts[:, 0] < robot_bbox[0, 1]) & \
    #                     (pts[:, 1] > robot_bbox[1, 0]) & (pts[:, 1] < robot_bbox[1, 1]) & \
    #                     (pts[:, 2] > robot_bbox[2, 0]) & (pts[:, 2] < robot_bbox[2, 1])

    # pts_is_robot_mask = np.zeros(pts.shape[0])
    pts_is_robot_mask = np.zeros(pts.shape[0], dtype=bool)
    for bbox in bboxs:
        x0, x1 = bbox[0]
        y0, y1 = bbox[1]
        z0, z1 = bbox[2]
        in_box = (
            (pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
            (pts[:, 1] >= y0) & (pts[:, 1] <= y1) &
            (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
        )
        pts_is_robot_mask |= in_box

    import pdb; pdb.set_trace()
    pts_robot = pts[pts_is_robot_mask]
    pts_scene = pts[~pts_is_robot_mask]

    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(pts_robot)
    robot_pcd.paint_uniform_color([0, 0, 1])
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pts_scene)
    scene_pcd.paint_uniform_color([1, 0, 0])
    if do_visualization:
        visualize_list([robot_pcd, scene_pcd, src_raw])

    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points)
    _, indices = knn.kneighbors(pts_robot)  # (n_scan_points, 1)
    indices_link = (indices / 2000).astype(np.int32).reshape(-1)
    print('length of indices_link: ', len(indices_link))
    print('indices_link: ', np.unique(indices_link))
    import pdb; pdb.set_trace()


    scan_colors = np.asarray(robot_pcd.colors)
    robot_mask = np.zeros(pts_robot.shape[0], dtype=np.int32)

    for i in range(indices_link.max() + 1):
        mask = indices_link == i
        robot_mask[mask] = i + 3  # skip 0=empty and 1=link_base
        scan_colors[mask] = colormap[i]

    # # offset gripper mask ids by 1
    # robot_mask[robot_mask >= 9] += 1  # skip 9=link_eef, make fingers 10-16 instead of 9-15

    robot_pcd.colors = o3d.utility.Vector3dVector(scan_colors)
    if do_visualization:
        visualize_list([robot_pcd])
        visualize_list([robot_pcd, scene_pcd])

    total_mask_full = np.zeros(pts.shape[0], dtype=np.int32) - 1
    total_mask_full[pts_is_robot_mask] = robot_mask
    total_mask_full = total_mask_full.astype(np.int32)

    np.save(scene_mask_save_path, total_mask_full)
    # sp.save(params, scene_gs_save_path)
    # sp.visualize_gs([scene_gs_save_path], transform=False, merged=False, axis_on=True)

def control_robot_with_gripper(gs_to_robo):
    robot_file = 'tmp.ply'
    total_mask_path = "scene_mask.npy"
    scene_gs_input_path = "/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/genesis/IMG_0701/point_cloud/iteration_30000/point_cloud.ply"

    sp = GSProcessor()
    params_full = sp.load(scene_gs_input_path)
    params_full = sp.rotate(params_full, gs_to_robo[:3, :3])
    params_full = sp.translate(params_full, gs_to_robo[:3, 3])
    total_mask_full = np.load(total_mask_path)
    total_mask_full = torch.from_numpy(total_mask_full).to(params_full['means3D'].device).to(params_full['means3D'].dtype)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    # gs_to_robo = ransac_icp()
    # print('gs_to_robo: ', gs_to_robo)
    # with open('gs_to_robo.txt', 'w') as f:
    #     f.write(str(gs_to_robo))

    result = ransac_icp()
    # with open('gs_to_robo.json', 'w') as f:
    #     json.dump(result, f, indent=4)

    segment_robot(result)
    import pdb; pdb.set_trace()
    # control_robot_with_gripper(gs_to_robo)
