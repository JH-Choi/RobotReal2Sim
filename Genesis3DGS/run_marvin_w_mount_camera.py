import os
import trimesh
import torch
import argparse
import imageio
import cv2
import genesis as gs
import numpy as np
from collections import defaultdict
from gs_processor import GSProcessor
from diff_gaussian_rasterization import GaussianRasterizer
from transform_utils import setup_camera
from apply_transform import build_transform
from gs_utils import alpha_blend_rgba
import genesis.utils.geom as gu



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

 
    urdf_file = "/mnt/hdd/code/Dongki_project/Genesis3DGS/data/marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
    initial_joint_angles_right = np.deg2rad([-90,-75,90,-90,-75,0,-20])
    initial_joint_angles_left = np.deg2rad([90,-75,-90,-90,75,0,20])
  
    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        # viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file,
            fixed=True,
            merge_fixed_links=False,
        ),
        vis_mode="collision",
    )
    ########################## set camera ##########################
    cameras = {}
    cameras["camera_third_person"] = scene.add_camera(
        res=(640, 480),
        GUI=True, fov=30,
        pos=(1.8, 0, 0.9),
        lookat=(0, 0, -0.4),
    )
    # cameras["camera_first_person"] = scene.add_camera(GUI=True, fov=70)

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

    ########################## GS Processor ##########################
    total_mask_path = "scene_mask.npy"
    scene_gs_input_path = "/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/genesis/IMG_0701/point_cloud/iteration_30000/point_cloud.ply"


    sp = GSProcessor()
    # params_full = sp.load(scene_gs_input_path)
    # params_full = sp.rotate(params_full, gs_to_robo[:3, :3])
    # params_full = sp.translate(params_full, gs_to_robo[:3, 3])

    device = 'cuda'
    bg = [0, 0, 0]
    max_sh_degrees = 3
    params_bg = sp.load_ply(scene_gs_input_path)
    render_data = {
        'means3D': params_bg['means3D'],
        'sh_colors': params_bg['sh_colors'],
        'unnorm_rotations': params_bg['rotations'],
        'logit_opacities': params_bg['opacities'],
        'log_scales': params_bg['scales'],
        'means2D': torch.zeros_like(params_bg['means3D'], requires_grad=True), # (N,3)
    }
    render_data = {k: v.to(device) for k, v in render_data.items()}
    # render_data = sp.rotate(render_data, rot_mat)
    # render_data = sp.translate(render_data, trans)
    render_data = {
        'means3D': render_data['means3D'],
        'shs': render_data['sh_colors'],
        'rotations': torch.nn.functional.normalize(render_data['unnorm_rotations'], dim=-1),
        'opacities': torch.sigmoid(render_data['logit_opacities']),
        'scales': torch.exp(render_data['log_scales']),
        'means2D': torch.zeros_like(render_data['means3D'], requires_grad=True), # (N,3)
    }
    
    # render_data = {
    #     'means3D': params_full['means3D'],
    #     'shs': params_full['sh_colors'],
    #     'rotations': torch.nn.functional.normalize(params_full['unnorm_rotations'], dim=-1),
    #     'opacities': torch.sigmoid(params_full['logit_opacities']),
    #     'scales': torch.exp(params_full['log_scales']),
    #     'means2D': torch.zeros_like(params_full['means3D'], requires_grad=True), # (N,3)
    # }
    # render_data = {k: v.to(device) for k, v in render_data.items()}

    geoms = robot.geoms
    link_dict = {}
    base_pos_dict = {}
    base_geom_dict = {}
    for i, geom in enumerate(geoms):
        link_dict[i] = geom.link.name
        base_pos_dict[i] = {'pos': geom.link.get_pos().clone(), 'quat': geom.link.get_quat().clone()}
        tmesh = geom.get_trimesh()
        tmesh = tmesh.sample(count=2000).copy()
        base_geom_dict[i] = gu.transform_by_trans_quat(tmesh, base_pos_dict[i]['pos'].cpu().numpy(), base_pos_dict[i]['quat'].cpu().numpy())

    # link_names = [geom.link.name for geom in geoms]  # or a manual list
    # ref = {}
    # for name in link_names:
    #     link = robot.get_link(name)
    #     ref[name] = {
    #         "p": link.get_pos(),    # (3,)
    #         "q": link.get_quat(),   # (4,) in Genesis order (usually wxyz)
    #     }
    # ref and base_pos_dict are same!

    
    print('link_dict: ', link_dict.keys())
    print('length of link_dict: ', len(link_dict.keys()))

    total_mask_full = np.load(total_mask_path)
    total_mask_full = torch.from_numpy(total_mask_full).to(params_bg['means3D'].device).to(params_bg['means3D'].dtype)
    unique_labels = torch.unique(total_mask_full).cpu().numpy()
    print('total_mask_full: ', total_mask_full.shape)
    print('total_mask_full: ', torch.unique(total_mask_full))
    link_id_list = [i for i in range(3, 25)]

    pts_is_scene_mask = (total_mask_full < 3) | (total_mask_full >= 25)
    pts_is_robot_mask = ~pts_is_scene_mask

    # points_links = {link_dict[i]: render_data['means3D'][total_mask_full == i] for i in link_id_list}
    # quats_links = {link_dict[i]: render_data['rotations'][total_mask_full == i] for i in link_id_list}
    # for i in link_id_list:
    #     pos = pos_dict[i]['pos']
    #     quat = pos_dict[i]['quat']

    output_dir = 'tmp_renders_marvin'
    os.makedirs(output_dir, exist_ok=True)

    ########################## step the simulation ##########################
    scene.step()  # This updates the geometry positions and vertices

    def min_jerk(s):
        # s in [0,1]
        return 10*s**3 - 15*s**4 + 6*s**5

    def lerp(a, b, w):
        return (1.0 - w) * a + w * b

    # --- gains: for very smooth motion, often lower kp and a bit more kv helps ---
    robot.set_dofs_kp(kp=np.array([1500] * len(dofs_idx)), dofs_idx_local=dofs_idx)
    robot.set_dofs_kv(kv=np.array([200]  * len(dofs_idx)), dofs_idx_local=dofs_idx)
    robot.set_dofs_force_range(
        lower=np.array([-87] * len(dofs_idx)),
        upper=np.array([ 87] * len(dofs_idx)),
        dofs_idx_local=dofs_idx,
    )

    # initial pose as start
    q0 = robot.get_dofs_position(dofs_idx)          # current joint positions
    # dq = np.zeros(len(dofs_idx))

    # short range goal (example: +0.48 rad on all dofs, edit per joint as needed)
    q1 = q0 + 0.28 * torch.ones(len(dofs_idx), device='cuda')

    T = 500  # duration (steps) for one segment

    for i in range(1250):
        # 0..249: q0 -> q1
        # 250..499: hold q1
        # 500..749: q1 -> q0
        # 750..1249: hold q0
        if i < T:
            s = i / (T - 1)
            w = min_jerk(s)
            q_des = lerp(q0, q1, w)
            robot.control_dofs_position(q_des, dofs_idx)
        elif i < 2*T:
            robot.control_dofs_position(q1, dofs_idx)
        elif i < 3*T:
            s = (i - 2*T) / (T - 1)
            w = min_jerk(s)
            q_des = lerp(q1, q0, w)
            robot.control_dofs_position(q_des, dofs_idx)
        else:
            robot.control_dofs_position(q0, dofs_idx)

        # optional debugging
        # print('control force:', robot.get_dofs_control_force(dofs_idx))
        # print('internal force:', robot.get_dofs_force(dofs_idx))

        scene.step()

        for k in cameras.keys():
            print(f'camera name: {k}')
            camera_inst = cameras[k]
            # if k == "camera_third_person":
            #     camera_inst.set_pose(
            #         pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
            #         lookat=(0, 0, 0.5),
            #     )
            
            rgb, depth, segmentation, _ = cameras[k].render(rgb=True, depth=True, segmentation=True)
            cv2.imwrite('tmp.png', rgb)

            c2w = camera_inst.transform
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            Ks = camera_inst.intrinsics
            metadata = {
                'w': camera_inst.res[0],
                'h': camera_inst.res[1],
                'k': Ks,
                'w2c': w2c,
                'near': 0.01,
                'far': 100.0,
            }

            cam = setup_camera(metadata['w'], metadata['h'], metadata['k'], 
                    metadata['w2c'], metadata['near'], metadata['far'], bg, 
                    sh_degree=max_sh_degrees, device=device)

            geoms = robot.geoms
            pos_dict = {}
            for j, geom in enumerate(geoms):
                # print(f'geom.link.name: {geom.link.name}, j: {j}')
                # get the geometry's transformation (position and quaternion)
                # pos_dict[j] = {'pos': geom.get_pos(), 'quat': geom.get_quat()}
                pos_dict[j] = {'pos': geom.link.get_pos(), 'quat': geom.link.get_quat()}
                # print(f'geom.link.name: {geom.link.name}, j: {j}, pos: {pos_dict[j]["pos"]}, quat: {pos_dict[j]["quat"]}')

            tot_pcd = []
            new_render_data = render_data.copy()
            for lbl in unique_labels:
                if lbl in link_id_list:
                    quat_l, pos_l = pos_dict[lbl]['quat'], pos_dict[lbl]['pos']
                    base_quat_l, base_pos_l = base_pos_dict[lbl]['quat'], base_pos_dict[lbl]['pos']

                    # # render_data['means3D'][total_mask_full == lbl] = gu.transform_by_trans_quat(render_data['means3D'][total_mask_full == lbl], rel_pos, rel_quat) 
                    # # render_data['rotations'][total_mask_full == lbl] = quat_mult_torch(render_data['rotations'][total_mask_full == lbl], rel_quat.unsqueeze(0))

                    new_render_data['means3D'][total_mask_full == lbl] = gu.transform_by_trans_quat(gu.inv_transform_by_trans_quat(render_data['means3D'][total_mask_full == lbl], base_pos_l, base_quat_l), pos_l, quat_l) 
                    q_base_inv = gu.inv_quat(base_quat_l)
                    new_render_data['rotations'][total_mask_full == lbl] = gu.transform_quat_by_quat(gu.transform_quat_by_quat(render_data['rotations'][total_mask_full == lbl], q_base_inv), quat_l)
                    # new_render_data['rotations'][total_mask_full == lbl] = gu.transform_quat_by_quat(quat_l, gu.transform_quat_by_quat(q_base_inv, render_data['rotations'][total_mask_full == lbl]))
                    tot_pcd.append(new_render_data['means3D'][total_mask_full == lbl])

                    # Debug with base pcd
                    # base_pcd = base_geom_dict[lbl]
                    # pcd = gu.inv_transform_by_trans_quat(torch.from_numpy(base_pcd).to(device), base_pos_l, base_quat_l)
                    # pcd = gu.transform_by_trans_quat(pcd, pos_l, quat_l)
                    # tot_pcd.append(pcd)

            # Debug 
            tot_pcd = torch.cat(tot_pcd, dim=0).cpu().numpy()
            _ = trimesh.PointCloud(tot_pcd).export('tmp_pcd.ply')

            # for k in new_render_data.keys():
            #     new_render_data[k] = new_render_data[k][pts_is_robot_mask]
            im_gs, _, depth_gs = GaussianRasterizer(raster_settings=cam)(**new_render_data)
            im_gs_np = (im_gs.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"iter_camera_third_person.png"), im_gs_np[..., ::-1]) # debug
            import pdb; pdb.set_trace()
            del new_render_data

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()