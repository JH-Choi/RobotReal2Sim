import os
import torch
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


def main():
    
    gs.init()
    
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.5, 0.0, 0.0), size=(0.05, 0.05, 0.05)))
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

    #########################################################################################
    # define GS Background
    #########################################################################################
    tx, ty, tz = 0.0, 0, 3.92
    rx, ry, rz = -113, 0.0, 0.0
    sx, sy, sz = 1.0, 1.0, 1.0

    transform = build_transform(tx, ty, tz, rx, ry, rz, sx, sy, sz)

    rot_mat = transform[:3, :3]
    trans = transform[:3, 3]

    with_gs_background = True
    output_dir = 'tmp_renders'
    os.makedirs(output_dir, exist_ok=True)
    gs_path = '/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/tnt/playroom/point_cloud/iteration_30000/point_cloud.ply'
    sp = GSProcessor()
    params_bg = sp.load_ply(gs_path)
    device = 'cuda'
    bg = [0, 0, 0]
    max_sh_degrees = 3
    render_data = {
        'means3D': params_bg['means3D'],
        'sh_colors': params_bg['sh_colors'],
        'unnorm_rotations': params_bg['rotations'],
        'logit_opacities': params_bg['opacities'],
        'log_scales': params_bg['scales'],
        'means2D': torch.zeros_like(params_bg['means3D'], requires_grad=True), # (N,3)
    }
    render_data = {k: v.to(device) for k, v in render_data.items()}
    render_data = sp.rotate(render_data, rot_mat)
    render_data = sp.translate(render_data, trans)
    render_data = {
        'means3D': render_data['means3D'],
        'shs': render_data['sh_colors'],
        'rotations': torch.nn.functional.normalize(render_data['unnorm_rotations'], dim=-1),
        'opacities': torch.sigmoid(render_data['logit_opacities']),
        'scales': torch.exp(render_data['log_scales']),
        'means2D': torch.zeros_like(render_data['means3D'], requires_grad=True), # (N,3)
    }

    #########################################################################################
    # add cameras
    #########################################################################################
    cameras = {}
    cameras["camera_third_person"] = scene.add_camera(
        res=(640, 480),
        GUI=True, fov=30,
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
    )
    cameras["camera_first_person"] = scene.add_camera(GUI=True, fov=70)
    
    scene.build()

    T = np.eye(4)
    T[:3, :3] = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    T[:3, 3] = np.array([0.1, 0.0, 0.1])
    cameras["camera_first_person"].attach(franka.get_link("hand"), T)

    target_quat = np.array([0, 1, 0, 0]) # pointing downwards
    center = np.array([0.5, 0.0, 0.5])
    angular_speed = np.random.random() * 10.0
    r = 0.1

    ee_link = franka.get_link('hand')

    # cam.start_recording()
    for k in cameras.keys():
        cameras[k].start_recording()

    rgb_dict = defaultdict(list)
    im_gs_dict, full_im_dict = defaultdict(list), defaultdict(list)

    # rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
    for i in range(300):
        target_pos = center.copy()
        target_pos[0] += np.cos(i/360*np.pi*angular_speed) * r
        target_pos[1] += np.sin(i/360*np.pi*angular_speed) * r

        q = franka.inverse_kinematics(
            link     = ee_link,
            pos      = target_pos,
            quat     = target_quat,
            rot_mask = [False, False, True], # for demo purpose: only restrict direction of z-axis
        )

        franka.set_qpos(q)
        scene.step()
        # cam.render()
        # rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

        for k in cameras.keys():
            camera_inst = cameras[k]
            if k == "camera_third_person":
                camera_inst.set_pose(
                    pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
                    lookat=(0, 0, 0.5),
                )


            rgb, depth, segmentation, _ = cameras[k].render(rgb=True, depth=True, segmentation=True)
            rgb_dict[k].append(rgb)

            # Ensure tensors and normalize RGB to [0, 255] for consistency
            if isinstance(rgb, np.ndarray):
                rgb_t = torch.from_numpy(rgb.copy())
            else:
                rgb_t = torch.as_tensor(rgb)
            if rgb_t.is_floating_point():
                # If already [0,1], scale to [0,255]
                try:
                    maxv = float(rgb_t.max().item())
                except Exception:
                    maxv = 1.0
                if maxv <= 1.01:
                    rgb_t = (rgb_t * 255.0).clamp(0, 255)

            if isinstance(depth, np.ndarray):
                depth_t = torch.from_numpy(depth.copy())
            else:
                depth_t = torch.as_tensor(depth)

            if with_gs_background:
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

                im_gs, _, depth_gs = GaussianRasterizer(raster_settings=cam)(**render_data)
                im_gs_np = (im_gs.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"iter_{k}_{i}.png"), im_gs_np[..., ::-1]) # debug
                im_gs_dict[k].append(im_gs_np)

                # Create foreground mask from segmentation
                if segmentation is not None:
                    seg_np = segmentation if isinstance(segmentation, np.ndarray) else segmentation.cpu().numpy()
                    # Exclude background and ground plane (typically ID 0 = ground, ID 1 = first object)
                    foreground_mask = seg_np > 1
                    mask = np.where(foreground_mask, 255, 0).astype(np.uint8)

                    # Blending RGB
                    sim_rgb = (
                        rgb_t.numpy().astype(np.uint8) if isinstance(rgb_t, torch.Tensor) else rgb_t.astype(np.uint8)
                    )
                    foreground = np.concatenate([sim_rgb, mask[..., None]], axis=-1)
                    background = im_gs_np
                    blended_rgb = alpha_blend_rgba(foreground, background)
                    rgb_t = blended_rgb.copy()

                    # Compose depth
                    bg_depth = depth_gs.permute(1, 2, 0).cpu().detach().numpy()
                    if bg_depth.ndim == 3 and bg_depth.shape[-1] == 1:
                        bg_depth = bg_depth[..., 0]
                    depth_np = depth_t.numpy() if isinstance(depth_t, torch.Tensor) else depth_t
                    depth_comp = np.where(foreground_mask, depth_np, bg_depth)
                    depth_t = depth_comp.copy()

                    # save image
                    # from PIL import Image
                    # rgb_t.save(os.path.join(output_dir, f"iter_{k}_{i}_full.png"))
                    full_im_dict[k].append(np.array(rgb_t))


    # cam.stop_recording(save_to_filename="video.mp4", fps=60)
    for k in cameras.keys():
        cameras[k].stop_recording(save_to_filename=f"video_{k}.mp4", fps=60)

    # write video
    for k in cameras.keys():
        all_rgb = np.stack(rgb_dict[k], axis=0)
        all_im_gs = np.stack(im_gs_dict[k], axis=0)
        all_full_im = np.stack(full_im_dict[k], axis=0)
        imageio.mimwrite(os.path.join(output_dir, f"video_{k}_rgb.mp4"), all_rgb, fps=60, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(output_dir, f"video_{k}_gs.mp4"), all_im_gs, fps=60, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(output_dir, f"video_{k}_full.mp4"), all_full_im, fps=60, quality=8, macro_block_size=1)


if __name__ == "__main__":
    main()