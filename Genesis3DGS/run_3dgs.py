import os
import torch
import torchvision
import math
import numpy as np
import genesis as gs
from gs_processor import GSProcessor
from diff_gaussian_rasterization import GaussianRasterizer
from transform_utils import setup_camera
from apply_transform import build_transform



def get_extrinsic(pos, lookat):
    cam_pos = np.array(pos)
    forward = np.array(lookat) - cam_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 0, 1])

    # If forward is too close to world_up, pick a different up vector
    if abs(np.dot(forward, world_up)) > 0.999:  # almost parallel
        world_up = np.array([0, 1, 0])  # pick another up axis

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    # Build c2w matrix (OpenGL convention)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    # convert to [1,-1,-1]
    transform_opencv_to_opengl = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])
    c2w = transform_opencv_to_opengl @ c2w
    c2w[:3, 3] = cam_pos
    return c2w




def set_camera_custom(self, center=(0, 0, 0), distance=0.8, elevation=20, azimuth=160.0, near=0.01, far=100.0):
    target = np.array(center)
    theta = 90 + azimuth
    z = distance * math.sin(math.radians(elevation))
    y = math.cos(math.radians(theta)) * distance * math.cos(math.radians(elevation))
    x = math.sin(math.radians(theta)) * distance * math.cos(math.radians(elevation))
    origin = target + np.array([x, y, z])
    
    look_at = target - origin
    look_at /= np.linalg.norm(look_at)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(look_at, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, look_at)
    w2c = np.eye(4)
    w2c[:3, 0] = right
    w2c[:3, 1] = -up
    w2c[:3, 2] = look_at
    w2c[:3, 3] = origin
    w2c = np.linalg.inv(w2c)
    return w2c

# tx, ty, tz = 0.0, 3.23, 0.0
# rx, ry, rz = -23.31, 0.0, -180.0
# sx, sy, sz = 1.0, 1.0, 1.0
tx, ty, tz = 0.0, 0, 3.19
rx, ry, rz = -113, 0.0, 0.0
sx, sy, sz = 1.0, 1.0, 1.0
transform = build_transform(tx, ty, tz, rx, ry, rz, sx, sy, sz)

# # --- Coordinate conversion matrix ---
# T_pc_to_gen = np.array([
#     [1,  0,  0, 0],
#     [0,  0,  -1, 0],
#     [0, 1,  0, 0],
#     [0,  0,  0, 1],
# ], dtype=np.float32)
# transform = T_pc_to_gen @ transform @ T_pc_to_gen.T

rot_mat = transform[:3, :3]
trans = transform[:3, 3]

output_dir = 'tmp_renders'
os.makedirs(output_dir, exist_ok=True)
gs_path = '/mnt/hdd/code/outdoor_relighting/gaussian-splatting/output/tnt/playroom/point_cloud/iteration_30000/point_cloud.ply'
sp = GSProcessor()
# params_bg = sp.load(gs_path)
params_bg = sp.load_ply(gs_path)

device = 'cuda'
bg = [0, 0, 0]
max_sh_degrees = 3
print('max_sh_degrees: ', max_sh_degrees)


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
# render_data = gs_processor(render_data, scale)

render_data = {
    'means3D': render_data['means3D'],
    'shs': render_data['sh_colors'],
    'rotations': torch.nn.functional.normalize(render_data['unnorm_rotations'], dim=-1),
    'opacities': torch.sigmoid(render_data['logit_opacities']),
    'scales': torch.exp(render_data['log_scales']),
    'means2D': torch.zeros_like(render_data['means3D'], requires_grad=True), # (N,3)
}

# for i in range(1000):
    # pos = np.array([i/100, 0, 2.5], dtype=np.float32)
for i in range(120):
    pos = np.array([3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5], dtype=np.float32)
    lookat = np.array([0, 0, 0.5], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    print('pos: ', pos)

    c2w = gs.utils.geom.pos_lookat_up_to_T(pos, lookat, up)
    c2w[:3, 1:3] *= -1
    # c2w2 = get_extrinsic(pos, lookat)
    # w2c = np.linalg.inv(c2w)
    # w2c = gs.utils.geom.inv(c2w)
    w2c = np.linalg.inv(c2w)
    print('w2c: ', w2c)
    print('w2c pos: ', w2c[:3, 3])

    metadata = {
        'w': 1280,
        'h': 960,
        'k': np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]]),
        'w2c': w2c,
        'near': 0.01,
        'far': 100.0,
    }

    # cam = setup_camera(metadata['w'], metadata['h'], metadata['k'], 
    #         metadata['w2c'], metadata['near'], metadata['far'], bg, 
    #         z_threshold=0.05, sh_degree=max_sh_degrees, device=device)

    cam = setup_camera(metadata['w'], metadata['h'], metadata['k'], 
            metadata['w2c'], metadata['near'], metadata['far'], bg, 
            sh_degree=max_sh_degrees, device=device)

    # im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
    im, _ = GaussianRasterizer(raster_settings=cam)(**render_data)

    torchvision.utils.save_image(im, os.path.join(output_dir, "tmp_iter{}.png".format(i)))
    print("Saved image to {}".format(os.path.join(output_dir, "tmp_iter{}.png".format(i))))