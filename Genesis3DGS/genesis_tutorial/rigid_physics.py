import genesis as gs
import numpy as np
import torch

TOL = 1e-5
show_viewer = True

gs.init(precision="32", logging_level="info")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, 0.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
    ),
    vis_options=gs.options.VisOptions(
        rendered_envs_idx=(1,),
    ),
    show_viewer=show_viewer,
)
robot = scene.add_entity(
    morph=gs.morphs.URDF(
        file="urdf/shadow_hand/shadow_hand.urdf",
    ),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.05, 0.05, 0.05),
        pos=(0.0, 0.2, 0.05),
    ),
)
scene.build(n_envs=2)
scene.reset()

index_finger_distal = robot.get_link("index_finger_distal")
middle_finger_distal = robot.get_link("middle_finger_distal")
wrist = robot.get_link("wrist")
index_finger_pos = np.array([[0.6, 0.5, 0.2]])
middle_finger_pos = np.array([[0.63, 0.5, 0.2]])
wrist_pos = index_finger_pos - np.array([[0.0, 0.0, 0.2]])

qpos, err = robot.inverse_kinematics_multilink(
    links=(index_finger_distal, middle_finger_distal, wrist),
    poss=(index_finger_pos, middle_finger_pos, wrist_pos),
    envs_idx=(1,),
    pos_tol=TOL,
    rot_tol=TOL,
    return_error=True,
)
assert qpos.shape == (1, robot.n_qs)
assert err.shape == (1, 3, 6)
assert err.abs().max() < TOL
import pdb; pdb.set_trace()
if show_viewer:
    robot.set_qpos(qpos, envs_idx=(1,))
    scene.visualizer.update()

links_pos, links_quat = robot.forward_kinematics(qpos, envs_idx=(1,))
# assert_allclose(links_pos[:, index_finger_distal.idx], index_finger_pos, tol=TOL)
# assert_allclose(links_pos[:, middle_finger_distal.idx], middle_finger_pos, tol=TOL)
# assert_allclose(links_pos[:, wrist.idx], wrist_pos, tol=TOL)

robot.set_qpos(qpos, envs_idx=(1,))
import pdb; pdb.set_trace()
scene.rigid_solver._func_forward_kinematics_entity(
    i_e=robot.idx, envs_idx=torch.tensor((1,), dtype=gs.tc_int, device=gs.device)
)
# assert_allclose(index_finger_distal.get_pos(envs_idx=(1,)), index_finger_pos, tol=TOL)
# assert_allclose(middle_finger_distal.get_pos(envs_idx=(1,)), middle_finger_pos, tol=TOL)
# assert_allclose(wrist.get_pos(envs_idx=(1,)), wrist_pos, tol=TOL)