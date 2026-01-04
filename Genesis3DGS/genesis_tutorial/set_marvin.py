import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

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
    urdf_file = "/mnt/hdd/code/Dongki_project/Genesis3DGS/data/marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file,
            fixed=True,
        ),
    )
    ########################## build ##########################
    scene.build()


    initial_joint_angles_right = np.deg2rad([-90,-75,90,-90,-75,0,-20])
    initial_joint_angles_left = np.deg2rad([90,-75,-90,-90,75,0,20])
    print('initial_joint_angles_right: ', initial_joint_angles_right, len(initial_joint_angles_right))
    print('initial_joint_angles_left: ', initial_joint_angles_left, len(initial_joint_angles_left))


    left_joint_names = [
        'Joint1_L',
        'Joint2_L',
        'Joint3_L',
        'Joint4_L',
        'Joint5_L',
        'Joint6_L',
        'Joint7_L',
    ]

    right_joint_names = [
        'Joint1_R',
        'Joint2_R',
        'Joint3_R',
        'Joint4_R',
        'Joint5_R',
        'Joint6_R',
        'Joint7_R',
    ]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in left_joint_names + right_joint_names]


    q_arms = np.concatenate([initial_joint_angles_left,
                            initial_joint_angles_right])   # shape (14,)


    robot.set_dofs_position(q_arms, dofs_idx)
    for i in range(10000):
        robot.set_dofs_position(q_arms, dofs_idx)
        scene.step()

if __name__ == "__main__":
    main()
