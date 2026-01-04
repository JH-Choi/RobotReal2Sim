import argparse
import numpy as np
import genesis as gs


def find_link_indices(entity, names):
    if not names:
        return []
    link_indices = list()
    for link in entity.links:
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
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_collision=False,
            gravity=(0, 0, -0),
        ),
    )
    

    ########################## entities ##########################
    urdf_file = '/mnt/hdd/code/Dongki_project/Genesis3DGS/real2sim-eval/assets/robots/xarm/xarm7_with_gripper.urdf'
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file, 
            fixed=True,
        # merge_fixed_links=False
        ),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    target_entity_fk = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
        ),
        surface=gs.surfaces.Default(color=(0.5, 0.5, 1.0, 1)),
    )

    ########################## build ##########################
    scene.build()

    target_quat = np.array([0, 1, 0, 0])  # pointing downwards
    center = np.array([0.4, -0.2, 0.25])
    r = 0.1

    for link in robot.links:
        print('link.name: ', link.name)

    ee_link = robot.get_link("left_finger")
    ee_link_idx = find_link_indices(robot, ["left_finger"])

    for i in range(0, 2000):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r

        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))

        q = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
        )
        pos, quat = robot.forward_kinematics(q, links_idx_local=ee_link_idx)
        target_entity_fk.set_qpos(np.concatenate([pos.cpu().numpy()[0], quat.cpu().numpy()[0]]))

        # Note that this IK example is only for visualizing the solved q, so here we do not call scene.step(), but only update the state and the visualizer
        # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
        robot.set_qpos(q)
        scene.visualizer.update()


if __name__ == "__main__":
    main()