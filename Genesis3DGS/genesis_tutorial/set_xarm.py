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
    urdf_file = '/mnt/hdd/code/Dongki_project/Genesis3DGS/real2sim-eval/assets/robots/xarm/xarm7_with_gripper.urdf'
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file,
            merge_fixed_links=False,
            fixed=True,
        ),
    )
    link_names = [
        'link1',
        'link2',
        'link3',
        'link4',
        'link5',
        'link6',
        'link7',
        'xarm_gripper_base_link',
        'left_outer_knuckle',
        'left_finger',
        'left_inner_knuckle',
        'right_outer_knuckle',
        'right_finger',
        'right_inner_knuckle',
    ]

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

    g = 750  # gripper openness
    g = (800 - g) * 180 / np.pi
    base_qpos = np.array([0, -45, 0, 30, 0, 75, 0, 
                          g*0.001, g*0.001, g*0.001, g*0.001, g*0.001, g*0.001]) * np.pi / 180

    # link_idx_ls = []
    # for link_name in link_names:
    #     for link_idx, link in enumerate(robot.links):
    #         if link.name == link_name:
    #             link_idx_ls.append(link_idx)
    #             break

    link_idx_ls = find_link_indices(robot, link_names)
    pos, quat = robot.forward_kinematics(base_qpos, links_idx_local=link_idx_ls)
    target_entity_fk.set_qpos(np.concatenate([pos.cpu().numpy()[0], quat.cpu().numpy()[0]]))

    robot.set_qpos(base_qpos)
    scene.visualizer.update()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
