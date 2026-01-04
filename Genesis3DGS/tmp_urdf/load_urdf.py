import os
import numpy as np
import trimesh
import genesis as gs
import pdb
import open3d as o3d
from pathlib import Path
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.ext import urdfpy
# https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/utils/urdf.py

def trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    return o3d_mesh

robot_urdf_path = "/mnt/hdd/code/Dongki_project/Genesis3DGS/data/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
# robot_urdf_path = "/mnt/hdd/code/Dongki_project/Genesis3DGS/real2sim-eval/assets/robots/xarm/xarm7_with_gripper.urdf"
# panda_xml_path = 'xml/franka_emika_panda/panda.xml'

gs.init(precision="32", logging_level="info")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.0, -2, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=200,
    ),
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
    ),
    show_viewer=True,
)

robot = scene.add_entity(
        gs.morphs.URDF(
            file=robot_urdf_path,
            pos=(0, 0, 0.4),
        ),
    )

# robot = scene.add_entity(
#     gs.morphs.MJCF(file=panda_xml_path),
# )

# Study this code
# https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/utils/urdf.py
surface = robot.surface
morph = robot.morph

def parse_urdf(morph, surface):
    if isinstance(morph.file, (str, Path)):
        path = os.path.join(morph.file)
        robot = urdfpy.URDF.load(path)
    else:
        robot = morph.file

    link_name_to_idx = dict()
    for idx, link in enumerate(robot.links):
        link_name_to_idx[link.name] = idx

    # Note that each link corresponds to one joint
    n_links = len(robot.links)
    assert n_links == len(robot.joints) + 1
    l_infos = [dict() for _ in range(n_links)]
    links_j_infos = [[] for _ in range(n_links)]
    links_g_infos = [[] for _ in range(n_links)]

    for link, l_info, link_g_infos in zip(robot.links, l_infos, links_g_infos):
        l_info["name"] = link.name

        # No parent by default. It will be overwritten latter on if appropriate.
        l_info["parent_idx"] = -1

        # Neutral pose by default. It will be overwritten latter on if necessary.
        l_info["pos"] = gu.zero_pos()
        l_info["quat"] = gu.identity_quat()

        # we compute urdf's invweight later
        l_info["invweight"] = np.full((2,), fill_value=-1.0)

        if link.inertial is None:
            l_info["inertial_pos"] = gu.zero_pos()
            l_info["inertial_quat"] = gu.identity_quat()
            l_info["inertial_i"] = None
            l_info["inertial_mass"] = None

        else:
            l_info["inertial_pos"] = link.inertial.origin[:3, 3]
            l_info["inertial_quat"] = gu.R_to_quat(link.inertial.origin[:3, :3])
            l_info["inertial_i"] = link.inertial.inertia
            l_info["inertial_mass"] = link.inertial.mass

        for geom in (*link.collisions, *link.visuals):
            link_g_infos_ = []
            geom_is_col = not isinstance(geom, urdfpy.Visual)
            if isinstance(geom.geometry.geometry, urdfpy.Mesh):
                geom_type = gs.GEOM_TYPE.MESH
                geom_data = None

                # One asset (.obj) can contain multiple meshes. Each mesh is one RigidGeom in genesis.
                for tmesh in geom.geometry.meshes:
                    scale = float(morph.scale)
                    if geom.geometry.geometry.scale is not None:
                        scale *= geom.geometry.geometry.scale

                    mesh_path = urdfpy.utils.get_filename(os.path.dirname(path), geom.geometry.geometry.filename)
                    mesh = gs.Mesh.from_trimesh(
                        tmesh,
                        scale=scale,
                        surface=gs.surfaces.Collision() if geom_is_col else surface,
                        metadata={"mesh_path": mesh_path},
                    )

                    if mesh_path.lower().endswith(gs.morphs.GLTF_FORMATS):
                        if morph.parse_glb_with_zup:
                            mesh.convert_to_zup()
                        else:
                            gs.logger.warning(
                                "This file contains GLTF mesh, which is using y-up while Genesis uses z-up. Please set "
                                "'parse_glb_with_zup=True' in morph options if you find the mesh is 90-degree rotated. "
                            )

                    if not geom_is_col and (morph.prioritize_urdf_material or not tmesh.visual.defined):
                        if geom.material is not None and geom.material.color is not None:
                            mesh.set_color(geom.material.color)

                    g_info = {"mesh" if geom_is_col else "vmesh": mesh}
                    link_g_infos_.append(g_info)
            else:
                # Each geometry primitive is one RigidGeom in genesis
                if isinstance(geom.geometry.geometry, urdfpy.Box):
                    tmesh = trimesh.creation.box(extents=geom.geometry.geometry.size)
                    geom_type = gs.GEOM_TYPE.BOX
                    geom_data = np.array(geom.geometry.geometry.size)
                elif isinstance(geom.geometry.geometry, urdfpy.Capsule):
                    tmesh = trimesh.creation.capsule(
                        radius=geom.geometry.geometry.radius, height=geom.geometry.geometry.length
                    )
                    geom_type = gs.GEOM_TYPE.CAPSULE
                    geom_data = np.array([geom.geometry.geometry.radius, geom.geometry.geometry.length])
                elif isinstance(geom.geometry.geometry, urdfpy.Cylinder):
                    tmesh = trimesh.creation.cylinder(
                        radius=geom.geometry.geometry.radius, height=geom.geometry.geometry.length
                    )
                    geom_type = gs.GEOM_TYPE.CYLINDER
                    geom_data = np.array([geom.geometry.geometry.radius, geom.geometry.geometry.length])
                elif isinstance(geom.geometry.geometry, urdfpy.Sphere):
                    if geom_is_col:
                        tmesh = trimesh.creation.icosphere(radius=geom.geometry.geometry.radius, subdivisions=2)
                    else:
                        tmesh = trimesh.creation.icosphere(radius=geom.geometry.geometry.radius)
                    geom_type = gs.GEOM_TYPE.SPHERE
                    geom_data = np.array([geom.geometry.geometry.radius])

                mesh = gs.Mesh.from_trimesh(
                    tmesh,
                    scale=morph.scale,
                    surface=gs.surfaces.Collision() if geom_is_col else surface,
                )

                if not geom_is_col:
                    if geom.material is not None and geom.material.color is not None:
                        mesh.set_color(geom.material.color)

                g_info = {"mesh" if geom_is_col else "vmesh": mesh}
                link_g_infos_.append(g_info)

            for g_info in link_g_infos_:
                g_info["type"] = geom_type
                g_info["data"] = geom_data
                g_info["pos"] = geom.origin[:3, 3].copy()
                g_info["quat"] = gu.R_to_quat(geom.origin[:3, :3])
                g_info["contype"] = 1 if geom_is_col else 0
                g_info["conaffinity"] = 1 if geom_is_col else 0
                g_info["friction"] = gu.default_friction()
                g_info["sol_params"] = gu.default_solver_params()
            link_g_infos += link_g_infos_


# parse_urdf(morph, surface)


def parse_urdf2(morph, surface, link_names=None):
    if isinstance(morph.file, (str, Path)):
        print('morph.file1: ', morph.file)
        path = os.path.join(morph.file)
        robot = urdfpy.URDF.load(path)
    else:
        print('morph.file2: ', morph.file)
        robot = morph.file

    n_links = len(robot.links)
    print('n_links: ', n_links)

    meshes = dict()
    scales = dict()
    offsets = dict()

    prev_offset = np.eye(4)
    for link in robot.links:
        # Optional filter for a subset of links
        if link_names is not None and link.name not in link_names:
            continue

        # Initialize containers for this link
        meshes[link.name] = []
        scales[link.name] = []
        offsets[link.name] = []
    
        # Only care about collisions, not visuals
        if not link.collisions:
            continue
            
        for collision in link.collisions:
            geom = collision
            geom_is_col = True 
        #     import pdb; pdb.set_trace()

            # 4x4 transform of this collision geom in the link frame
            origin_T = geom.origin.copy()  # np.ndarray (4, 4)
            # Store the transform
            offsets[link.name].append(origin_T)

            # Handle mesh vs primitive, similar to parse_urdf
            if isinstance(geom.geometry.geometry, urdfpy.Mesh):
                print('link.name1:', link.name)
                # Mesh geometry
                geom_type = gs.GEOM_TYPE.MESH
    
                    # One URDF mesh asset can contain multiple sub-meshes
                for tmesh in geom.geometry.meshes:
                    # Base scale from morph
                    scale = float(morph.scale)


                # One URDF mesh asset can contain multiple sub-meshes
                for tmesh in geom.geometry.meshes:
                    # Base scale from morph
                    scale = float(morph.scale)

                    # URDF mesh-specific scale
                    if geom.geometry.geometry.scale is not None:
                        # urdfpy usually stores this as [sx, sy, sz]
                        # If you want uniform scale, you can take one component.
                        # Here we keep the full vector for clarity.
                        scale_vec = np.array(geom.geometry.geometry.scale, dtype=float)
                        # You can either multiply per component or pick one:
                        # scale *= np.mean(scale_vec)
                        # For Genesis Mesh.from_trimesh, scale can be scalar or vector.
                        scale = scale_vec * scale

                    # Resolve mesh file path (same as parse_urdf)
                    mesh_path = urdfpy.utils.get_filename(
                        os.path.dirname(path),
                        geom.geometry.geometry.filename,
                    )
                    print('mesh_path: ', mesh_path)

                    mesh = gs.Mesh.from_trimesh(
                        tmesh,
                        scale=scale,
                        surface=gs.surfaces.Collision() if geom_is_col else surface,
                        metadata={"mesh_path": mesh_path},
                    )

                    # Handle GLTF up-axis issue (same as parse_urdf)
                    if mesh_path.lower().endswith(gs.morphs.GLTF_FORMATS):
                        if morph.parse_glb_with_zup:
                            mesh.convert_to_zup()
                        else:
                            gs.logger.warning(
                                "GLTF mesh uses y-up while Genesis uses z-up. "
                                "Set 'parse_glb_with_zup=True' if you see a 90° rotation."
                            )

                    # Store results
                    meshes[link.name].append(trimesh_to_open3d(mesh.trimesh))
                    # Here we store a scalar for convenience; you can store full vector if you prefer.
                    if np.isscalar(scale):
                        scales[link.name].append(float(scale))
                    else:
                        scales[link.name].append(scale)
            else:
                # Primitive: Box / Capsule / Cylinder / Sphere
                geom_obj = geom.geometry.geometry

                if isinstance(geom_obj, urdfpy.Box):
                    tmesh = trimesh.creation.box(extents=geom_obj.size)
                    geom_type = gs.GEOM_TYPE.BOX
                elif isinstance(geom_obj, urdfpy.Capsule):
                    tmesh = trimesh.creation.capsule(
                        radius=geom_obj.radius,
                        height=geom_obj.length,
                    )
                    geom_type = gs.GEOM_TYPE.CAPSULE
                elif isinstance(geom_obj, urdfpy.Cylinder):
                    tmesh = trimesh.creation.cylinder(
                        radius=geom_obj.radius,
                        height=geom_obj.length,
                    )
                    geom_type = gs.GEOM_TYPE.CYLINDER
                elif isinstance(geom_obj, urdfpy.Sphere):
                    # Collisions often use low-res sphere
                    tmesh = trimesh.creation.icosphere(
                        radius=geom_obj.radius,
                        subdivisions=2,
                    )
                    geom_type = gs.GEOM_TYPE.SPHERE
                else:
                    # Unknown primitive; skip safely
                    continue

                # Primitive scale usually just comes from morph.scale
                scale = float(morph.scale)

                mesh = gs.Mesh.from_trimesh(
                    tmesh,
                    scale=scale,
                    surface=gs.surfaces.Collision(),
                )

                meshes[link.name].append(trimesh_to_open3d(mesh.trimesh))
                scales[link.name].append(scale)

    return meshes, scales, offsets
   

meshes, scales, offsets = parse_urdf2(morph, surface)

pdb.set_trace()

