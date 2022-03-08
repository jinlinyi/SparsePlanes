import cv2
import os
import shutil
import quaternion
import torch
import numpy as np
from typing import Optional
import imageio
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import utils as struct_utils
from sparseplane.utils.camera import (
    create_cylinder_mesh,
    create_color_palette,
    get_cone_edges,
)


def transform_meshes(meshes, camera_info):
    """
    input:
    @meshes: mesh in local frame
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    mesh in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    verts_packed = meshes.verts_packed()
    verts_packed = verts_packed * torch.tensor(
        [1.0, -1.0, -1.0], dtype=torch.float32
    )  # suncg2habitat
    faces_list = meshes.faces_list()
    tex = meshes.textures
    rot_matrix = torch.tensor(quaternion.as_rotation_matrix(rot), dtype=torch.float32)
    verts_packed = torch.mm(rot_matrix, verts_packed.T).T + torch.tensor(
        tran, dtype=torch.float32
    )
    verts_list = list(verts_packed.split(meshes.num_verts_per_mesh().tolist(), dim=0))
    return Meshes(verts=verts_list, faces=faces_list, textures=tex)


def rotate_mesh_for_webview(meshes):
    """
    input:
    @meshes: mesh in global (habitat) frame
    output:
    mesh is rotated around x axis by -11 degrees such that floor is horizontal
    """
    verts_packed = meshes.verts_packed()
    faces_list = meshes.faces_list()
    tex = meshes.textures
    rot_matrix = torch.FloatTensor(
        np.linalg.inv(
            np.array([[1, 0, 0], [0, 0.9816272, -0.1908090], [0, 0.1908090, 0.9816272]])
        )
    )
    verts_packed = torch.mm(rot_matrix, verts_packed.T).T
    verts_list = list(verts_packed.split(meshes.num_verts_per_mesh().tolist(), dim=0))
    return Meshes(verts=verts_list, faces=faces_list, textures=tex)


def transform_verts_list(verts_list, camera_info):
    """
    input:
    @meshes: verts_list in local frame
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    verts_list in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    verts_list_to_packed = struct_utils.list_to_packed(verts_list)
    verts_packed = verts_list_to_packed[0]
    num_verts_per_mesh = verts_list_to_packed[1]
    verts_packed = verts_packed * torch.tensor(
        [1.0, -1.0, -1.0], dtype=torch.float32
    )  # suncg2habitat
    rot_matrix = torch.tensor(quaternion.as_rotation_matrix(rot), dtype=torch.float32)
    verts_packed = torch.mm(rot_matrix, verts_packed.T).T + torch.tensor(
        tran, dtype=torch.float32
    )
    verts_list = list(verts_packed.split(num_verts_per_mesh.tolist(), dim=0))
    return verts_list


def get_plane_params_in_global(planes, camera_info):
    """
    input:
    @planes: plane params
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    plane parameters in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    start = np.ones((len(planes), 3)) * tran
    end = planes * np.array([1, -1, -1])  # suncg2habitat
    end = (quaternion.as_rotation_matrix(rot) @ (end).T).T + tran  # cam2world
    a = end
    b = end - start
    planes_world = ((a * b).sum(axis=1) / np.linalg.norm(b, axis=1) ** 2).reshape(-1, 1) * b
    return planes_world


def get_plane_params_in_local(planes, camera_info):
    """
    input:
    @planes: plane params
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    plane parameters in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    b = planes
    a = np.ones((len(planes), 3)) * tran
    planes_world = (
        a
        + b
        - ((a * b).sum(axis=1) / np.linalg.norm(b, axis=1) ** 2).reshape(-1, 1) * b
    )
    end = (
        quaternion.as_rotation_matrix(rot.inverse()) @ (planes_world - tran).T
    ).T  # world2cam
    planes_local = end * np.array([1, -1, -1])  # habitat2suncg
    return planes_local


def save_obj(
    folder,
    prefix,
    meshes,
    cam_meshes=None,
    decimal_places=None,
    blend_flag=False,
    map_files=None,
    uv_maps=None,
):
    os.makedirs(folder, exist_ok=True)

    # pytorch3d does not support map_files
    # map_files = meshes.textures.map_files()
    # assert map_files is not None
    if map_files is None and uv_maps is None:
        raise RuntimeError("either map_files or uv_maps should be set!")

    # generate map_files from uv_map
    if uv_maps is not None and map_files is None:
        map_files = []
        uv_dir = os.path.join(folder, "uv_maps")
        if not os.path.exists(uv_dir):
            os.mkdir(uv_dir)
        for map_id, uv_map in enumerate(uv_maps):
            uv_path = os.path.join(uv_dir, "{}_uv_plane_{}.png".format(prefix, map_id))
            imageio.imwrite(uv_path, uv_map)
            map_files.append(uv_path)

    f_mtl = open(os.path.join(folder, prefix + ".mtl"), "w")
    f = open(os.path.join(folder, prefix + ".obj"), "w")
    try:
        seen = set()
        uniq_map_files = [
            m for m in list(map_files) if m not in seen and not seen.add(m)
        ]
        for map_id, map_file in enumerate(uniq_map_files):
            if uv_maps is not None:
                # we do not need to copy map_files,
                # they are already in uv_maps/...
                f_mtl.write(
                    _get_mtl_map(
                        os.path.basename(map_file).split(".")[0],
                        os.path.join("uv_maps", os.path.basename(map_file)),
                    )
                )
                continue

            if not blend_flag:
                shutil.copy(map_file, folder)
                os.chmod(os.path.join(folder, os.path.basename(map_file)), 0o755)
                f_mtl.write(
                    _get_mtl_map(
                        os.path.basename(map_file).split(".")[0],
                        os.path.basename(map_file),
                    )
                )
            else:
                rgb = cv2.imread(map_file, cv2.IMREAD_COLOR)
                if cam_meshes is not None:
                    blend_color = (
                        np.array(
                            cam_meshes.textures.verts_features_packed()
                            .numpy()
                            .tolist()[map_id]
                        )
                        * 255
                    )
                else:
                    blend_color = np.array(create_color_palette()[map_id + 10])
                alpha = 0.7
                blend = (rgb * alpha + blend_color[::-1] * (1 - alpha)).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(
                        folder, os.path.basename(map_file).split(".")[0] + "_debug.png"
                    ),
                    blend,
                )
                f_mtl.write(
                    _get_mtl_map(
                        os.path.basename(map_file).split(".")[0],
                        os.path.basename(map_file).split(".")[0] + "_debug.png",
                    )
                )

        f.write(f"mtllib {prefix}.mtl\n\n")
        # we want [list]    verts, vert_uvs, map_files;
        #         [packed]  faces;
        #         face per mesh
        verts_list = meshes.verts_list()
        verts_uvs_list = meshes.textures.verts_uvs_list()
        faces_list = meshes.faces_packed().split(
            meshes.num_faces_per_mesh().tolist(), dim=0
        )
        for idx, (verts, verts_uvs, faces, map_file) in enumerate(
            zip(verts_list, verts_uvs_list, faces_list, map_files)
        ):
            f.write(f"# mesh {idx}\n")
            trunc_verts_uvs = verts_uvs[: verts.shape[0]]
            _save(
                f,
                verts,
                faces,
                verts_uv=trunc_verts_uvs,
                map_file=map_file,
                idx=idx,
                decimal_places=decimal_places,
            )
        if cam_meshes:
            face_offset = np.sum([len(v) for v in verts_list])
            cam_verts_list = cam_meshes.verts_list()
            cam_verts_rgbs_list = (
                cam_meshes.textures.verts_features_packed().numpy().tolist()
            )
            cam_faces_list = (cam_meshes.faces_packed() + face_offset).split(
                cam_meshes.num_faces_per_mesh().tolist(), dim=0
            )
            assert len(cam_verts_rgbs_list) == len(cam_verts_list)
            for idx, (verts, faces, rgb) in enumerate(
                zip(cam_verts_list, cam_faces_list, cam_verts_rgbs_list)
            ):
                f.write(f"# camera {idx}\n")
                f_mtl.write(_get_mtl_rgb(idx, rgb))
                _save(f, verts, faces, rgb=rgb, idx=idx, decimal_places=decimal_places)
    finally:
        f.close()
        f_mtl.close()


def _get_mtl_map(material_name, map_Kd):
    return f"""newmtl {material_name}
map_Kd {map_Kd}
# Test colors
Ka 1.000 1.000 1.000  # white
Kd 1.000 1.000 1.000  # white
Ks 0.000 0.000 0.000  # black
Ns 10.0\n"""


def _get_mtl_rgb(material_idx, rgb):
    return f"""newmtl color_{material_idx}
Kd {rgb[0]} {rgb[1]} {rgb[2]}
Ka 0.000 0.000 0.000\n"""


def _save(
    f,
    verts,
    faces,
    verts_uv=None,
    map_file=None,
    rgb=None,
    idx=None,
    double_sided=True,
    decimal_places: Optional[int] = None,
):
    if decimal_places is None:
        float_str = "%f"
    else:
        float_str = "%" + ".%df" % decimal_places

    lines = ""

    V, D = verts.shape
    for i in range(V):
        vert = [float_str % verts[i, j] for j in range(D)]
        lines += "v %s\n" % " ".join(vert)

    if verts_uv is not None:
        V, D = verts_uv.shape
        for i in range(V):
            vert_uv = [float_str % verts_uv[i, j] for j in range(D)]
            lines += "vt %s\n" % " ".join(vert_uv)

    if map_file is not None:
        lines += f"usemtl {os.path.basename(map_file).split('.')[0]}\n"
    elif rgb is not None:
        lines += f"usemtl color_{idx}\n"

    if faces != []:
        F, P = faces.shape
        for i in range(F):
            if verts_uv is not None:
                face = ["%d/%d" % (faces[i, j] + 1, faces[i, j] + 1) for j in range(P)]
            else:
                face = ["%d" % (faces[i, j] + 1) for j in range(P)]
            lines += "f %s\n" % " ".join(face)
            if double_sided:
                if verts_uv is not None:
                    face = [
                        "%d/%d" % (faces[i, j] + 1, faces[i, j] + 1)
                        for j in reversed(range(P))
                    ]
                else:
                    face = ["%d" % (faces[i, j] + 1) for j in reversed(range(P))]
                lines += "f %s\n" % " ".join(face)
    else:
        tqdm.write(f"face = []")
    f.write(lines)


def get_camera_meshes(camera_list, radius=0.02):
    verts_list = []
    faces_list = []
    color_list = []
    rots = np.array(
        [
            quaternion.as_rotation_matrix(camera_info["rotation"])
            for camera_info in camera_list
        ]
    )

    # ai habitat frame
    lookat = np.array([0, 0, -1])
    vertical = np.array([0, 1, 0])

    positions = np.array([camera_info["position"].flatten() for camera_info in camera_list])
    lookats = rots @ lookat.T
    verticals = rots @ vertical.T
    predetermined_color = [
        [0.10196, 0.32157, 1.0],
        [1.0, 0.0667, 0.1490],
        [197/255, 181/255, 24/255],
        [73/255, 145/255, 115/255],
        [198/255, 120/255, 221/255],
    ][:len(camera_list)]
    assert len(predetermined_color) == len(camera_list)
    for idx, (position, lookat, vertical, color) in enumerate(
        zip(positions, lookats, verticals, predetermined_color)
    ):
        cur_num_verts = 0
        edges = get_cone_edges(position, lookat, vertical)
        cam_verts = []
        cam_inds = []
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cyl_verts = [x for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            cur_num_verts += len(cyl_verts)
            cam_verts.extend(cyl_verts)
            cam_inds.extend(cyl_ind)
        # Create a textures object
        verts_list.append(torch.tensor(cam_verts, dtype=torch.float32))
        faces_list.append(torch.tensor(cam_inds, dtype=torch.float32))
        color_list.append(color)

    color_tensor = torch.tensor(color_list, dtype=torch.float32).unsqueeze_(1)
    tex = TexturesVertex(verts_features=color_tensor)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
    return meshes
