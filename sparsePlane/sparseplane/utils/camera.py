"""
Modified from
https://github.com/Sekunde/3D-SIS/blob/master/tools/visualization.py
"""
import math
import numpy as np


def create_color_palette():
    return [
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144),
    ]


def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
    def compute_length_vec3(vec3):
        return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

    def rotation(axis, angle):
        rot = np.eye(4)
        c = np.cos(-angle)
        s = np.sin(-angle)
        t = 1.0 - c
        axis /= compute_length_vec3(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rot[0, 0] = 1 + t * (x * x - 1)
        rot[0, 1] = z * s + t * x * y
        rot[0, 2] = -y * s + t * x * z
        rot[1, 0] = -z * s + t * x * y
        rot[1, 1] = 1 + t * (y * y - 1)
        rot[1, 2] = x * s + t * y * z
        rot[2, 0] = y * s + t * x * z
        rot[2, 1] = -x * s + t * y * z
        rot[2, 2] = 1 + t * (z * z - 1)
        return rot

    verts = []
    indices = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)
    for i in range(stacks + 1):
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array(
                [
                    radius * math.cos(theta),
                    radius * math.sin(theta),
                    height * i / stacks,
                ]
            )
            verts.append(pos)
    for i in range(stacks):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            indices.append(
                np.array(
                    [(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1],
                    dtype=np.uint32,
                )
            )
            indices.append(
                np.array(
                    [(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1],
                    dtype=np.uint32,
                )
            )
    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]
            if math.fabs(dotx) != 1.0:
                axis = np.array([1, 0, 0]) - dotx * va
            else:
                axis = np.array([0, 1, 0]) - va[1] * va
            axis /= compute_length_vec3(axis)
        transform = rotation(axis, -angle)
    transform[:3, 3] += p0
    verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
    verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

    return verts, indices


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, "w")
    file.write("ply \n")
    file.write("format ascii 1.0\n")
    file.write("element vertex {:d}\n".format(len(verts)))
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property uchar red\n")
    file.write("property uchar green\n")
    file.write("property uchar blue\n")
    file.write("element face {:d}\n".format(len(indices)))
    file.write("property list uchar uint vertex_indices\n")
    file.write("end_header\n")
    for vert, color in zip(verts, colors):
        file.write(
            "{:f} {:f} {:f} {:d} {:d} {:d}\n".format(
                vert[0],
                vert[1],
                vert[2],
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            )
        )
    for ind in indices:
        file.write("3 {:d} {:d} {:d}\n".format(ind[0], ind[1], ind[2]))
    file.close()


def get_cone_edges(position, lookat, vertical):
    def get_cone_verts(position, lookat, vertical):
        vertical = np.array(vertical) / np.linalg.norm(vertical)
        lookat = np.array(lookat) / np.linalg.norm(lookat)
        right = np.cross(np.array(lookat), np.array(vertical))
        right = right / np.linalg.norm(right)
        top = np.cross(right, lookat)
        top = top / np.linalg.norm(top)

        right *= 0.4
        lookat *= 0.4
        top *= 0.1
        verts = {
            "topR": position + lookat + top + right,
            "topL": position + lookat + top - right,
            "center": position,
            "bottomR": position + lookat - top + right,
            "bottomL": position + lookat - top - right,
        }
        return verts

    cone_verts = get_cone_verts(position, lookat, vertical)
    edges = [
        (cone_verts["center"], cone_verts["topR"]),
        (cone_verts["center"], cone_verts["topL"]),
        (cone_verts["center"], cone_verts["bottomR"]),
        (cone_verts["center"], cone_verts["bottomL"]),
        (cone_verts["topR"], cone_verts["topL"]),
        (cone_verts["bottomR"], cone_verts["topR"]),
        (cone_verts["bottomR"], cone_verts["bottomL"]),
        (cone_verts["topL"], cone_verts["bottomL"]),
    ]
    return edges


def write_obj(verts, colors, indices, output_file, mtl_filename):
    """
    Write the current Mesh instance to an obj file.
    """
    obj_f = open(output_file, "w")
    if mtl_filename is not None:
        mtl_f = open(mtl_filename, "w")

        # define mtllib
        fname = mtl_filename.split("/")[-1]
        obj_f.write("mtllib {}\n".format(fname))
    for idx, (vert, color, face) in enumerate(zip(verts, colors, indices)):
        name = "c" + str(idx)
        if mtl_filename is not None:
            # write mtl
            mtl_f.write("newmtl {}\n".format(name))
            mtl_f.write("Kd {} {} {}\n".format(color[0], color[1], color[2]))
            mtl_f.write("Ka 0 0 0\n")

            # write obj
            obj_f.write("usemtl {}\n".format(name))
        for v in vert:
            obj_f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for f in face:
            obj_f.write("f {} {} {}\n".format(int(f[0]), int(f[1]), int(f[2])))

    obj_f.close()

    if mtl_filename is not None:
        mtl_f.close()


def write_cone_ply(positions, lookats, vertical, output_file=None):
    """
    positions: np array (n, 3)
    output_file: string
    """
    radius = 0.02
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []
    for idx, (position, lookat) in enumerate(zip(positions, lookats)):
        r, g, b = create_color_palette()[idx]
        edges = get_cone_edges(position, lookat, vertical)
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cur_num_verts = len(verts)
            cyl_color = [[r / 255.0, g / 255.0, b / 255.0] for _ in cyl_verts]
            cyl_verts = [x + offset for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            verts.extend(cyl_verts)
            indices.extend(cyl_ind)
            colors.extend(cyl_color)
    if output_file:
        write_ply(verts, colors, indices, output_file)
    else:
        return verts, colors, indices


def write_cone_obj(positions, lookats, verticals, radius=0.02, output_file=None):
    """
    positions: np array (n, 3)
    output_file: string
    """
    verts = []
    indices = []
    colors = []
    cur_num_verts = 1
    for idx, (position, lookat, vertical) in enumerate(
        zip(positions, lookats, verticals)
    ):
        r, g, b = create_color_palette()[idx + 10]
        edges = get_cone_edges(position, lookat, vertical)
        color = [r / 255.0, g / 255.0, b / 255.0]
        cam_verts = []
        cam_inds = []
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cyl_verts = [x for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            cur_num_verts += len(cyl_verts)
            cam_verts.extend(cyl_verts)
            cam_inds.extend(cyl_ind)
        verts.append(cam_verts)
        indices.append(cam_inds)
        colors.append(color)
    if output_file:
        write_obj(
            verts, colors, indices, output_file, output_file.split(".")[0] + ".mtl"
        )
    else:
        return np.array(verts), np.array(colors), np.array(indices)


def main():
    positions = [
        [0, 0, 0],
        [-2.03869411, -0.66838259, 3.33877471],
    ]
    rot = np.array(
        [
            [-0.6863150, -0.1426361, 0.7131807],
            [0.1426361, 0.9351418, 0.3242912],
            [-0.7131807, 0.3242912, -0.6214568],
        ]
    )

    lookat = [[0, 0, 1], rot @ np.array([0, 0, 1]).T]
    vertical = [0, -1, 0]

    write_cone_obj(positions, lookat, vertical, "cone.obj")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = create_color_palette()
    colors = [np.array(list(c)) / 255 for c in colors]
    for c in colors:
        print(c)
    sns.palplot(colors)
    plt.show()
    main()
