import argparse
import numpy as np
import pytorch3d
import open3d as o3d
from pytorch3d.io import load_ply_mp3d
from detectron2.utils.colormap import colormap
from plyfile import PlyData, PlyElement
import os
from glob import glob
from tqdm import tqdm
import torch
import trimesh

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument('--base', type=int, default=0,
                    help='start scene idx.')
parser.add_argument('--num', type=int, default=1,
                    help='number of scenes to be processed.')
parser.add_argument('--id', type=int, default=0,
                    help='process id.')
args = parser.parse_args()


ROOT_FOLDER = '/Pool1/users/jinlinyi/dataset/matterport3d/v1/extracted'
PLY_FOLDER = '/Pool1/users/jinlinyi/dataset/mp3d_habitat/mp3d'
PLANE_FOLDER = '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v3/planes_ply'

def face2vert(face_id, face):
    vert_id = np.ones(face.max()+1)*-1
    for i in range(3):
        vert_id[face[:,i]] = face_id
    return vert_id


def get_full_house(house_id):
    ply_path = os.path.join(PLY_FOLDER, f'{house_id}/{house_id}_semantic.ply')

    house_mesh = PlyData.read(ply_path)
    house_verts = np.stack([house_mesh['vertex']['x'], house_mesh['vertex']['y'], house_mesh['vertex']['z']], axis=1)
    
    region_folder = os.path.join(ROOT_FOLDER, f'{house_id}/region_segmentations')
    regions = list(sorted(glob(os.path.join(region_folder, 'region*.ply'))))
    region_meshes = {}
    for region in regions:
        region_id = int(region.split('/')[-1].split('.')[0].split('region')[-1])
        region_meshes[region_id] = PlyData.read(region)

    # TEST ORDER
    sum_verts = []
    for key in sorted(region_meshes.keys()):
        verts = np.stack([region_meshes[key]['vertex']['x'], region_meshes[key]['vertex']['y'], region_meshes[key]['vertex']['z']], axis=1)
        sum_verts.append(verts)
    sum_verts = np.vstack(sum_verts)
    assert((sum_verts - house_verts).sum() == 0)
    # Load plane annotation
    plane_folder = os.path.join(PLANE_FOLDER, f'{house_id}/plane_annotation')
    plane_ids_vert = []
    plane_params = []
    offset = 0
    for region_id in sorted(region_meshes.keys()):
        plane_ann = np.load(os.path.join(plane_folder, 'region'+str(region_id)+'_planeid.npy'), allow_pickle=True)
        plane_param = np.load(os.path.join(plane_folder, 'region'+str(region_id)+'.npy'), allow_pickle=True)
        assert(len(plane_param) == sum(np.unique(plane_ann) != -1))
        assert(plane_ann.max() - plane_ann.min() + 1) == len(np.unique(plane_ann))
        plane_params.extend(plane_param)
        mask = plane_ann >= 0
        additional = len(np.unique(plane_ann[mask]))
        plane_ann[mask] += offset
        plane_ids_vert.extend(plane_ann)
        offset += additional
    plane_params = np.array(plane_params)
    assert(len(plane_params) == sum(np.unique(plane_ids_vert) != -1))

    plane_id_face = []
    for face in house_mesh['face']['vertex_indices']:
        if plane_ids_vert[face[0]] == plane_ids_vert[face[1]] and plane_ids_vert[face[0]] == plane_ids_vert[face[2]]:
            plane_id_face.append(plane_ids_vert[face[0]])
        else:
            plane_id_face.append(-1)

    house_mesh['face']['object_id'] = np.array(plane_id_face).astype(np.int32) + 1
    np.save(os.path.join(PLANE_FOLDER, house_id, 'house_plane_params.npy'), plane_params)
    house_mesh.write(os.path.join(PLANE_FOLDER, house_id, house_id+'_semantic.ply'))
    cmd = f"ln -s {os.path.join(PLY_FOLDER, house_id, house_id+'.glb')} {os.path.join(PLANE_FOLDER, house_id)}"
    os.system(cmd)
    cmd = f"ln -s {os.path.join(PLY_FOLDER, house_id, house_id+'.navmesh')} {os.path.join(PLANE_FOLDER, house_id)}"
    os.system(cmd)
    cmd = f"ln -s {os.path.join(PLY_FOLDER, house_id, house_id+'.house')} {os.path.join(PLANE_FOLDER, house_id)}"
    os.system(cmd)
    


def main():
    ply_path = '/Pool1/users/jinlinyi/dataset/matterport3d/v1/extracted/17DRP5sb8fy/house_segmentations/17DRP5sb8fy.ply'
    mesh_o3d = o3d.io.read_triangle_mesh(ply_path)
    mesh_p3d = load_ply_mp3d(ply_path)
    cmap = colormap() / 255    
    vert_instance_id = face2vert(mesh_p3d['segment_id'], mesh_p3d['faces'])
    vert_category_id = face2vert(mesh_p3d['category_id'], mesh_p3d['faces'])
    new_color_cat = cmap[vert_category_id.astype(int)%len(cmap)]
    new_color_ins = cmap[vert_instance_id.astype(int)%len(cmap)]
    o3d.io.write_triangle_mesh("debug/origin.gltf", mesh_o3d)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_color_cat)
    o3d.io.write_triangle_mesh("debug/cat.gltf", mesh_o3d)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_color_ins)
    o3d.io.write_triangle_mesh("debug/ins.gltf", mesh_o3d)


if __name__=='__main__':
    scene_ids = sorted(os.listdir(ROOT_FOLDER))
    try: 
        scene_ids.remove('sens')
    except:
        pass
    scene_ids = scene_ids[args.base:args.base+args.num]
    assert(len(scene_ids) == args.num)

    for scene_id in tqdm(scene_ids, desc=f"process_id {args.id}"):
        print('converting', scene_id)
        get_full_house(scene_id)