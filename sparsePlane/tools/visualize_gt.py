#!/usr/bin/env python
import argparse
import json
import torch
import numpy as np
import os
import shutil
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import cv2
from tqdm import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances, pairwise_iou
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from pytorch3d.structures import join_meshes_as_batch

from sparseplane.data import PlaneRCNNMapper
from sparseplane.visualization import draw_match
from sparseplane.utils.vis import get_single_image_mesh_plane
from sparseplane.utils.mesh_utils import save_obj, get_camera_meshes, transform_meshes, get_plane_params_in_global, get_plane_params_in_local


def get_camera_dicts(camera_pickle):
    """
    key: image id. 0_0_0
    value: {save_path': '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v3/rgb/17DRP5sb8fy/0_0_0.png', 'img_name': '0_0_0.png', 'house_idx': '17DRP5sb8fy', 'position': array([-9.56018401,  1.577794  , -2.63851021]), 'rotation': quaternion(0.984984462909517, -0.0948432164229211, 0.143593870170991, 0.0138265170857656)}
    """
    with open(camera_pickle, 'rb') as f:
        cameras_list = pickle.load(f)
    cameras_dict = {}
    for c in cameras_list:
        cameras_dict[c['img_name'].split('.')[0]] = c
    return cameras_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="mp3d_val")
    parser.add_argument("--vis-num", default=10, type=int, help="number of visualizations")
    parser.add_argument("--rgb-path", default='/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20/rgb/', type=str, help="path to rgb dataset")
    parser.add_argument("--camera-path", default='/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20/cameras', type=str)
    parser.add_argument("--save-obj", action='store_true', help="whether save obj file")
    
    args = parser.parse_args()

    logger = setup_logger()
    np.random.seed(2021)
    
    os.makedirs(args.output, exist_ok=True)
    metadata = MetadataCatalog.get(args.dataset)
    dicts = list(DatasetCatalog.get(args.dataset))
    dicts = np.random.choice(dicts, args.vis_num, replace=False).tolist()
    for dic in tqdm(dicts):
        key0 = dic['0']['image_id']
        key1 = dic['1']['image_id']
        key = key0 + '__' + key1
        uv_maps = []
        meshes_list = []
        cam_list = []
        # gt
        for i in range(2):
            house_name, basename = dic[str(i)]["image_id"].split('_', 1)
            img_file = os.path.join(args.rgb_path, house_name, basename+'.png')
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)[:,:,::-1]
            cameras_dict = get_camera_dicts(os.path.join(args.camera_path, house_name+'.pkl'))
            vis = Visualizer(img, metadata)
            seg_gt = vis.draw_dataset_dict(dic[str(i)]).get_image()
            plane_params = [ann['plane'] for ann in dic[str(i)]['annotations']]
            segmentations = [ann['segmentation'] for ann in dic[str(i)]['annotations']]
            if args.save_obj:
                gt_meshes, uv_map = get_single_image_mesh_plane(
                    plane_params, 
                    segmentations, 
                    img_file=img_file, 
                    height=dic[str(i)]['height'], 
                    width=dic[str(i)]['width'],
                    webvis=False,
                )
                uv_maps.extend(uv_map)
                
                gt_meshes = transform_meshes(gt_meshes, cameras_dict[basename])
                meshes_list.append(gt_meshes)
                cam_list.append(cameras_dict[basename])
        
        joint_mesh = join_meshes_as_batch(meshes_list)
        cam_meshes = get_camera_meshes(cam_list)
        save_obj(
            os.path.join(args.output, house_name), 
            key + '_gt', 
            joint_mesh, 
            cam_meshes, 
            decimal_places=10,
            blend_flag=True,
            map_files=None,
            uv_maps=uv_maps,
        )