"""
This file converts the coordinate frame of interiornet from 
https://github.com/RuojinCai/ExtremeRotation_code
to our coordinate frame.
"""

import cv2
import os
import os.path as osp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from PIL import Image
import quaternion
from sparseplane.utils.vis import get_single_image_mesh_plane
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.structures import Boxes, BoxMode, Instances
from sparseplane.utils.mesh_utils import transform_meshes, get_camera_meshes, save_obj
from pytorch3d.structures import join_meshes_as_batch


def create_instances(kps, img):
    ret = Instances(img.shape[:2])
    pt_per_ray = 10
    masks = []
    planes = []
    for kp in kps:
        x, y = kp
        mask = np.zeros_like(img)[..., 0]
        mask[int(y), int(x)] = 1
        mask = cv2.dilate(mask, np.ones((10, 10), dtype=np.uint8), iterations=1)
        mask = GenericMask(mask, img.shape[0], img.shape[1])
        masks.extend([mask.polygons for _ in range(pt_per_ray)])
        planes.extend([[0, 0, (i+1)] for i in range(pt_per_ray)])
    ret.pred_masks = masks
    ret.pred_planes = np.array(planes).astype(np.float32)
    return ret

def save_pair_objects(
    img_file1,
    img_file2,
    height, 
    width,
    focal_length,
    p_instances,
    output_dir,
    prefix="",
    pred_camera=None,
    plane_param_override=None,
    show_camera=True,
    webvis=False,
):
    """
    save predictions on image pair as obj file
    """
    image_paths = {"0": img_file1, "1": img_file2}
    meshes_list = []
    # map_files = []
    uv_maps = []
    cam_list = []
    # get plane parameters
    plane_locals = {}
    for i in range(2):
        if plane_param_override is None:
            plane_locals[str(i)] = p_instances[str(i)].pred_planes
        else:
            plane_locals[str(i)] = plane_param_override[str(i)]
    # get camera 1 to 2
    camera1to2 = {
        "position": np.array(pred_camera["position"]),
        "rotation": quaternion.from_float_array(pred_camera["rotation"]),
    }


    os.makedirs(output_dir, exist_ok=True)
    for i in range(2):
        if i == 0:
            camera_info = camera1to2
        else:
            camera_info = {
                "position": np.array([0, 0, 0]),
                "rotation": np.quaternion(1, 0, 0, 0),
            }
        p_instance = p_instances[str(i)]
        plane_params = plane_locals[str(i)]
        segmentations = p_instance.pred_masks
        meshes, uv_map = get_single_image_mesh_plane(
            plane_params,
            segmentations,
            img_file=image_paths[str(i)],
            height=height,
            width=width,
            focal_length=focal_length,
            webvis=False,
        )
        uv_maps.extend(uv_map)
        meshes = transform_meshes(meshes, camera_info)
        meshes_list.append(meshes)
        cam_list.append(camera_info)
    joint_mesh = join_meshes_as_batch(meshes_list)
    if webvis:
        joint_mesh = rotate_mesh_for_webview(joint_mesh)

    # add camera into the mesh
    if show_camera:
        cam_meshes = get_camera_meshes(cam_list)
        if webvis:
            cam_meshes = rotate_mesh_for_webview(cam_meshes)
    else:
        cam_meshes = None
    # save obj
    if len(prefix) == 0:
        prefix = "pred"
    save_obj(
        folder=output_dir,
        prefix=prefix,
        meshes=joint_mesh,
        cam_meshes=cam_meshes,
        decimal_places=10,
        blend_flag=True,
        map_files=None,
        uv_maps=uv_maps,
    )



def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """
    Copy from https://gist.github.com/isker/11be0c50c4f78cad9549
    Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
        
    r = 2
    thickness = 1
        
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        
        cv2.line(new_img, end1, end2, np.array(c, dtype='uint8').tolist(), thickness)
        cv2.circle(new_img, end1, r, np.array(c, dtype='uint8').tolist(), thickness)
        cv2.circle(new_img, end2, r, np.array(c, dtype='uint8').tolist(), thickness)
    
    return new_img


def get_match(img1, kp1, des1, img2, kp2, des2):
    # BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    MIN_MATCH_COUNT = 3 # Affine is 6 DoF
            
    if len(good)>= MIN_MATCH_COUNT:
        src_pts = np.array([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2).astype('int32')
        dst_pts = np.array([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2).astype('int32')
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, 
                                       ransacReprojThreshold=max(img1.shape)*0.01)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = []
    good = [good[i] for i in range(len(matchesMask)) if matchesMask[i] == 1]
    return good

def main(inv_R_on=False, rot_x=0, rot_y=0, rot_z=0):
    # read in two images.
    img_root = 'tools/images/interior_00'
    if 1:
        img_file1 = osp.join(img_root, 'r_0.png')
        img_file2 = osp.join(img_root, 'r_1.png')
        R = np.array([
            [ 0.9750,  0.0138, -0.2219],
            [ 0.1013,  0.8607,  0.4989],
            [ 0.1979, -0.5089,  0.8378],
        ])
    if 0:
        R = [
            [ 0.9953,  0.0216, -0.0949],
            [-0.0055,  0.9859,  0.1673],
            [ 0.0972, -0.1660,  0.9813]
        ]
        img_file1 = osp.join(img_root, 'r1_0.png')
        img_file2 = osp.join(img_root, 'r1_1.png')
    if 0:
        R = [
            [ 0.7416,  0.0056, -0.6709],
            [ 0.0109,  0.9997,  0.0203],
            [ 0.6708, -0.0224,  0.7413]
        ]
        img_file1 = osp.join(img_root, 'r2_0.png')
        img_file2 = osp.join(img_root, 'r2_1.png')
    if 0:
        R = [
            [ 0.9950,  0.0610,  0.0792],
            [-0.0413,  0.9725, -0.2293],
            [-0.0910,  0.2249,  0.9701]
        ]
        img_file1 = osp.join(img_root, 'r3_0.png')
        img_file2 = osp.join(img_root, 'r3_1.png')
    T = np.array([0,0,0])
    imgs = [
        cv2.imread(img_file1),
        cv2.imread(img_file2),
    ]
    """
    begin transformation
    """
    rotx_90 = np.array(
        [
            [1,0,0],
            [0,0,-1],
            [0,1,0],
        ]
    )
    roty_90 = np.array(
        [
            [0,0,-1],
            [0,1,0],
            [1,0,0],
        ]
    )
    rotz_90 = np.array(
        [
            [0,-1,0],
            [1,0,0],
            [0,0,1],
        ]
    )
    mapping = np.eye(3)
    for _ in range(np.abs(rot_z)):
        if rot_z > 0: 
            rot_m = rotz_90
        else:
            rot_m = rotz_90.T
        mapping = mapping @ rot_m
    for _ in range(np.abs(rot_y)):
        if rot_y > 0: 
            rot_m = roty_90
        else:
            rot_m = roty_90.T
        mapping = mapping @ rot_m
    for _ in range(np.abs(rot_x)):
        if rot_x > 0: 
            rot_m = rotx_90
        else:
            rot_m = rotx_90.T
        mapping = mapping @ rot_m

    if inv_R_on:
        R = mapping.T @ R.T @ mapping
    else:
        R = mapping.T @ R @ mapping

    # R = np.eye(3)
    print(mapping)
    # R = np.eye(3)
    """
    end transformation
    """

    # convert to sparseplane format
    camera_info = {
        'position': T,
        'rotation': quaternion.as_float_array(quaternion.from_rotation_matrix(R)),
    }
    # find sift correspondences, 
    # rays originated from these corresponding points from two views should align
    sift = cv2.SIFT_create(contrastThreshold=0.01, sigma=0.8)
    (kp1, des1) = sift.detectAndCompute(imgs[0], np.ones_like(imgs[0])[...,0])
    (kp2, des2) = sift.detectAndCompute(imgs[1], np.ones_like(imgs[1])[...,0])

    good = get_match(imgs[0], kp1, des1, imgs[1], kp2, des2)[:10]
    
    new_img = draw_matches(imgs[0], kp1, imgs[1], kp2, good)
    cv2.imwrite("debug/match.jpg", new_img)
    # visualize rays in 3D. 
    p_instances = {}

    src_pts = np.array([ kp1[m.queryIdx].pt for m in good ]).astype('int32')
    dst_pts = np.array([ kp2[m.trainIdx].pt for m in good ]).astype('int32')
    p_instances['0'] = create_instances(
        src_pts, imgs[0]
    )
    p_instances['1'] = create_instances(
        dst_pts, imgs[1]
    )
    print(f'{inv_R_on}_{rot_x}_{rot_y}_{rot_z}')
    save_pair_objects(
        img_file1,
        img_file2,
        256,
        256,
        256 / 2 / np.tan(np.radians(90 / 2)), # Interiornet image has 90 deg fov.
        p_instances,
        './debug',
        prefix=f"{osp.basename(img_file1).split('.')[0]}_{osp.basename(img_file2).split('.')[0]}_{inv_R_on}_{rot_x}_{rot_y}_{rot_z}",
        pred_camera=camera_info,
    )


if __name__=='__main__':
    """
    Figuring out coordinate frame transformation from two dataset given just a rotation matrix is painful.
    We enumerate different possible transformations, and visualize the alignment.
    """
    # main(False, 0, 2, -3)
    main(False, 0, 2, 3)
    # for i in np.arange(-3, 4):
    #     main(True, i, 0, 0)
    #     main(False, i, 0, 0)
    #     main(True, 0, i, 0)
    #     main(False, 0, i, 0)
    #     main(True, 0, 0, i)
    #     main(False, 0, 0, i)
