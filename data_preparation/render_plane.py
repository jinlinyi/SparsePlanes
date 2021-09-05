import argparse
import cv2
import colorsys
import os
import numpy as np
import pickle
import quaternion
from glob import glob
from tqdm import tqdm
from PIL import Image

import habitat_sim

import utils


parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument('--base', type=int, default=0,
                    help='start scene idx.')
parser.add_argument('--num', type=int, default=1,
                    help='number of scenes to be processed.')
parser.add_argument('--id', type=int, default=0,
                    help='process id.')
args = parser.parse_args()

DATASET_DIR = '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v3'
ROOT_FOLDER = '/Pool1/users/jinlinyi/dataset/matterport3d/v1/extracted'
PLANE_FOLDER = '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v3/planes_ply_mp3dcoord_refined_sep20'
SAVE_FOLDER = '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20'

bad_objects = [
    "void",
    "",
]

def get_K_inv_dot_xy_1(h=480, w=640):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, h, w))

    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640
                
            ray = np.dot(K_inv,
                         np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]

    return K_inv_dot_xy_1

K_inv_dot_xy_1_global = get_K_inv_dot_xy_1()

def mp3d2habitat(planes):
    rotation = np.array([
        [1,0,0],
        [0,0,-1],
        [0,1,0],
    ])
    rotation = np.linalg.inv(rotation)
    return (rotation@np.array(planes).T).T

def remove_small_plane(plane_instance_id, bkgd=0):
    # remove small contours / bubbles
    h, w = plane_instance_id.shape
    area_threshold = 480*640*0.01
    for i in sorted(np.unique(plane_instance_id)):
        if i == bkgd:
            continue
        mask = plane_instance_id == i
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        big_contours = []
        for c in contours:
            if cv2.contourArea(c) > area_threshold:
                big_contours.append(c)
        bad_mask = np.ones((h,w))
        cv2.fillPoly(bad_mask, big_contours, 0)
        remove_mask = np.logical_and(bad_mask, mask)
        plane_instance_id[remove_mask] = bkgd        
    return plane_instance_id

def erode_planes(plane_instance_id, bkgd=0):
    kernel = np.ones((5,5),np.uint8)
    for i in sorted(np.unique(plane_instance_id)):
        if i == bkgd:
            continue
        mask = plane_instance_id == i
        erosion = cv2.erode(mask.astype('uint8'),kernel,iterations = 1)
        remove_mask = mask - erosion
        plane_instance_id[remove_mask==1] = i
    return plane_instance_id

def convex_hull_fill_holes(plane_instance_id, depth, plane_params, camera_info, bkgd=0, depth_threshold=0.1): 
    h, w = plane_instance_id.shape
    for i in sorted(np.unique(plane_instance_id)):
        if i == bkgd:
            continue
        mask = plane_instance_id == i
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        hull = cv2.convexHull(np.vstack(c for c in contours))
        convex_mask = np.zeros((h, w))
        cv2.fillConvexPoly(convex_mask, hull, 1)

        # Convert plane parameter in local
        plane_global = mp3d2habitat([plane_params[i-1]])
        plane_local = get_plane_params_in_local(plane_global, camera_info)[0]
        offset = np.linalg.norm(plane_local)
        normal = plane_local / (offset + 1e-5)
        # get depth from plane parameter, camera Extrinsics and Intrinsics
        projected_depth = (offset/(normal@K_inv_dot_xy_1_global.reshape(3,-1))).reshape(h, w)
        # Find region where depth error is small
        depth_good_mask = np.abs(projected_depth - depth) < depth_threshold

        filled_pixels = np.logical_and(plane_instance_id == bkgd, convex_mask)
        filled_pixels = np.logical_and(filled_pixels, depth_good_mask)
        plane_instance_id[filled_pixels] = i
    return plane_instance_id


def filter_by_depth(plane_instance_id, depth, plane_params, camera_info, bkgd=0, threshold=0.3):    
    h, w = depth.shape
    for i in sorted(np.unique(plane_instance_id)):
        if i == bkgd:
            continue
        # Get mask of the plane
        mask = plane_instance_id == i
        # Convert plane parameter in local
        plane_global = mp3d2habitat([plane_params[i-1]])
        plane_local = get_plane_params_in_local(plane_global, camera_info)[0]
        offset = np.linalg.norm(plane_local)
        normal = plane_local / (offset + 1e-5)
        # get depth from plane parameter, camera Extrinsics and Intrinsics
        projected_depth = (offset/(normal@K_inv_dot_xy_1_global.reshape(3,-1))).reshape(h, w)
        # Find region where depth error is small
        err_mask = np.abs(projected_depth - depth) > threshold
        err_mask = np.logical_and(err_mask, mask)
        plane_instance_id[err_mask] = bkgd
    return plane_instance_id


def get_plane_params_in_local(planes, camera_info):
    """
    input: 
    @planes: plane params
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    plane parameters in global frame.
    """
    tran = camera_info['position']
    rot = camera_info['rotation']
    b = planes
    a = np.ones((len(planes),3))*tran
    planes_world = a + b - ((a*b).sum(axis=1) / np.linalg.norm(b, axis=1)**2).reshape(-1,1)*b
    end = (quaternion.as_rotation_matrix(rot.inverse())@(planes_world - tran).T).T #world2cam
    planes_local = end*np.array([1, -1, -1])# habitat2suncg
    return planes_local


def check_quality(plane_instance_id, bkgd=0):
    threshold=0.3*640*480
    bkgd_mask = plane_instance_id == bkgd
    if bkgd_mask.sum() > threshold:
        return False
    else:
        return True

def select_top_k_area(plane_instance_id, k=20, background=0):
    count = {}
    for i in sorted(np.unique(plane_instance_id)):
        if i == background:
            continue
        count[i] = (plane_instance_id == i).sum()    
    sorted_keys = [k for k, v in sorted(count.items(), key=lambda item: item[1])][::-1]
    for key in sorted_keys[k:]:
        plane_instance_id[plane_instance_id == key] = background
    return plane_instance_id


def render_scene(test_scene):
    os.makedirs(os.path.join(SAVE_FOLDER, 'rgb', test_scene), exist_ok=True)
    os.makedirs(os.path.join(SAVE_FOLDER, 'observations', test_scene), exist_ok=True)
    # house_idx = test_scene.split('/')[-2]
    camera_path = os.path.join(DATASET_DIR, 'cameras', test_scene + '.pkl')
    with open(camera_path, 'rb') as f:
        cameras = pickle.load(f)
    img_width = 640
    img_height = 480
    sim_settings = {
        "width": img_width,  # Spatial resolution of the observations    
        "height": img_height,
        "scene": os.path.join(PLANE_FOLDER, test_scene, test_scene+'.glb'),  # Scene path
        "default_agent": 0,  
        "sensor_height": 0,  # Height of sensors in meters
        "color_sensor": True,  # RGB sensor
        "semantic_sensor": True,  # Semantic sensor
        "depth_sensor": True,  # Depth sensor
        "seed": 1,
    }
    plane_params = np.load(os.path.join(PLANE_FOLDER, test_scene, 'house_plane_params.npy'), allow_pickle=True)

    cmap = random_colors(len(plane_params))
    cmap = np.array(cmap)*255

    cfg = utils.make_cfg(sim_settings, intrinsics=None)
    sim = habitat_sim.Simulator(cfg)
    # Print semantic annotation information (id, category, bounding box details) 
    # about levels, regions and objects in a hierarchical fashion
    scene = sim.semantic_scene

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    for camera in cameras:
        agent_state = habitat_sim.AgentState()
        # agent_state.position = grid + np.random.random(3)*np.array([GRID_SIZE, HEIGHT_RANGE, GRID_SIZE])
        agent_state.position = camera['position']
        agent_state.rotation = camera['rotation']
        # print(agent_state)
        agent.set_state(agent_state)
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        plane_instance_id = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
        
        # plane_instance_id = select_top_k_area(plane_instance_id)
        plane_instance_id = erode_planes(plane_instance_id)
        plane_instance_id = filter_by_depth(plane_instance_id, depth, plane_params, camera)
        plane_instance_id = convex_hull_fill_holes(plane_instance_id, depth, plane_params, camera)
        plane_instance_id = remove_small_plane(plane_instance_id)
        plane_instance_id = convex_hull_fill_holes(plane_instance_id, depth, plane_params, camera)
        quality = check_quality(plane_instance_id)

        image = observations['color_sensor'][:,:,:3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plane_seg = cmap[plane_instance_id.flatten()%len(cmap)].reshape((img_height,img_width,3))
        blend_pred = (plane_seg * 0.5 + image * 0.5).astype(np.uint8)
        blend_pred[plane_instance_id == 0] = image[plane_instance_id==0]
        """
        # DEBUG
        save_path = os.path.join(SAVE_FOLDER, test_scene, 'vis_plane')
        os.makedirs(save_path, exist_ok=True)
        if quality:
            cv2.imwrite(os.path.join(save_path, camera['img_name']), blend_pred)
        else:
            cv2.imwrite(os.path.join(save_path, camera['img_name'].split('.')[0]+'_bad.png'), blend_pred)
        """
        semantic_info = {}
        planes = []
        planeSegmentation = np.zeros(plane_instance_id.shape)
        semantic_id = 1
        for instance_id in sorted(np.unique(plane_instance_id)):
            if instance_id == 0:
                continue
            planeSegmentation[plane_instance_id == instance_id] = semantic_id
            semantic_info[semantic_id] = {'plane_instance_id': instance_id}
            semantic_id += 1
            planes.append(plane_params[instance_id - 1])
        planes = np.array(planes)
        if len(planes) == 0:
            quality = False
        else:
            planes = get_plane_params_in_local(planes, camera)
        summary = {
            'planes': planes,
            'planeSegmentation': planeSegmentation,
            'backgroundidx': 0,
            'numPlanes': len(planes),
            'semantic_info': semantic_info,
            'good_quality': quality,
        }
        # with open(os.path.join(DATASET_DIR, 'planes_from_ply', test_scene, camera['img_name'].split('.')[0]+'.pkl'), 'wb') as f:
        # Save Plane
        save_path = os.path.join(SAVE_FOLDER, 'planes_ply_mp3dcoord_refined', test_scene, 'planes_from_ply')
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, camera['img_name'].split('.')[0]+'.pkl'), 'wb') as f:
            pickle.dump(summary, f)
        # Save RGB
        save_path = os.path.join(SAVE_FOLDER, 'rgb', test_scene, camera['img_name'].split('.')[0]+'.png')
        rgb_img = Image.fromarray(observations["color_sensor"], mode="RGBA")
        rgb_img.save(save_path)
                            
        # Save observations
        obs_f = os.path.join(SAVE_FOLDER, 'observations', test_scene, camera['img_name'].split('.')[0]+'.pkl')
        with open(obs_f, 'wb') as f:
            pickle.dump(observations, f)
        

        
    sim.close()

def colors():
    return np.array([[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255],
                   [80, 128, 255],
                   [255, 230, 180],
                   [255, 0, 255],
                   [0, 255, 255],
                   [100, 0, 0],
                   [0, 100, 0],
                   [255, 255, 0],
                   [50, 150, 0],
                   [200, 255, 255],
                   [255, 200, 255],
                   [128, 128, 80],
                   # [0, 50, 128],
                   # [0, 100, 100],
                   [0, 255, 128],
                   [0, 128, 255],
                   [255, 0, 128],
                   [128, 0, 255],
                   [255, 128, 0],
                   [128, 255, 0],
                   [0, 0, 0]
                   ])

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors

def plot_category(blend_pred, segmentation, semantic_info):
    import scipy.ndimage as ndimage
    def get_centers(segmentation):
        """
        input: segmentation map, 20 is background
        output: center of mass of each segment
        """
        centers = []
        for i in sorted(np.unique(segmentation)):
            mask = segmentation == i
            centers.append(np.array(ndimage.measurements.center_of_mass(mask))[::-1])
        return centers
    centers = get_centers(segmentation)
    for idx, segm_id in enumerate(sorted(np.unique(segmentation))):
        cv2.putText(blend_pred, semantic_info[segm_id]['category_name'], tuple(centers[idx].astype(np.int32)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    return blend_pred 

def main():
    np.random.seed(2020)
    scene_ids = sorted(os.listdir(ROOT_FOLDER))
    try: 
        scene_ids.remove('sens')
    except:
        pass
    scene_ids = scene_ids[args.base:args.base+args.num]
    assert(len(scene_ids) == args.num)

    for scene_id in tqdm(scene_ids, desc=f"process_id {args.id}"):
        print('rendering', scene_id)
        render_scene(scene_id)


def check_renderings():
    scene_ids = sorted(os.listdir(ROOT_FOLDER))
    try: 
        scene_ids.remove('sens')
    except:
        pass
    scene_ids = scene_ids[args.base:args.base+args.num]
    assert(len(scene_ids) == args.num)

    for scene_id in tqdm(scene_ids, desc=f"process_id {args.id}"):
        camera_path = os.path.join(DATASET_DIR, 'cameras', scene_id + '.pkl')
        with open(camera_path, 'rb') as f:
            cameras = pickle.load(f)
        for camera in cameras:
            save_path = os.path.join(SAVE_FOLDER, scene_id, 'planes_from_ply')
            check_exist_file = os.path.join(save_path, camera['img_name'].split('.')[0]+'.pkl')
            assert(os.path.exists(check_exist_file))
    
        
                    
if __name__ == '__main__':
    main()
    # check_renderings()