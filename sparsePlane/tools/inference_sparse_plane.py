import numpy as np
import argparse, os, cv2, torch, pickle, quaternion
import pycocotools.mask as mask_util
from collections import defaultdict
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.ndimage.measurements import center_of_mass
from scipy.special import softmax
from scipy.optimize import least_squares
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from pytorch3d.structures import join_meshes_as_batch

from sparseplane.config import get_sparseplane_cfg_defaults
from sparseplane.modeling.roi_heads.plane_loss import GeoConsistencyLoss
from sparseplane.utils.mesh_utils import (
    save_obj,
    get_camera_meshes,
    transform_meshes,
    rotate_mesh_for_webview,
    get_plane_params_in_global,
    get_plane_params_in_local,
)
from sparseplane.utils.vis import get_single_image_mesh_plane
from sparseplane.visualization import create_instances, get_labeled_seg, draw_match

import KMSolver
from local_refinement_sift import (
    get_pixel_matching,
    vec6dToSo3,
    rotation_matrix_from_array,
    so3ToVec6d,
    fun_with_precalculated_sift_reduce_rot,
)


def km_solver(distance_matrix, weight):
    """
    km: Hungarian Algo
    if the distance > threshold, even it is smallest, it is also false.
    """
    cost_matrix = (distance_matrix.numpy() * 1000).astype(np.int)
    prediction_matrix_km = KMSolver.solve(
        cost_matrix, threshold=int((1 - weight["threshold"]) * 1000)
    )
    return prediction_matrix_km


class PlaneRCNN_Branch:
    def __init__(self, cfg, cpu_device="cpu"):
        self.predictor = DefaultPredictor(cfg)
        self._cpu_device = cpu_device
        self._K_inv_dot_xy_1 = torch.FloatTensor(self.get_K_inv_dot_xy_1()).to("cuda")
        self._camera_on = cfg.MODEL.CAMERA_ON
        self._embedding_on = cfg.MODEL.EMBEDDING_ON
        self.img_format = cfg.INPUT.FORMAT

    def inference(
        self,
        img_file1,
        img_file2,
    ):
        """
        input: im0, im1 path.
        """
        im0 = utils.read_image(img_file1, format=self.img_format)
        im1 = utils.read_image(img_file2, format=self.img_format)
        # Equivalent
        # im0 = cv2.imread(img_file1)
        # im1 = cv2.imread(img_file2)

        im0 = cv2.resize(im0, (640, 480))
        im1 = cv2.resize(im1, (640, 480))

        im0 = torch.as_tensor(im0.transpose(2, 0, 1).astype("float32"))
        im1 = torch.as_tensor(im1.transpose(2, 0, 1).astype("float32"))
        with torch.no_grad():
            pred = self.predictor.model([{"0": {"image": im0}, "1": {"image": im1}}])[0]
        return pred

    def process(self, output):
        prediction = {"0": {}, "1": {}}
        tmp_instances = {"0": {}, "1": {}}
        for i in range(2):
            if "instances" in output[str(i)]:
                instances = output[str(i)]["instances"].to(self._cpu_device)
                prediction[str(i)]["instances"] = instances_to_coco_json(
                    instances, "demo"
                )
                prediction[str(i)]["pred_plane"] = output[str(i)][
                    "instances"
                ].pred_plane.to(self._cpu_device)
                tmp_instances[str(i)]["embeddingbox"] = {
                    "pred_boxes": instances.pred_boxes,
                    "scores": instances.scores,
                }
            if "proposals" in output[str(i)]:
                prediction[str(i)]["proposals"] = output[str(i)]["proposals"].to(
                    self._cpu_device
                )
            if output["depth"][str(i)] is not None:
                prediction[str(i)]["pred_depth"] = output["depth"][str(i)].to(
                    self._cpu_device
                )
                xyz = self.depth2XYZ(output["depth"][str(i)])
                prediction[str(i)] = self.override_depth(xyz, prediction[str(i)])
        if self._embedding_on:
            if "pred_aff" in output:
                tmp_instances["pred_aff"] = output["pred_aff"].to(self._cpu_device)
            if "geo_aff" in output:
                tmp_instances["geo_aff"] = output["geo_aff"].to(self._cpu_device)
            if "emb_aff" in output:
                tmp_instances["emb_aff"] = output["emb_aff"].to(self._cpu_device)
            prediction["corrs"] = tmp_instances
        if self._camera_on:
            camera_dict = {
                "logits": {
                    "tran": output["camera"]["tran"].to(self._cpu_device),
                    "rot": output["camera"]["rot"].to(self._cpu_device),
                },
                "logits_sms": {
                    "tran": softmax(output["camera"]["tran"].to(self._cpu_device)),
                    "rot": softmax(output["camera"]["rot"].to(self._cpu_device)),
                },
            }
            prediction["camera"] = camera_dict
        return prediction

    def depth2XYZ(self, depth):
        """
        Convert depth to point clouds
        X - width
        Y - depth
        Z - height
        """
        XYZ = self._K_inv_dot_xy_1 * depth
        return XYZ

    @staticmethod
    def get_K_inv_dot_xy_1(h=480, w=640):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x], [0, focal_length, offset_y], [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))

        K_inv_dot_xy_1 = np.zeros((3, h, w))

        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = np.dot(K_inv, np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]
        return K_inv_dot_xy_1.reshape(3, h, w)

    @staticmethod
    def override_depth(xyz, instance):
        pred_masks = [p["segmentation"] for p in instance["instances"]]
        override_list = []
        for mask, plane in zip(pred_masks, instance["pred_plane"]):
            bimask = mask_util.decode(mask)
            if bimask.sum() == 0:
                override_list.append(plane)
                continue
            xyz_tmp = xyz[:, torch.BoolTensor(bimask)]
            offset = np.linalg.norm(plane)
            normal = plane / max(offset, 1e-8)
            offset_new = (normal @ xyz_tmp.cpu().numpy()).mean()
            override_list.append(normal * offset_new)
        if len(override_list) > 0:
            instance["pred_plane"] = torch.stack(override_list)
        return instance


class Camera_Branch:
    def __init__(self, d2_cfg):
        self.cfg = d2_cfg
        if self.cfg.MODEL.CAMERA_ON:
            with open(self.cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH, "rb") as f:
                self.kmeans_trans = pickle.load(f)
            with open(self.cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH, "rb") as f:
                self.kmeans_rots = pickle.load(f)

    def xyz2class(self, x, y, z):
        return self.kmeans_trans.predict([[x, y, z]])

    def quat2class(self, w, xi, yi, zi):
        return self.kmeans_rots.predict([[w, xi, yi, zi]])

    def class2xyz(self, cls):
        assert (cls >= 0).all() and (cls < self.kmeans_trans.n_clusters).all()
        return self.kmeans_trans.cluster_centers_[cls]

    def class2quat(self, cls):
        assert (cls >= 0).all() and (cls < self.kmeans_rots.n_clusters).all()
        return self.kmeans_rots.cluster_centers_[cls]

    def get_rel_camera(self, pred_dict, tran_topk=0, rot_topk=0):

        sorted_idx_tran = np.argsort(pred_dict["camera"]["logits"]["tran"].numpy())[
            ::-1
        ]
        sorted_idx_rot = np.argsort(pred_dict["camera"]["logits"]["rot"].numpy())[::-1]
        tran = self.class2xyz(sorted_idx_tran[tran_topk])
        tran_p = pred_dict["camera"]["logits_sms"]["tran"][sorted_idx_tran[tran_topk]]
        rot = self.class2quat(sorted_idx_rot[rot_topk])
        rot_p = pred_dict["camera"]["logits_sms"]["rot"][sorted_idx_rot[rot_topk]]
        camera_info = {
            "position": tran,
            "position_prob": tran_p,
            "rotation": rot,
            "rotation_prob": rot_p,
        }
        return camera_info


class Discrete_Optimizer:
    def __init__(self, cfg):
        self.weight = {
            "threshold": 0.7,
            "lambda_emb": 0.47,
            "lambda_geo_l2": 0.00,
            "l2_clamp": 5,
            "lambda_geo_normal": 0.25,
            "lambda_geo_offset": 0.28,
            "offset_clamp": 4,
            "topk_tran": 32,
            "topk_rot": 32,
            # [assignment.sum(), pred_cam['position_prob'], pred_cam['rotation_prob'], (embedding_matrix*assignment).numpy().mean(),
            #              (l2_matrix*assignment).numpy().mean(), (normal_matrix*assignment).numpy().mean(), (offset_matrix*assignment).numpy().mean(),
            # [assignment.sum(), log(pcam_tran), log(pcam_rot), distance*assignment]
            "score_weight": [0.311, 0.166, 0.092, -0.432],
            "assignment": "km_search_cam",
        }
        # Initialize camera
        self.camera_branch = Camera_Branch(d2_cfg=cfg)
        # class for geometric distance
        self.geo_consistency_loss = GeoConsistencyLoss("cpu")

    def optimize(self, pred_dict):
        embedding_matrix = 1 - pred_dict["corrs"]["pred_aff"]
        weight = self.weight
        # discrete optimization
        best_score = np.NINF
        best_assignment = None
        best_camera = None
        best_tran_topk = None
        best_rot_topk = None
        best_distance_m = None
        score_weight = np.array(weight["score_weight"]).reshape(-1, 1)
        for k_tran in range(weight["topk_tran"]):
            for k_rot in range(weight["topk_rot"]):
                pred_cam = self.camera_branch.get_rel_camera(pred_dict, k_tran, k_rot)

                geo_matrix = defaultdict(dict)
                # l2
                (
                    geo_distance_matrix,
                    numPlanes1,
                    numPlanes2,
                ) = self.geo_consistency_loss.inference(
                    [pred_dict["0"]], [pred_dict["1"]], [pred_cam], distance="l2"
                )
                geo_matrix.update(geo_distance_matrix)
                # normal angle
                (
                    normal_angle_matrix,
                    numPlanes1,
                    numPlanes2,
                ) = self.geo_consistency_loss.inference(
                    [pred_dict["0"]], [pred_dict["1"]], [pred_cam], distance="normal"
                )
                geo_matrix.update(normal_angle_matrix)

                l2_matrix = (
                    np.clip(geo_matrix["l2"], 0, weight["l2_clamp"])
                    / weight["l2_clamp"]
                )
                normal_matrix = geo_matrix["normal"] / np.pi
                offset_matrix = (
                    np.clip(geo_matrix["offset"], 0, weight["offset_clamp"])
                    / weight["offset_clamp"]
                )

                distance_matrix = (
                    weight["lambda_emb"] * embedding_matrix
                    + weight["lambda_geo_l2"] * l2_matrix
                    + weight["lambda_geo_normal"] * normal_matrix
                    + weight["lambda_geo_offset"] * offset_matrix
                )

                assignment = km_solver(distance_matrix[0], weight=weight)
                x = np.array(
                    [
                        assignment.sum(),
                        np.log(pred_cam["position_prob"]),
                        np.log(pred_cam["rotation_prob"]),
                        (distance_matrix * assignment).numpy().mean(),
                    ]
                )
                score = x @ score_weight
                if score > best_score:
                    best_score = score
                    best_assignment = assignment
                    best_distance_m = distance_matrix
                    best_camera = pred_cam
                    best_tran_topk = k_tran
                    best_rot_topk = k_rot
        return {
            "best_camera": best_camera,
            "best_assignment": best_assignment,
            "distance_m": best_distance_m,
            "best_tran_topk": best_tran_topk,
            "best_rot_topk": best_rot_topk,
        }


class Continuous_Optimizer:
    def __init__(self):
        self.weight = {
            "huber_delta": 0.01,
            "lambda_R": 1.0,
        }

    def optimize(self, img_file1, img_file2, pred_dict, optimized_dict):
        """
        Initialize camera pose
        """
        init_R = optimized_dict["best_camera"]["rotation"]
        init_T = optimized_dict["best_camera"]["position"]
        x0 = np.concatenate((so3ToVec6d(rotation_matrix_from_array(init_R)), init_T))
        """
        Select correspondence assignment
        """
        assignment_m = optimized_dict["best_assignment"]
        assignment = np.argwhere(assignment_m)
        """
        Select plane params
        """
        x1_full = np.array(pred_dict["0"]["pred_plane"])
        x2_full = np.array(pred_dict["1"]["pred_plane"])

        if len(assignment) == 0:
            rtn = {
                "n_corr": len(assignment),
                "cost": 0,
                "best_camera": {"position": init_T, "rotation": init_R},
                "best_assignment": assignment_m,
                "plane_param_override": {"0": x1_full, "1": x2_full},
            }
            return rtn

        x1 = x1_full[assignment[:, 0]]
        x2 = x2_full[assignment[:, 1]]

        """
        Select optimized function
        """
        boxes1 = np.array([inst["bbox"] for inst in pred_dict["0"]["instances"]])[
            assignment[:, 0]
        ]
        boxes2 = np.array([inst["bbox"] for inst in pred_dict["1"]["instances"]])[
            assignment[:, 1]
        ]
        segms1 = np.array(
            [inst["segmentation"] for inst in pred_dict["0"]["instances"]]
        )[assignment[:, 0]]
        segms2 = np.array(
            [inst["segmentation"] for inst in pred_dict["1"]["instances"]]
        )[assignment[:, 1]]
        offsets1 = np.linalg.norm(x1, axis=1)
        normals1 = x1 / (offsets1.reshape(-1, 1) + 1e-5)
        offsets2 = np.linalg.norm(x2, axis=1)
        normals2 = x2 / (offsets2.reshape(-1, 1) + 1e-5)

        x0 = np.concatenate((x0, offsets1, offsets2))

        img1 = cv2.imread(img_file1, cv2.IMREAD_COLOR)[:, :, ::-1]
        img2 = cv2.imread(img_file2, cv2.IMREAD_COLOR)[:, :, ::-1]
        img1 = cv2.resize(img1, (640, 480))
        img2 = cv2.resize(img2, (640, 480))
        xys1, xys2 = [], []
        for i in range(len(boxes1)):
            try:
                xy1, xy2 = get_pixel_matching(
                    img1, boxes1[i], segms1[i], x1[i], img2, boxes2[i], segms2[i], x2[i]
                )
            except:
                xy1 = []
                xy2 = []
            xys1.append(np.array(xy1))
            xys2.append(np.array(xy2))

        rst = least_squares(
            fun_with_precalculated_sift_reduce_rot,
            x0,
            args=(
                len(boxes1),
                img1,
                xys1,
                normals1,
                img2,
                xys2,
                normals2,
                rotation_matrix_from_array(init_R),
                self.weight,
            ),
        )

        offsets1 = rst.x[9 : 9 + len(boxes1)]
        offsets2 = rst.x[9 + len(boxes1) : 9 + len(boxes1) * 2]
        x1_full[assignment[:, 0]] = offsets1.reshape(-1, 1) * normals1
        x2_full[assignment[:, 1]] = offsets2.reshape(-1, 1) * normals2

        pred_R = quaternion.as_float_array(
            quaternion.from_rotation_matrix(vec6dToSo3(rst.x[:6]))
        )
        pred_T = rst.x[6:9]
        rtn = {
            "n_corr": len(assignment),
            "cost": rst.cost,
            "best_camera": {"position": pred_T, "rotation": pred_R},
            "best_assignment": assignment_m,
            "plane_param_override": {"0": x1_full, "1": x2_full},
        }
        return rtn


def save_matching(
    img_file1,
    img_file2,
    pred_dict,
    assignment,
    output_dir,
    prefix="",
    paper_img=False,
    score_threshold=0.7,
):
    """
    fp: whether show fp or fn
    gt_box: whether use gtbox
    """
    image_paths = {"0": img_file1, "1": img_file2}
    blended = {}
    # centroids for matching
    centroids = {"0": [], "1": []}

    for i in range(2):
        img = cv2.imread(image_paths[str(i)], cv2.IMREAD_COLOR)[:, :, ::-1]
        img = cv2.resize(img, (640, 480))
        height, width, _ = img.shape
        vis = Visualizer(img)

        p_instance = create_instances(
            pred_dict[str(i)]["instances"],
            img.shape[:2],
            pred_planes=pred_dict[str(i)]["pred_plane"].numpy(),
            conf_threshold=score_threshold,
        )
        seg_blended = get_labeled_seg(
            p_instance, score_threshold, vis, paper_img=paper_img
        )
        blended[str(i)] = seg_blended
        # centroid of mask
        for ann in pred_dict[str(i)]["instances"]:
            M = center_of_mass(mask_util.decode(ann["segmentation"]))
            centroids[str(i)].append(M[::-1])  # reverse for opencv
        centroids[str(i)] = np.array(centroids[str(i)])

    pred_corr_list = np.array(torch.FloatTensor(assignment).nonzero().tolist())

    correct_list_pred = [True for pair in pred_corr_list]
    pred_matching_fig = draw_match(
        blended["0"],
        blended["1"],
        centroids["0"],
        centroids["1"],
        np.array(pred_corr_list),
        correct_list_pred,
        vertical=False,
    )
    os.makedirs(output_dir, exist_ok=True)
    pred_matching_fig.save(os.path.join(output_dir, prefix + ".png"))


def merge_plane_params_from_local_params(plane_locals, corr_list, camera_pose):
    """
    input: plane parameters in camera frame
    output: merged plane parameters using corr_list
    """
    param1, param2 = plane_locals["0"], plane_locals["1"]
    param1_global = get_plane_params_in_global(param1, camera_pose)
    param2_global = get_plane_params_in_global(
        param2, {"position": np.array([0, 0, 0]), "rotation": np.quaternion(1, 0, 0, 0)}
    )
    param1_global, param2_global = merge_plane_params_from_global_params(
        param1_global, param2_global, corr_list
    )
    param1 = get_plane_params_in_local(param1_global, camera_pose)
    param2 = get_plane_params_in_local(
        param2_global,
        {"position": np.array([0, 0, 0]), "rotation": np.quaternion(1, 0, 0, 0)},
    )
    return {"0": param1, "1": param2}


def merge_plane_params_from_global_params(param1, param2, corr_list):
    """
    input: plane parameters in global frame
    output: merged plane parameters using corr_list
    """
    pred = {"0": {}, "1": {}}
    pred["0"]["offset"] = np.maximum(
        np.linalg.norm(param1, ord=2, axis=1), 1e-5
    ).reshape(-1, 1)
    pred["0"]["normal"] = param1 / pred["0"]["offset"]
    pred["1"]["offset"] = np.maximum(
        np.linalg.norm(param2, ord=2, axis=1), 1e-5
    ).reshape(-1, 1)
    pred["1"]["normal"] = param2 / pred["1"]["offset"]
    for ann_id in corr_list:
        # average normal
        normal_pair = np.vstack(
            (pred["0"]["normal"][ann_id[0]], pred["1"]["normal"][ann_id[1]])
        )
        w, v = eigh(normal_pair.T @ normal_pair)
        avg_normals = v[:, np.argmax(w)]
        if (avg_normals @ normal_pair.T).sum() < 0:
            avg_normals = -avg_normals
        # average offset
        avg_offset = (
            pred["0"]["offset"][ann_id[0]] + pred["1"]["offset"][ann_id[1]]
        ) / 2
        avg_plane = avg_normals * avg_offset
        param1[ann_id[0]] = avg_plane
        param2[ann_id[1]] = avg_plane
    return param1, param2


def save_pair_objects(
    img_file1,
    img_file2,
    p_instances,
    output_dir,
    prefix="",
    pred_camera=None,
    plane_param_override=None,
    show_camera=True,
    corr_list=[],
    webvis=False,
):
    """
    if tran_topk == -2 and rot_topk == -2, then pred_camera should not be None, this is used for non-binned camera.
    if exclude is not None, exclude some instances to make fig 2.
    idx=7867
    exclude = {
        '0': [2,3,4,5,6,7],
        '1': [0,1,2,4,5,6,7],
    }
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

    # Merge planes if they are in correspondence
    if len(corr_list) != 0:
        plane_locals = merge_plane_params_from_local_params(
            plane_locals, corr_list, camera1to2
        )

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
            height=480,
            width=640,
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


def get_parser():
    parser = argparse.ArgumentParser(description="SparsePlane Demo")
    parser.add_argument(
        "--config-file",
        default="./tools/demo/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="./tools/demo/teaser",
        help="A path to a folder of input images",
    )
    parser.add_argument(
        "--img-list", default=None, help="A path to a text file for inference"
    )
    parser.add_argument(
        "--output", default="./debug", help="A directory to save output visualizations"
    )
    return parser


def inference_pair(output_dir, model, dis_opt, con_opt, im0, im1):
    """
    Network inference on a single pair of images.
    """
    pred = model.inference(im0, im1)
    pred_dict = model.process(pred)
    # save segmentation only
    image_paths = {"0": im0, "1": im1}
    p_instances = {}
    for i in range(2):
        img = cv2.imread(image_paths[str(i)], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 480))
        vis = Visualizer(img)

        p_instance = create_instances(
            pred_dict[str(i)]["instances"],
            img.shape[:2],
            pred_planes=pred_dict[str(i)]["pred_plane"].numpy(),
            conf_threshold=0.7,
        )
        p_instances[str(i)] = p_instance
        seg_blended = get_labeled_seg(p_instance, 0.7, vis, paper_img=True)
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"view{i}_pred.jpg"), seg_blended)
        cv2.imwrite(os.path.join(output_dir, f"view{i}.jpg"), img)

    # Optimize
    optimized_dict = dis_opt.optimize(pred_dict)
    optimized_dict = con_opt.optimize(im0, im1, pred_dict, optimized_dict)

    # visualize
    save_matching(
        im0,
        im1,
        pred_dict,
        optimized_dict["best_assignment"],
        output_dir,
        prefix="corr",
        paper_img=True,
    )
    # save original image (resized)
    cv2.imwrite(
        os.path.join(output_dir, "view0.jpg"), cv2.resize(cv2.imread(im0), (640, 480))
    )
    cv2.imwrite(
        os.path.join(output_dir, "view1.jpg"), cv2.resize(cv2.imread(im1), (640, 480))
    )
    # save obj
    save_pair_objects(
        os.path.join(output_dir, "view0.jpg"),
        os.path.join(output_dir, "view1.jpg"),
        p_instances,
        os.path.join(output_dir),
        prefix="refined",
        pred_camera=optimized_dict["best_camera"],
        plane_param_override=optimized_dict["plane_param_override"],
        show_camera=True,
        corr_list=np.argwhere(optimized_dict["best_assignment"]),
        webvis=True,
    )


def main():
    args = get_parser().parse_args()
    # Load cfg
    cfg = get_cfg()
    get_sparseplane_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)

    # Initialize network
    model = PlaneRCNN_Branch(cfg)
    # Initialize optimizer
    dis_opt = Discrete_Optimizer(cfg)
    con_opt = Continuous_Optimizer()

    if args.img_list:  # a text file
        f = open(args.img_list)
        lines = f.readlines()
        f.close()
        for line_idx, line in enumerate(tqdm(lines)):
            output_dir = os.path.join(args.output, "{:0>4}".format(line_idx))
            os.makedirs(output_dir, exist_ok=True)
            line = line.strip()
            splits = line.split(" ")
            im0 = os.path.join(args.input, splits[0])
            im1 = os.path.join(args.input, splits[1])
            inference_pair(output_dir, model, dis_opt, con_opt, im0, im1)

    else:  # a directory
        input_dir = args.input
        output_dir = args.output
        im0 = os.path.join(input_dir, "view_0.png")
        im1 = os.path.join(input_dir, "view_1.png")
        inference_pair(output_dir, model, dis_opt, con_opt, im0, im1)


if __name__ == "__main__":
    main()
