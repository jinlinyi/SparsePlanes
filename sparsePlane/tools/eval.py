import argparse
import numpy as np
import time
import torch
import os
import pickle
import quaternion
from tqdm import tqdm
from scipy.linalg import eigh
import multiprocessing
from multiprocessing import Pool, Process, Queue
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from sparseplane.modeling.meta_arch.camera_branch import Camera_Branch
from sparseplane.modeling.roi_heads.plane_loss import GeoConsistencyLoss
from sparseplane.utils.mesh_utils import get_plane_params_in_global
from sparseplane.visualization import create_instances
from sparseplane.config import get_sparseplane_cfg_defaults
from sparseplane.data import PlaneRCNNMapper
from inference_sparse_plane import Discrete_Optimizer, Continuous_Optimizer


EP_mask_delta_thresh =      [0.5,   0.5,        0.5,        0.,]
EP_normal_delta_thresh =    [30.,   30.,        1000.,      30.,]
EP_offset_delta_thresh =    [1.,    1000.,      1.,         1.,]
EP_ap_str =                 ['all', '-offset',  '-normal',  '-mask']


def setup(args):
    cfg = get_cfg()
    get_sparseplane_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def angle_error_tran_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_IPAA(pred_assignment_m, gt_assignment_list, IPAA_dict):
    wrong_count = 0
    gt_assignment_list = np.array(gt_assignment_list)
    if len(gt_assignment_list) != 0:
        common_row_idxs = gt_assignment_list[:, 0]
        common_colomn_idxs = gt_assignment_list[:, 1]
    else:
        common_row_idxs = []
        common_colomn_idxs = []
    if len(gt_assignment_list) != 0:
        for [row, column] in gt_assignment_list:
            if pred_assignment_m[row, column] != 1:
                wrong_count += 1
    for i in range(pred_assignment_m.shape[0]):
        if i not in common_row_idxs:
            if sum(pred_assignment_m[i, :]) != 0:
                wrong_count += 1
    for j in range(pred_assignment_m.shape[1]):
        if j not in common_colomn_idxs:
            if sum(pred_assignment_m[:, j]) != 0:
                wrong_count += 1
    p = float(wrong_count) / (
        pred_assignment_m.shape[0]
        + pred_assignment_m.shape[1]
        - len(gt_assignment_list)
    )
    for key in IPAA_dict.keys():
        if (1 - p) * 100 >= key:
            IPAA_dict[key] += 1


class Evaluator:
    def __init__(self, args, score_threshold=0.7, dataset="mp3d_test"):
        cfg = setup(args)
        self.score_threshold = score_threshold

        self.gt_box = cfg.TEST.EVAL_GT_BOX
        rcnn_cached_file = args.rcnn_cached_file
        with open(rcnn_cached_file, "rb") as f:
            self.rcnn_data = torch.load(f)
        if not self.gt_box:
            pass
        else:
            if "file_name" not in self.rcnn_data[0]["0"].keys():
                for idx in range(len(self.rcnn_data)):
                    for i in range(2):
                        self.rcnn_data[idx][str(i)][
                            "file_name"
                        ] = file_name_from_image_id(
                            self.rcnn_data[idx][str(i)]["image_id"]
                        )
        self.camera_branch = Camera_Branch(d2_cfg=cfg, rpnet_args=args)
        self.geo_consistency_loss = GeoConsistencyLoss("cpu")
        self.metadata = MetadataCatalog.get(dataset)
        self.load_input_dataset(dataset)
        self.dis_opt = Discrete_Optimizer(cfg)
        self.con_opt = Continuous_Optimizer()
        self.sanity_check()

    def sanity_check(self):
        for idx, key in enumerate(self.dataset_dict.keys()):
            assert self.rcnnidx2datasetkey(idx) == key

    def rcnnidx2datasetkey(self, idx):
        key0 = self.rcnn_data[idx]["0"]["image_id"]
        key1 = self.rcnn_data[idx]["1"]["image_id"]
        key = key0 + "__" + key1
        return key

    def load_input_dataset(self, dataset):
        dataset_dict = {}
        dataset_list = list(DatasetCatalog.get(dataset))
        for dic in dataset_list:
            key0 = dic["0"]["image_id"]
            key1 = dic["1"]["image_id"]
            key = key0 + "__" + key1
            for i in range(len(dic["0"]["annotations"])):
                dic["0"]["annotations"][i]["bbox_mode"] = BoxMode(
                    dic["0"]["annotations"][i]["bbox_mode"]
                )
            for i in range(len(dic["1"]["annotations"])):
                dic["1"]["annotations"][i]["bbox_mode"] = BoxMode(
                    dic["1"]["annotations"][i]["bbox_mode"]
                )
            dataset_dict[key] = dic
        self.dataset_dict = dataset_dict

    def get_camera_info(self, idx, tran_topk, rot_topk):
        # if topk is -1, then use gt pose (NOT GT BINs!)
        return self.camera_branch.get_rel_camera(
            [self.rcnn_data[idx]], tran_topk, rot_topk
        )[0]

    def get_gt_affinity(self, idx, rtnformat="matrix", gtbox=True):
        """
        return gt affinity.
        If gtbox is True, return gt affinity for gt boxes;
        else return gt affinity for pred boxes.
        """
        if gtbox:
            key0 = self.rcnn_data[idx]["0"]["image_id"]
            key1 = self.rcnn_data[idx]["1"]["image_id"]
            key = key0 + "__" + key1
            corrlist = np.array(self.dataset_dict[key]["gt_corrs"])
        else:
            corrlist = self.get_gt_affinity_from_pred_box(idx)
        if rtnformat == "list":
            return corrlist
        elif rtnformat == "matrix":
            if gtbox:
                mat = torch.zeros(
                    (
                        len(self.dataset_dict[key]["0"]["annotations"]),
                        len(self.dataset_dict[key]["1"]["annotations"]),
                    )
                )
            else:
                mat = torch.zeros(
                    (
                        len(self.rcnn_data[idx]["0"]["instances"]),
                        len(self.rcnn_data[idx]["1"]["instances"]),
                    )
                )
            for i in corrlist:
                mat[i[0], i[1]] = 1
            return mat
        else:
            raise NotImplementedError

    def evaluate_camera(self, return_dict=None):
        tran_errs = []
        tran_angle_errs = []
        rot_errs = []
        for idx in tqdm(range(len(self.rcnn_data))):
            gt_cam = self.get_camera_info(idx, -1, -1)
            if return_dict is None:
                pred_cam = self.get_camera_info(idx, 0, 0)
            else:
                pred_cam = return_dict[idx]["best_camera"]
            # Error - translation
            tran_errs.append(np.linalg.norm(pred_cam["position"] - gt_cam["position"]))
            # Error - rotation
            if type(pred_cam["rotation"]) != np.ndarray:
                raise "Need to convert quaternion to np array"
            d = np.abs(np.sum(np.multiply(pred_cam["rotation"], gt_cam["rotation"])))
            d = np.clip(d, -1, 1)
            rot_errs.append(2 * np.arccos(d) * 180 / np.pi)

        tran_acc = sum(_ < 1 for _ in tran_errs) / len(tran_errs)
        rot_acc = sum(_ < 30 for _ in rot_errs) / len(rot_errs)

        median_tran_err = np.median(np.array(tran_errs))
        mean_tran_err = np.mean(np.array(tran_errs))
        median_rot_err = np.median(np.array(rot_errs))
        mean_rot_err = np.mean(np.array(rot_errs))

        print(
            "Mean Error [tran, rot]: {:.2f}, {:.2f}".format(mean_tran_err, mean_rot_err)
        )
        print(
            "Median Error [tran, rot]: {:.2f}, {:.2f}".format(
                median_tran_err, median_rot_err
            )
        )
        print(
            "Accuracy [tran, rot]: {:.1f}, {:.1f}".format(tran_acc * 100, rot_acc * 100)
        )

        camera_eval_dict = {
            "tran_errs": np.array(tran_errs),
            "rot_errs": np.array(rot_errs),
            "mean_tran_err": mean_tran_err,
            "mean_rot_err": mean_rot_err,
            "median_tran_err": median_tran_err,
            "median_rot_err": median_rot_err,
            "tran_acc": tran_acc,
            "rot_acc": rot_acc,
        }
        return camera_eval_dict

    def evaluate_matching(self, return_dict):
        IPAA_dict = {}
        for i in range(11):
            IPAA_dict[i * 10] = 0
        for idx in sorted(return_dict.keys()):
            pred_assignment = return_dict[idx]["best_assignment"]
            gt_assignment_list = self.get_gt_affinity(
                idx, rtnformat="list", gtbox=self.gt_box
            )
            compute_IPAA(pred_assignment, gt_assignment_list, IPAA_dict)
        print("[IPAA_dict]")
        for key in IPAA_dict.keys():
            IPAA_dict[key] /= len(self.rcnn_data)
        print(IPAA_dict)
        return IPAA_dict

    def evaluate_ap_by_idx(self, idx):
        """
        get plane errors and mask errors
        """
        key = self.rcnnidx2datasetkey(idx)

        if "plane_param_override" not in self.optimized_dict[idx].keys():
            self.optimized_dict[idx]["plane_param_override"] = None
        pred_corr = np.argwhere(self.optimized_dict[idx]["best_assignment"])
        tran_topk = -2
        rot_topk = -2
        pred_camera = self.optimized_dict[idx]["best_camera"]
        plane_param_override = self.optimized_dict[idx]["plane_param_override"]
        """
        PRED
        """
        # Load predict camera
        if pred_camera is None:
            pred_camera = self.get_camera_info(idx, tran_topk, rot_topk)
            pred_camera = {
                "position": np.array(pred_camera["position"]),
                "rotation": quaternion.from_float_array(pred_camera["rotation"]),
            }
        else:
            assert tran_topk == -2 and rot_topk == -2
            pred_camera = {
                "position": np.array(pred_camera["position"]),
                "rotation": quaternion.from_float_array(pred_camera["rotation"]),
            }

        # Load single view prediction
        pred = {
            "0": {},
            "1": {},
            "merged": {},
            "corrs": pred_corr,
            "camera": pred_camera,
            "0_local": {},
            "1_local": {},
        }
        for i in range(2):
            if i == 0:
                camera_info = pred_camera
            else:
                camera_info = {
                    "position": np.array([0, 0, 0]),
                    "rotation": np.quaternion(1, 0, 0, 0),
                }
            p_instance = create_instances(
                self.rcnn_data[idx][str(i)]["instances"],
                [
                    self.dataset_dict[key][str(i)]["height"],
                    self.dataset_dict[key][str(i)]["width"],
                ],
                pred_planes=self.rcnn_data[idx][str(i)]["pred_plane"].numpy(),
                conf_threshold=self.score_threshold,
            )
            if plane_param_override is None:
                pred_plane_single = p_instance.pred_planes
            else:
                pred_plane_single = plane_param_override[str(i)]
            # Local frame
            offset = np.maximum(
                np.linalg.norm(pred_plane_single, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)
            normal = pred_plane_single / offset
            pred[str(i) + "_local"]["offset"] = offset
            pred[str(i) + "_local"]["normal"] = normal
            pred[str(i) + "_local"]["scores"] = p_instance.scores

            # Global frame
            plane_global = get_plane_params_in_global(pred_plane_single, camera_info)
            offset = np.maximum(
                np.linalg.norm(plane_global, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)
            normal = plane_global / offset
            pred[str(i)]["offset"] = offset
            pred[str(i)]["normal"] = normal
            pred[str(i)]["scores"] = p_instance.scores
        # Merge prediction
        merged_offset = []
        merged_normal = []
        merged_score = []
        for i in range(2):
            for ann_id in range(len(pred[str(i)]["scores"])):
                if len(pred["corrs"]) == 0 or ann_id not in pred["corrs"][:, i]:
                    merged_offset.append(pred[str(i)]["offset"][ann_id])
                    merged_normal.append(pred[str(i)]["normal"][ann_id])
                    merged_score.append(pred[str(i)]["scores"][ann_id])
        for ann_id in pred["corrs"]:
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
            merged_offset.append(avg_offset)
            merged_normal.append(avg_normals)
            # max score
            merged_score.append(
                max(pred["0"]["scores"][ann_id[0]], pred["1"]["scores"][ann_id[1]])
            )
        pred["merged"] = {
            "merged_offset": np.array(merged_offset),
            "merged_normal": np.array(merged_normal),
            "merged_score": np.array(merged_score)[:, np.newaxis],
        }
        """
        GT
        """
        gt_camera = self.get_camera_info(idx, -1, -1)
        gt_camera = {
            "position": np.array(gt_camera["position"]),
            "rotation": quaternion.from_float_array(gt_camera["rotation"]),
        }
        gt_corr = self.get_gt_affinity(idx, rtnformat="list", gtbox=True)

        # Load single view gt
        gt = {
            "0": {},
            "1": {},
            "merged": {},
            "corrs": gt_corr,
            "camera": gt_camera,
            "0_local": {},
            "1_local": {},
        }
        for i in range(2):
            if i == 0:
                camera_info = gt_camera
            else:
                camera_info = {
                    "position": np.array([0, 0, 0]),
                    "rotation": np.quaternion(1, 0, 0, 0),
                }
            plane_params = np.array(
                [ann["plane"] for ann in self.dataset_dict[key][str(i)]["annotations"]]
            )

            # Local frame
            offset = np.maximum(
                np.linalg.norm(plane_params, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)
            normal = plane_params / offset
            gt[str(i) + "_local"]["offset"] = offset
            gt[str(i) + "_local"]["normal"] = normal

            # Global frame
            plane_global = get_plane_params_in_global(plane_params, camera_info)
            offset = np.maximum(
                np.linalg.norm(plane_global, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)
            normal = plane_global / offset
            gt[str(i)]["offset"] = offset
            gt[str(i)]["normal"] = normal
        # Merge gt
        merged_offset = []
        merged_normal = []
        for i in range(2):
            for ann_id in range(len(gt[str(i)]["offset"])):
                if len(gt["corrs"]) == 0 or ann_id not in gt["corrs"][:, i]:
                    merged_offset.append(gt[str(i)]["offset"][ann_id])
                    merged_normal.append(gt[str(i)]["normal"][ann_id])
        for ann_id in gt["corrs"]:
            # average normal
            assert (
                np.linalg.norm(
                    gt["0"]["normal"][ann_id[0]] - gt["1"]["normal"][ann_id[1]]
                )
                < 1e-5
            )
            assert (
                np.abs(gt["0"]["offset"][ann_id[0]] - gt["1"]["offset"][ann_id[1]])
                < 1e-5
            )
            merged_offset.append(gt["0"]["offset"][ann_id[0]])
            merged_normal.append(gt["0"]["normal"][ann_id[0]])
        gt["merged"] = {
            "merged_offset": np.array(merged_offset),
            "merged_normal": np.array(merged_normal),
        }
        """
        ERRORs
        """
        # compute individual error in its own frame
        individual_error_offset = {}
        individual_error_normal = {}
        for i in range(2):
            individual_error_offset[str(i)] = np.abs(
                pred[str(i) + "_local"]["offset"] - gt[str(i) + "_local"]["offset"].T
            )
            individual_error_normal[str(i)] = (
                np.arccos(
                    np.clip(
                        np.abs(
                            pred[str(i) + "_local"]["normal"]
                            @ gt[str(i) + "_local"]["normal"].T
                        ),
                        -1,
                        1,
                    )
                )
                / np.pi
                * 180
            )
        individual_miou = self.get_maskiou(idx)

        # compute merged error
        err_offsets = np.abs(
            pred["merged"]["merged_offset"] - gt["merged"]["merged_offset"].T
        )
        err_normals = (
            np.arccos(
                np.clip(
                    np.abs(
                        pred["merged"]["merged_normal"]
                        @ gt["merged"]["merged_normal"].T
                    ),
                    -1,
                    1,
                )
            )
            / np.pi
            * 180
        )
        mask_iou = self.get_maskiou_merged(
            idx, pred_corr=pred["corrs"], gt_corr=gt["corrs"]
        )
        output = {
            "err_offsets": err_offsets,
            "err_normals": err_normals,
            "mask_iou": mask_iou,
            "scores": pred["merged"]["merged_score"],
            "individual_error_offset": individual_error_offset,
            "individual_error_normal": individual_error_normal,
            "individual_miou": individual_miou,
            "individual_score": {
                "0": pred["0"]["scores"].reshape(-1, 1),
                "1": pred["1"]["scores"].reshape(-1, 1),
            },
        }
        return output

    def get_maskiou(self, idx):
        """
        calculate mask iou between predicted mask and gt masks
        """
        key0 = self.rcnn_data[idx]["0"]["image_id"]
        key1 = self.rcnn_data[idx]["1"]["image_id"]
        key = key0 + "__" + key1
        mious = {}
        for i in range(2):
            gt_mask_rles = []
            for ann in self.dataset_dict[key][str(i)]["annotations"]:
                if isinstance(ann["segmentation"], list):
                    polygons = [
                        np.array(p, dtype=np.float64) for p in ann["segmentation"]
                    ]
                    rles = mask_util.frPyObjects(
                        polygons,
                        self.dataset_dict[key][str(i)]["height"],
                        self.dataset_dict[key][str(i)]["width"],
                    )
                    rle = mask_util.merge(rles)
                elif isinstance(ann["segmentation"], dict):  # RLE
                    rle = ann["segmentation"]
                else:
                    raise TypeError(
                        f"Unknown segmentation type {type(ann['segmentation'])}!"
                    )
                gt_mask_rles.append(rle)

            pred_mask_rles = [
                ins["segmentation"] for ins in self.rcnn_data[idx][str(i)]["instances"]
            ]
            miou = mask_util.iou(pred_mask_rles, gt_mask_rles, [0] * len(gt_mask_rles))
            mious[str(i)] = miou
        return mious

    def get_maskiou_merged(self, idx, pred_corr=None, gt_corr=None):
        """
        calculate mask iou between merged pred and merged gt
                gt_1    gt_2    gt_m
        pred_1  miou    0       miou(1)
        pred_2  0       miou    miou(2)
        pred_m  miou(1)  miou(2)  avg_miou(1,2)
        """
        mious = self.get_maskiou(idx)
        single2merge_dict = self.get_single2merge(
            idx, pred_corr=pred_corr, gt_corr=gt_corr
        )

        entry2gt_single_view = single2merge_dict["entry2gt_single_view"]
        gt_single_view2entry = single2merge_dict["gt_single_view2entry"]
        entry2pred_single_view = single2merge_dict["entry2pred_single_view"]
        pred_single_view2entry = single2merge_dict["pred_single_view2entry"]

        num_pred_entry = len(entry2pred_single_view.keys())
        num_gt_entry = len(entry2gt_single_view.keys())
        # pred_gt_merged_mask
        mask_iou = np.zeros((num_pred_entry, num_gt_entry))
        for r in range(num_pred_entry):
            for c in range(num_gt_entry):
                pred_merged = entry2pred_single_view[r]["merged"]
                gt_merged = entry2gt_single_view[c]["merged"]
                pair_id_pred = entry2pred_single_view[r]["pair"]
                pair_id_gt = entry2gt_single_view[c]["pair"]
                ann_id_pred = entry2pred_single_view[r]["ann_id"]
                ann_id_gt = entry2gt_single_view[c]["ann_id"]
                if not pred_merged and not gt_merged:
                    # pred_single & gt_single
                    # Should be in the same image
                    if pair_id_pred != pair_id_gt:
                        continue
                    else:
                        miou_single = mious[pair_id_pred]
                        mask_iou[r][c] = miou_single[ann_id_pred, ann_id_gt]
                elif pred_merged and not gt_merged:
                    # pred_merged & gt_single
                    miou_single = mious[pair_id_gt]
                    mask_iou[r][c] = miou_single[
                        ann_id_pred[int(pair_id_gt)], ann_id_gt
                    ]
                elif not pred_merged and gt_merged:
                    # pred_single & gt_merged
                    miou_single = mious[pair_id_pred]
                    mask_iou[r][c] = miou_single[
                        ann_id_pred, ann_id_gt[int(pair_id_pred)]
                    ]
                elif pred_merged and gt_merged:
                    # pred_merge & gt_merged, average both
                    miou_single = mious[str(0)]
                    iou0 = miou_single[ann_id_pred[0], ann_id_gt[0]]
                    miou_single = mious[str(1)]
                    iou1 = miou_single[ann_id_pred[1], ann_id_gt[1]]
                    mask_iou[r][c] = (iou0 + iou1) / 2
                else:
                    raise "BUG"
        return mask_iou

    def get_single2merge(self, idx, pred_corr=None, gt_corr=None):
        key = self.rcnnidx2datasetkey(idx)
        # GT merged mapping
        entry2gt_single_view = {}
        gt_single_view2entry = {"0": {}, "1": {}}
        if gt_corr is not None:
            gt_entry_id = 0
            for i in range(2):
                single_gt_idx = len(self.dataset_dict[key][str(i)]["annotations"])
                for s_i in range(single_gt_idx):
                    if s_i not in gt_corr[:, i]:
                        entry2gt_single_view[gt_entry_id] = {
                            "pair": str(i),
                            "ann_id": s_i,
                            "merged": False,
                        }
                        gt_single_view2entry[str(i)][s_i] = gt_entry_id
                        gt_entry_id += 1
            for pair in gt_corr:
                entry2gt_single_view[gt_entry_id] = {
                    "pair": ["0", "1"],
                    "ann_id": pair,
                    "merged": True,
                }
                gt_single_view2entry["0"][pair[0]] = gt_entry_id
                gt_single_view2entry["1"][pair[1]] = gt_entry_id
                gt_entry_id += 1

        # Pred merged mapping
        entry2pred_single_view = {}
        pred_single_view2entry = {"0": {}, "1": {}}
        if pred_corr is not None:
            pred_entry_id = 0
            for i in range(2):
                single_idx = len(self.rcnn_data[idx][str(i)]["pred_plane"])
                for s_i in range(single_idx):
                    if len(pred_corr) == 0 or s_i not in pred_corr[:, i]:
                        entry2pred_single_view[pred_entry_id] = {
                            "pair": str(i),
                            "ann_id": s_i,
                            "merged": False,
                        }
                        pred_single_view2entry[str(i)][s_i] = pred_entry_id
                        pred_entry_id += 1
            for pair in pred_corr:
                entry2pred_single_view[pred_entry_id] = {
                    "pair": ["0", "1"],
                    "ann_id": pair,
                    "merged": True,
                }
                pred_single_view2entry["0"][pair[0]] = pred_entry_id
                pred_single_view2entry["1"][pair[1]] = pred_entry_id
                pred_entry_id += 1
        return {
            "entry2gt_single_view": entry2gt_single_view,
            "gt_single_view2entry": gt_single_view2entry,
            "entry2pred_single_view": entry2pred_single_view,
            "pred_single_view2entry": pred_single_view2entry,
        }

    def optimize_by_idx(self, idx, evaluate):
        optimized_dict = self.dis_opt.optimize(self.rcnn_data[idx])
        if evaluate == "correspondence":
            return optimized_dict
        elif method in ["AP", "camera"]:
            im0 = self.rcnn_data[idx]["0"]["file_name"]
            im1 = self.rcnn_data[idx]["1"]["file_name"]
            optimized_dict = self.con_opt.optimize(
                im0, im1, self.rcnn_data[idx], optimized_dict
            )
            return optimized_dict
        else:
            raise NotImplementedError

    def optimize_by_list(self, idxs, return_dict, evaluate):
        for idx in idxs:
            rtn = self.optimize_by_idx(idx, evaluate)
            return_dict[idx] = rtn

    def evaluate_by_list(self, idxs, return_dict):
        for idx in idxs:
            rtn = self.evaluate_ap_by_idx(idx)
            return_dict[idx] = rtn


def multiprocess_by_list(ev, num_process, idx_list, evaluate, optimize=True):
    max_iter = len(idx_list)
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    per_thread = int(np.ceil(max_iter / num_process))
    split_by_thread = [
        idx_list[i * per_thread : (i + 1) * per_thread] for i in range(num_process)
    ]
    for i in range(num_process):
        if optimize:
            p = Process(
                target=ev.optimize_by_list,
                args=(split_by_thread[i], return_dict, evaluate),
            )
        else:
            p = Process(
                target=ev.evaluate_by_list, args=(split_by_thread[i], return_dict)
            )
        p.start()
        jobs.append(p)

    prev = 0
    with tqdm(total=max_iter) as pbar:
        while True:
            time.sleep(0.1)
            curr = len(return_dict.keys())
            pbar.update(curr - prev)
            prev = curr
            if curr == max_iter:
                break

    for job in jobs:
        job.join()

    return return_dict


def save_dict(return_dict, folder, prefix=None):
    os.makedirs(folder, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if prefix is None:
        save_path = os.path.join(folder, f"optimized_{timestr}.pkl")
    else:
        save_path = os.path.join(folder, prefix + ".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(return_dict.copy(), f)


def evaluate_by_idx(eval_dict):
    ndt, ngt = eval_dict["mask_iou"].shape
    if ndt == 0:
        stats = []
        for i in range(len(EP_ap_str)):
            tp = np.zeros((0, 1), dtype=bool)
            fp = np.zeros((0, 1), dtype=bool)
            sc = np.zeros((0, 1), dtype=bool)
            num_inst = ngt
            stats.append([tp, fp, sc, num_inst, None, None, None])
        # tqdm.write(str(0.0))
        return stats
    # Run the benchmarking code here.
    threshs = [EP_mask_delta_thresh, EP_normal_delta_thresh, EP_offset_delta_thresh]
    fn = [np.greater_equal, np.less_equal, np.less_equal]
    overlaps = [
        eval_dict["mask_iou"],
        eval_dict["err_normals"],
        eval_dict["err_offsets"],
    ]

    _dt = {"sc": eval_dict["scores"]}
    _gt = {"diff": np.zeros((ngt, 1), dtype=np.bool)}
    _bopts = {"minoverlap": 0.5}
    stats = []
    for i in range(len(EP_ap_str)):
        # Compute a single overlap that ands all the thresholds.
        ov = []
        for j in range(len(overlaps)):
            ov.append(fn[j](overlaps[j], threshs[j][i]))
        _ov = np.all(np.array(ov), 0).astype(np.float32)
        # Benchmark for this setting.
        tp, fp, sc, num_inst, dup_det, inst_id, ov = inst_bench_image(
            _dt, _gt, _bopts, _ov
        )
        stats.append([tp, fp, sc, num_inst, dup_det, inst_id, ov])
    return stats


def inst_bench_image(dt, gt, bOpts, overlap=None):
    nDt = len(dt["sc"])
    nGt = len(gt["diff"])
    numInst = np.sum(gt["diff"] == False)

    # if overlap is None:
    #  overlap = bbox_utils.bbox_overlaps(dt['boxInfo'].astype(np.float), gt['boxInfo'].astype(np.float))
    # assert(issorted(-dt.sc), 'Scores are not sorted.\n');
    sc = dt["sc"]

    det = np.zeros((nGt, 1)).astype(np.bool)
    tp = np.zeros((nDt, 1)).astype(np.bool)
    fp = np.zeros((nDt, 1)).astype(np.bool)
    dupDet = np.zeros((nDt, 1)).astype(np.bool)
    instId = np.zeros((nDt, 1)).astype(np.int32)
    ov = np.zeros((nDt, 1)).astype(np.float32)

    # Walk through the detections in decreasing score
    # and assign tp, fp, fn, tn labels
    for i in range(nDt):
        # assign detection to ground truth object if any
        if nGt > 0:
            maxOverlap = overlap[i, :].max()
            maxInd = overlap[i, :].argmax()
            instId[i] = maxInd
            ov[i] = maxOverlap
        else:
            maxOverlap = 0
            instId[i] = -1
            maxInd = -1
        # assign detection as true positive/don't care/false positive
        if maxOverlap >= bOpts["minoverlap"]:
            if gt["diff"][maxInd] == False:
                if det[maxInd] == False:
                    # true positive
                    tp[i] = True
                    det[maxInd] = True
                else:
                    # false positive (multiple detection)
                    fp[i] = True
                    dupDet[i] = True
        else:
            # false positive
            fp[i] = True
    return tp, fp, sc, numInst, dupDet, instId, ov


def inst_bench(dt, gt, bOpts, tp=None, fp=None, score=None, numInst=None):
    """
    ap, rec, prec, npos, details = inst_bench(dt, gt, bOpts, tp = None, fp = None, sc = None, numInst = None)
    dt  - a list with a dict for each image and with following fields
        .boxInfo - info that will be used to cpmpute the overlap with ground truths, a list
        .sc - score
    gt
        .boxInfo - info used to compute the overlap,  a list
        .diff - a logical array of size nGtx1, saying if the instance is hard or not
    bOpt
        .minoverlap - the minimum overlap to call it a true positive
    [tp], [fp], [sc], [numInst]
        Optional arguments, in case the inst_bench_image is being called outside of this function
    """
    details = None
    if tp is None:
        # We do not have the tp, fp, sc, and numInst, so compute them from the structures gt, and out
        tp = []
        fp = []
        numInst = []
        score = []
        dupDet = []
        instId = []
        ov = []
        for i in range(len(gt)):
            # Sort dt by the score
            sc = dt[i]["sc"]
            bb = dt[i]["boxInfo"]
            ind = np.argsort(sc, axis=0)
            ind = ind[::-1]
            if len(ind) > 0:
                sc = np.vstack((sc[i, :] for i in ind))
                bb = np.vstack((bb[i, :] for i in ind))
            else:
                sc = np.zeros((0, 1)).astype(np.float)
                bb = np.zeros((0, 4)).astype(np.float)

            dtI = dict({"boxInfo": bb, "sc": sc})
            tp_i, fp_i, sc_i, numInst_i, dupDet_i, instId_i, ov_i = inst_bench_image(
                dtI, gt[i], bOpts
            )
            tp.append(tp_i)
            fp.append(fp_i)
            score.append(sc_i)
            numInst.append(numInst_i)
            dupDet.append(dupDet_i)
            instId.append(instId_i)
            ov.append(ov_i)

        details = {
            "tp": list(tp),
            "fp": list(fp),
            "score": list(score),
            "dupDet": list(dupDet),
            "numInst": list(numInst),
            "instId": list(instId),
            "ov": list(ov),
        }

    tp = np.vstack(tp[:])
    fp = np.vstack(fp[:])
    sc = np.vstack(score[:])

    cat_all = np.hstack((tp, fp, sc))
    ind = np.argsort(cat_all[:, 2])
    cat_all = cat_all[ind[::-1], :]
    tp = np.cumsum(cat_all[:, 0], axis=0)
    fp = np.cumsum(cat_all[:, 1], axis=0)
    thresh = cat_all[:, 2]
    npos = np.sum(numInst, axis=0)

    # Compute precision/recall
    rec = tp / npos
    prec = np.divide(tp, (fp + tp))
    ap = VOCap(rec, prec)
    return ap, rec, prec, npos, details


def VOCap(rec, prec):
    rec = rec.reshape(rec.size, 1)
    prec = prec.reshape(prec.size, 1)
    z = np.zeros((1, 1))
    o = np.ones((1, 1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    I = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in I:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def main(args):
    ev = Evaluator(args, dataset=args.dataset_phase)
    optimized_dict = None
    if len(args.optimized_dict_path) != 0 and os.path.exists(args.optimized_dict_path):
        print(f"reading from {args.optimized_dict_path}")
        with open(args.optimized_dict_path, "rb") as f:
            optimized_dict = pickle.load(f)
    else:
        print("No optimized dict found, generating...")
        optimized_dict = multiprocess_by_list(
            ev, args.num_process, np.arange(len(ev.rcnn_data)), args.evaluate, True
        )
    # save_dict(optimized_dict, './results/gtbox', 'discrete')

    if args.evaluate == "AP":
        ev.optimized_dict = optimized_dict
        error_dict = multiprocess_by_list(
            ev, args.num_process, np.arange(len(ev.rcnn_data)), args.evaluate, False
        )

        bench_stats = []
        for idx in tqdm(range(len(ev.rcnn_data))):
            errs = error_dict[idx]
            bench_image_stats = evaluate_by_idx(errs)
            bench_stats.append(bench_image_stats)

        # Accumulate stats
        bb = list(zip(*bench_stats))
        bench_summarys = []
        for i in range(len(EP_ap_str) - 1):
            tp, fp, sc, num_inst, dup_det, inst_id, ov = zip(*bb[i])
            ap, rec, prec, npos, details = inst_bench(
                None, None, None, tp, fp, sc, num_inst
            )
            bench_summary = {
                "prec": prec.tolist(),
                "rec": rec.tolist(),
                "ap": ap[0],
                "npos": npos,
            }
            print("{:>20s}: {:5.3f}".format(EP_ap_str[i], ap[0] * 100.0))
            bench_summarys.append(bench_summary)

    elif args.evaluate == "camera":
        ev.evaluate_camera(optimized_dict)
    elif args.evaluate == "correspondence":
        assert ev.gt_box, "Correspondence evaluation requires gt box!"
        ev.evaluate_matching(optimized_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument(
        "--rcnn-cached-file", required=True, help="path to instances_predictions.pth"
    )
    parser.add_argument(
        "--camera-cached-file", required=True, help="path to summary.pkl"
    )
    parser.add_argument(
        "--evaluate", default="correspondence", help="AP / camera / correspondence"
    )
    parser.add_argument(
        "--num-process",
        default=50,
        type=int,
        help="number of process for multiprocessing",
    )
    parser.add_argument(
        "--num-data",
        default=-1,
        type=int,
        help="number of data to process, if -1 then all.",
    )
    parser.add_argument(
        "--dataset-phase", default="mp3d_test", type=str, help="dataset and phase"
    )
    parser.add_argument(
        "--optimized-dict-path", default="", type=str, help="path to optimized_dict.pkl"
    )
    parser.add_argument("--opts", default=[])
    args = parser.parse_args()
    print(args)
    main(args)
