import copy
import itertools
import json
import logging
import numpy as np
import pickle
import os
from collections import OrderedDict, Counter
from scipy.special import softmax
import detectron2.utils.comm as comm
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import setup_logger, create_small_table
from pycocotools.coco import COCO
from iopath.common.file_io import PathManager, file_lock
from sklearn.metrics import average_precision_score, roc_auc_score

from .detectron2coco import convert_to_coco_dict
import sparseplane.utils.VOCap as VOCap
from sparseplane.utils.metrics import compare_planes

try:
    import KMSolver
except:
    print("[warning] cannot import KMSolver")
logger = logging.getLogger(__name__)
if not logger.isEnabledFor(logging.INFO):
    setup_logger(name=__name__)


class MP3DEvaluator(COCOEvaluator):
    """
    Evaluate object proposal, instance detection, segmentation and affinity
    outputs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self.cfg = cfg
        self._tasks = self._tasks_from_config(cfg)
        self._plane_tasks = self._specific_tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._device = cfg.MODEL.DEVICE
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._coco_api = COCO(self._siamese_to_coco(self._metadata.json_file))
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self._kpt_oks_sigmas = None

        self._filter_iou = 0.7
        self._filter_score = 0.7

        self._visualize = cfg.TEST.SAVE_VIS
        self._K_inv_dot_xy_1 = torch.FloatTensor(self.get_K_inv_dot_xy_1()).to(
            self._device
        )
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX
        if self.cfg.MODEL.CAMERA_ON:
            kmeans_trans_path = cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH
            kmeans_rots_path = cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH
            assert os.path.exists(kmeans_trans_path)
            assert os.path.exists(kmeans_rots_path)
            with open(kmeans_trans_path, "rb") as f:
                self.kmeans_trans = pickle.load(f)
            with open(kmeans_rots_path, "rb") as f:
                self.kmeans_rots = pickle.load(f)

    def get_K_inv_dot_xy_1(self, h=480, w=640):
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

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        return tasks

    def _specific_tasks_from_config(self, cfg):
        tasks = ()
        if cfg.MODEL.EMBEDDING_ON and cfg.MODEL.ROI_EMBEDDING_HEAD.LOSS_WEIGHT != 0:
            tasks = tasks + ("embedding",)
        if cfg.MODEL.CAMERA_ON:
            tasks = tasks + ("camera",)
        return tasks

    def _siamese_to_coco(self, siamese_json):
        assert self._output_dir
        save_json = os.path.join(self._output_dir, "siamese2coco.json")
        pm = PathManager()
        pm.mkdirs(os.path.dirname(save_json))
        with file_lock(save_json):
            if pm.exists(save_json):
                logger.warning(
                    f"Using previously cached COCO format annotations at '{save_json}'. "
                    "You need to clear the cache file if your dataset has been modified."
                )
            else:
                logger.info(
                    f"Converting annotations of dataset '{siamese_json}' to COCO format ...)"
                )
                with pm.open(siamese_json, "r") as f:
                    siamese_data = json.load(f)
                coco_data = {"data": []}
                exist_imgid = set()
                for key, datas in siamese_data.items():
                    # copy 'info', 'categories'
                    if key != "data":
                        coco_data[key] = datas
                    else:
                        for data in datas:
                            for i in range(2):
                                img_data = data[str(i)]
                                if img_data["image_id"] in exist_imgid:
                                    continue
                                else:
                                    exist_imgid.add(img_data["image_id"])
                                    coco_data[key].append(img_data)
                self._logger.info(f"Number of unique images: {len(exist_imgid)}.")
                coco_data = convert_to_coco_dict(coco_data["data"], self._metadata)
                with pm.open(save_json, "w") as f:
                    json.dump(coco_data, f)
        return save_json

    def _siamese_to_single(self, siamese_predictions):
        single_predictions = []
        exist_imgid = set()
        for pred in siamese_predictions:
            for i in range(2):
                single_pred = pred[str(i)]["instances"]
                if len(single_pred) == 0:
                    continue
                imgid = single_pred[0]["image_id"]
                if imgid in exist_imgid:
                    continue
                exist_imgid.add(imgid)
                single_predictions.append(pred[str(i)])
        return single_predictions

    def depth2XYZ(self, depth):
        """
        Convert depth to point clouds
        X - width
        Y - depth
        Z - height
        """
        XYZ = self._K_inv_dot_xy_1 * depth
        return XYZ

    def override_offset(self, xyz, instance_coco, instance_d2):
        plane_normals = F.normalize(instance_d2["instances"].pred_plane, p=2)
        masks = instance_d2["instances"].pred_masks
        offsets = ((plane_normals.view(-1, 3, 1, 1) * xyz).sum(1) * masks).sum(-1).sum(-1) / torch.clamp(masks.sum(-1).sum(-1), min=1e-4)
        plane_parameters = plane_normals * offsets.view((-1, 1))
        valid = (masks.sum(-1).sum(-1) > 0).cpu()
        instance_coco["pred_plane"][valid] = plane_parameters.cpu()[valid]
        return instance_coco

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"0": {}, "1": {}}
            tmp_instances = {"0": {}, "1": {}}
            for i in range(2):
                # TODO this is ugly
                prediction[str(i)]["image_id"] = input[str(i)]["image_id"]
                prediction[str(i)]["file_name"] = input[str(i)]["file_name"]
                if "instances" in output[str(i)]:
                    instances = output[str(i)]["instances"].to(self._cpu_device)
                    prediction[str(i)]["instances"] = instances_to_coco_json(
                        instances, input[str(i)]["image_id"]
                    )
                    tmp_instances[str(i)]["embeddingbox"] = {
                        "pred_boxes": instances.pred_boxes,
                        "scores": instances.scores,
                    }
                if "proposals" in output[str(i)]:
                    prediction[str(i)]["proposals"] = output[str(i)]["proposals"].to(
                        self._cpu_device
                    )
                if "annotations" in input[str(i)]:
                    tmp_instances[str(i)]["gt_bbox"] = [
                        ann["bbox"] for ann in input[str(i)]["annotations"]
                    ]
                    if len(input[str(i)]["annotations"]) > 0:
                        tmp_instances[str(i)]["gt_bbox"] = np.array(
                            tmp_instances[str(i)]["gt_bbox"]
                        ).reshape(-1, 4)  # xywh from coco
                        original_mode = input[str(i)]["annotations"][0]["bbox_mode"]
                        tmp_instances[str(i)]["gt_bbox"] = BoxMode.convert(
                            tmp_instances[str(i)]["gt_bbox"],
                            BoxMode(original_mode),
                            BoxMode.XYXY_ABS,
                        )
                        if hasattr(output[str(i)]["instances"], "pred_plane"):
                            prediction[str(i)]["pred_plane"] = output[str(i)][
                                "instances"
                            ].pred_plane.to(self._cpu_device)
                if output["depth"][str(i)] is not None:
                    prediction[str(i)]["pred_depth"] = output["depth"][str(i)].to(
                        self._cpu_device
                    )
                    xyz = self.depth2XYZ(output["depth"][str(i)])
                    prediction[str(i)] = self.override_offset(
                        xyz, prediction[str(i)], output[str(i)]
                    )
                    depth_rst = get_depth_err(
                        output["depth"][str(i)], input[str(i)]["depth"].to(self._device)
                    )
                    prediction[str(i)]["depth_l1_dist"] = depth_rst.to(self._cpu_device)

            if "pred_aff" in output:
                tmp_instances["pred_aff"] = output["pred_aff"].to(self._cpu_device)
            if "geo_aff" in output:
                tmp_instances["geo_aff"] = output["geo_aff"].to(self._cpu_device)
            if "emb_aff" in output:
                tmp_instances["emb_aff"] = output["emb_aff"].to(self._cpu_device)
            if "gt_corrs" in input:
                tmp_instances["gt_corrs"] = input["gt_corrs"]
            prediction["corrs"] = tmp_instances
            if "embedding" in self._plane_tasks:
                if self._eval_gt_box:
                    aff_rst = get_affinity_label_score(
                        tmp_instances,
                        filter_iou=self._filter_iou,
                        filter_score=self._filter_score,
                        device=self._device,
                    )
                else:
                    aff_rst = get_affinity_label_score(
                        tmp_instances,
                        hungarian_threshold=[],
                        filter_iou=self._filter_iou,
                        filter_score=self._filter_score,
                        device=self._device,
                    )
                prediction.update(aff_rst)
            if "camera" in self._plane_tasks:
                camera_dict = {
                    "logits": {
                        "tran": output["camera"]["tran"].to(self._cpu_device),
                        "rot": output["camera"]["rot"].to(self._cpu_device),
                    },
                    "gts": {
                        "tran": input["rel_pose"]["position"],
                        "rot": input["rel_pose"]["rotation"],
                        "tran_cls": input["rel_pose"]["tran_cls"],
                        "rot_cls": input["rel_pose"]["rot_cls"],
                    },
                }
                prediction["camera"] = camera_dict
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            gt_corrs = self._gt_corrs

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            pm = PathManager()
            pm.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with pm.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        # if not self._visualize:
        single_predictions = self._siamese_to_single(predictions)
        if "proposals" in single_predictions[0]:
            self._eval_box_proposals(single_predictions)
        if "instances" in single_predictions[0]:
            # self._eval_predictions(set(self._tasks), single_predictions)
            self._eval_plane(single_predictions)
        if "depth_l1_dist" in single_predictions[0]:
            self._eval_depth(single_predictions)
        if "embedding" in self._plane_tasks:
            self._eval_affinity(predictions)
        if "camera" in self._plane_tasks:
            summary = self._eval_camera(predictions)
            file_path = os.path.join(self._output_dir, "summary.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(summary, f)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

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

    def _eval_camera(self, predictions):
        acc_threshold = {
            "tran": 1.0,
            "rot": 30,
        }  # threshold for translation and rotation error to say prediction is correct.
        tran_logits = torch.stack(
            [p["camera"]["logits"]["tran"] for p in predictions]
        ).numpy()
        rot_logits = torch.stack(
            [p["camera"]["logits"]["rot"] for p in predictions]
        ).numpy()
        gt_tran_cls = torch.stack(
            [p["camera"]["gts"]["tran_cls"] for p in predictions]
        ).numpy()
        gt_rot_cls = torch.stack(
            [p["camera"]["gts"]["rot_cls"] for p in predictions]
        ).numpy()
        gt_tran = np.vstack([p["camera"]["gts"]["tran"] for p in predictions])
        gt_rot = np.vstack([p["camera"]["gts"]["rot"] for p in predictions])
        topk_acc = get_camera_top_k_acc(
            logits={"tran": tran_logits, "rot": rot_logits},
            gts={"tran_cls": gt_tran_cls, "rot_cls": gt_rot_cls},
            n_clusters={
                "tran": self.kmeans_trans.n_clusters,
                "rot": self.kmeans_rots.n_clusters,
            },
        )
        topk_acc["tran"] = np.cumsum(topk_acc["tran"]) / np.sum(topk_acc["tran"])
        topk_acc["rot"] = np.cumsum(topk_acc["rot"]) / np.sum(topk_acc["rot"])
        pred_tran = self.class2xyz(np.argmax(tran_logits, axis=1))
        pred_rot = self.class2quat(np.argmax(rot_logits, axis=1))

        top1_error = {
            "tran": np.linalg.norm(gt_tran - pred_tran, axis=1),
            "rot": angle_error_vec(pred_rot, gt_rot),
        }
        top1_accuracy = {
            "tran": (top1_error["tran"] < acc_threshold["tran"]).sum()
            / len(top1_error["tran"]),
            "rot": (top1_error["rot"] < acc_threshold["rot"]).sum()
            / len(top1_error["rot"]),
        }
        camera_metrics = {
            f"top1 T err < {acc_threshold['tran']}": top1_accuracy["tran"] * 100,
            f"top1 R err < {acc_threshold['rot']}": top1_accuracy["rot"] * 100,
            f"T mean err": np.mean(top1_error["tran"]),
            f"R mean err": np.mean(top1_error["rot"]),
            f"T median err": np.median(top1_error["tran"]),
            f"R median err": np.median(top1_error["rot"]),
        }
        logger.info("Camera metrics: \n" + create_small_table(camera_metrics))
        topk_metrics = {
            f"top1 T acc": topk_acc["tran"][0] * 100,
            f"top5 T acc": topk_acc["tran"][4] * 100,
            f"top10 T acc": topk_acc["tran"][9] * 100,
            f"top32 T acc": topk_acc["tran"][31] * 100,
            f"top1 R acc": topk_acc["rot"][0] * 100,
            f"top5 R acc": topk_acc["rot"][4] * 100,
            f"top10 R acc": topk_acc["rot"][9] * 100,
            f"top32 R acc": topk_acc["rot"][31] * 100,
        }
        logger.info("Camera topk: \n" + create_small_table(topk_metrics))
        camera_metrics.update(topk_metrics)
        self._results.update(camera_metrics)
        summary = {
            "errors": np.array([top1_error["tran"], top1_error["rot"]]),
            "preds": {
                "tran": pred_tran,
                "rot": pred_rot,
                "tran_cls": np.argmax(tran_logits, axis=1).reshape(-1, 1),
                "rot_cls": np.argmax(rot_logits, axis=1).reshape(-1, 1),
            },
            "gts": {
                "tran": gt_tran,
                "rot": gt_rot,
                "tran_cls": gt_tran_cls,
                "rot_cls": gt_rot_cls,
            },
            "logits_sms": {
                "tran": softmax(tran_logits, axis=1),
                "rot": softmax(rot_logits, axis=1),
            },
            "accuracy": [top1_accuracy["tran"], top1_accuracy["rot"]],
            "keys": [p["0"]["file_name"] + p["1"]["file_name"] for p in predictions],
        }
        return summary

    def _eval_affinity(self, predictions):
        """
        Evaluate plane correspondence.
        """
        logger.info("Evaluating embedding affinity ...")
        labels = []
        preds = []
        for pred in predictions:
            labels.extend(pred["labels"])
            preds.extend(pred["preds"])

        best_auc_ipaa = 0
        best_threshold = 0
        best_ipaa_dict = None
        for th in predictions[0]["ipaa_by_threshold"].keys():
            IPAA_dict = {}
            for i in range(11):
                IPAA_dict[i * 10] = 0
            for pred in predictions:
                for key in IPAA_dict.keys():
                    if pred["ipaa_by_threshold"][th] >= key / 100:
                        IPAA_dict[key] += 1
            auc_ipaa = compute_auc(IPAA_dict)
            if auc_ipaa > best_auc_ipaa:
                best_auc_ipaa = auc_ipaa
                best_threshold = th
                best_ipaa_dict = IPAA_dict

        if not len(labels):
            return
        auc = roc_auc_score(labels, preds) * 100
        ap = average_precision_score(labels, preds) * 100
        if best_ipaa_dict is None:
            results = {
                f"ap@iou={self._filter_iou}": ap,
                f"auc@iou={self._filter_iou}": auc,
            }
        else:
            results = {
                f"ap@iou={self._filter_iou}": ap,
                f"auc@iou={self._filter_iou}": auc,
                f"ipaa-80": best_ipaa_dict[80] / len(predictions),
                f"ipaa-90": best_ipaa_dict[90] / len(predictions),
                f"ipaa-100": best_ipaa_dict[100] / len(predictions),
                f"auc-ipaa": best_auc_ipaa,
                f"hungarian-threshold": best_threshold,
            }
        logger.info("Affinity metrics: \n" + create_small_table(results))
        self._results.update(results)

    def _eval_plane(self, predictions):
        results = evaluate_for_planes(
            predictions,
            self._coco_api,
            self._metadata,
            self._filter_iou,
            device=self._device,
        )
        self._results.update(results)

    def _eval_depth(self, predictions):
        depth_l1_dist = [p["depth_l1_dist"] for p in predictions]
        result = {f"depth_l1_dist": np.mean(depth_l1_dist)}
        logger.info("Depth metrics: \n" + create_small_table(result))
        self._results.update(result)


def l1LossMask(pred, gt, mask):
    """L1 loss with a mask"""
    return torch.sum(torch.abs(pred - gt) * mask) / torch.clamp(mask.sum(), min=1)


def get_depth_err(pred_depth, gt_depth, device=None):
    l1dist = l1LossMask(pred_depth, gt_depth, (gt_depth > 1e-4).float())
    return l1dist


def angle_error_vec(v1, v2):
    return 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(v1, v2), axis=1)), -1.0, 1.0)) * 180 / np.pi


def get_camera_top_k_acc(logits, gts, n_clusters):
    topk_acc = {}
    for phase in ["tran", "rot"]:
        idx = np.flip(np.argsort(logits[phase], axis=1), axis=1)
        k = np.where(idx == gts[phase + "_cls"])[1]
        count_dict = Counter(k)
        keys = np.array(list(count_dict.keys()))
        values = np.array(list(count_dict.values()))
        sorted_key = np.argsort(keys)
        xs = np.arange(n_clusters[phase])
        topk_acc[phase] = np.zeros(n_clusters[phase])
        xs = keys[sorted_key]
        ys = values[sorted_key]
        for idx, y in zip(xs, ys):
            topk_acc[phase][idx] = y
    return topk_acc


def evaluate_for_planes(
    predictions,
    dataset,
    metadata,
    filter_iou,
    iou_thresh=0.5,
    normal_threshold=30,
    offset_threshold=0.3,
    device=None,
):
    if device is None:
        device = torch.device("cpu")
    # classes
    cat_ids = sorted(dataset.getCatIds())
    reverse_id_mapping = {
        v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
    }

    # initialize tensors to record box & mask AP, number of gt positives
    box_apscores, box_aplabels = {}, {}
    mask_apscores, mask_aplabels = {}, {}
    plane_apscores, plane_aplabels = {}, {}
    plane_offset_errs, plane_normal_errs = [], []
    npos = {}
    for cat_id in cat_ids:
        box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        mask_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mask_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        plane_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        plane_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        npos[cat_id] = 0.0

    # number of gt positive instances per class
    for gt_ann in dataset.dataset["annotations"]:
        gt_label = gt_ann["category_id"]
        npos[gt_label] += 1.0

    for prediction in predictions:
        original_id = prediction["image_id"]
        image_width = dataset.loadImgs([original_id])[0]["width"]
        image_height = dataset.loadImgs([original_id])[0]["height"]
        if "instances" not in prediction:
            continue

        num_img_preds = len(prediction["instances"])
        if num_img_preds == 0:
            continue

        # predictions
        scores, boxes, labels, masks_rles = [], [], [], []
        for ins in prediction["instances"]:
            scores.append(ins["score"])
            boxes.append(ins["bbox"])
            labels.append(ins["category_id"])
            masks_rles.append(ins["segmentation"])
        boxes = np.array(boxes)  # xywh from coco
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        boxes = Boxes(torch.tensor(np.array(boxes))).to(device)
        planes = prediction["pred_plane"]

        # ground truth
        # anotations corresponding to original_id (aka coco image_id)
        gt_ann_ids = dataset.getAnnIds(imgIds=[original_id])
        gt_anns = dataset.loadAnns(gt_ann_ids)
        # get original ground truth mask, box, label & mesh
        gt_boxes, gt_labels, gt_mask_rles, gt_planes = [], [], [], []
        for ann in gt_anns:
            gt_boxes.append(ann["bbox"])
            gt_labels.append(ann["category_id"])
            if isinstance(ann["segmentation"], list):
                polygons = [np.array(p, dtype=np.float64) for p in ann["segmentation"]]
                rles = mask_util.frPyObjects(polygons, image_height, image_width)
                rle = mask_util.merge(rles)
            elif isinstance(ann["segmentation"], dict):  # RLE
                rle = ann["segmentation"]
            else:
                raise TypeError(
                    f"Unknown segmentation type {type(ann['segmentation'])}!"
                )
            gt_mask_rles.append(rle)
            gt_planes.append(ann["plane"])

        gt_boxes = np.array(gt_boxes)  # xywh from coco
        gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        faux_gt_targets = Boxes(
            torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        )

        # box iou
        boxiou = pairwise_iou(boxes, faux_gt_targets)

        # filter predictions with iou > filter_iou
        # valid_pred_ids = (boxiou > filter_iou).sum(axis=1) > 0

        # mask iou
        miou = mask_util.iou(masks_rles, gt_mask_rles, [0] * len(gt_mask_rles))

        plane_metrics = compare_planes(planes, gt_planes)

        # sort predictions in descending order
        scores = torch.tensor(np.array(scores), dtype=torch.float32)
        scores_sorted, idx_sorted = torch.sort(scores, descending=True)
        # record assigned gt.
        box_covered = []
        mask_covered = []
        plane_covered = []

        for pred_id in range(num_img_preds):
            # remember we only evaluate the preds that have overlap more than
            # iou_filter with the ground truth prediction
            # if valid_pred_ids[idx_sorted[pred_id]] == 0:
            #     continue
            # Assign pred to gt
            gt_id = torch.argmax(boxiou[idx_sorted[pred_id]])
            gt_label = gt_labels[gt_id]
            # map to dataset category id
            pred_label = reverse_id_mapping[labels[idx_sorted[pred_id]]]
            pred_miou = miou[idx_sorted[pred_id], gt_id]
            pred_biou = boxiou[idx_sorted[pred_id], gt_id]
            pred_score = scores[idx_sorted[pred_id]].view(1).to(device)

            normal = plane_metrics["norm"][idx_sorted[pred_id], gt_id].item()
            offset = plane_metrics["offset"][idx_sorted[pred_id], gt_id].item()
            plane_offset_errs.append(offset)
            plane_normal_errs.append(normal)

            # mask
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_miou > iou_thresh)
                and (gt_id not in mask_covered)
            ):
                tpfp[0] = 1
                mask_covered.append(gt_id)
            mask_apscores[pred_label].append(pred_score)
            mask_aplabels[pred_label].append(tpfp)

            # box
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_biou > iou_thresh)
                and (gt_id not in box_covered)
            ):
                tpfp[0] = 1
                box_covered.append(gt_id)
            box_apscores[pred_label].append(pred_score)
            box_aplabels[pred_label].append(tpfp)

            # plane
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (normal < normal_threshold)
                and (offset < offset_threshold)
                and (gt_id not in plane_covered)
            ):
                tpfp[0] = 1
                plane_covered.append(gt_id)
            plane_apscores[pred_label].append(pred_score)
            plane_aplabels[pred_label].append(tpfp)

    # check things for eval
    # assert npos.sum() == len(dataset.dataset["annotations"])
    # convert to tensors
    detection_metrics = {}
    boxap, maskap, planeap = 0.0, 0.0, 0.0
    valid = 0.0
    for cat_id in cat_ids:
        cat_name = dataset.loadCats([cat_id])[0]["name"]
        if npos[cat_id] == 0:
            continue
        valid += 1

        cat_box_ap = VOCap.compute_ap(
            torch.cat(box_apscores[cat_id]),
            torch.cat(box_aplabels[cat_id]),
            npos[cat_id],
        ).item()
        boxap += cat_box_ap
        detection_metrics["box_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_box_ap

        cat_mask_ap = VOCap.compute_ap(
            torch.cat(mask_apscores[cat_id]),
            torch.cat(mask_aplabels[cat_id]),
            npos[cat_id],
        ).item()
        maskap += cat_mask_ap
        detection_metrics["mask_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_mask_ap

        cat_plane_ap = VOCap.compute_ap(
            torch.cat(plane_apscores[cat_id]),
            torch.cat(plane_aplabels[cat_id]),
            npos[cat_id],
        ).item()
        planeap += cat_plane_ap
        detection_metrics[
            "plane_ap@iou%.1fnormal%.1foffset%.1f - %s"
            % (iou_thresh, normal_threshold, offset_threshold, cat_name)
        ] = cat_plane_ap
    detection_metrics["box_ap@%.1f" % iou_thresh] = boxap / valid
    detection_metrics["mask_ap@%.1f" % iou_thresh] = maskap / valid
    detection_metrics[
        "plane_ap@iou%.1fnormal%.1foffset%.1f"
        % (iou_thresh, normal_threshold, offset_threshold)
    ] = (planeap / valid)
    logger.info("Detection metrics: \n" + create_small_table(detection_metrics))
    plane_metrics = {}
    plane_normal_errs = np.array(plane_normal_errs)
    plane_offset_errs = np.array(plane_offset_errs)
    plane_metrics["%normal<10"] = (
        sum(plane_normal_errs < 10) / len(plane_normal_errs) * 100
    )
    plane_metrics["%normal<30"] = (
        sum(plane_normal_errs < 30) / len(plane_normal_errs) * 100
    )
    plane_metrics["%offset<0.5"] = (
        sum(plane_offset_errs < 0.5) / len(plane_offset_errs) * 100
    )
    plane_metrics["%offset<0.3"] = (
        sum(plane_offset_errs < 0.3) / len(plane_offset_errs) * 100
    )
    plane_metrics["mean_normal"] = plane_normal_errs.mean()
    plane_metrics["median_normal"] = np.median(plane_normal_errs)
    plane_metrics["mean_offset"] = plane_offset_errs.mean()
    plane_metrics["median_offset"] = np.median(plane_offset_errs)
    logger.info("Plane metrics: \n" + create_small_table(plane_metrics))
    plane_metrics.update(detection_metrics)
    return plane_metrics


def get_affinity_label_score(
    prediction_pair,
    filter_iou,
    filter_score,
    hungarian_threshold=[0.5, 0.6, 0.7, 0.8, 0.9],
    device=None,
):
    if device is None:
        device = torch.device("cpu")
    cpu_device = torch.device("cpu")

    labels = []
    preds = []
    pred_aff = prediction_pair["pred_aff"].to(cpu_device)
    selected_pred_box_id = {}
    for i in range(2):
        prediction = prediction_pair[str(i)]
        if "embeddingbox" not in prediction:
            continue
        num_img_preds = len(prediction["embeddingbox"]["pred_boxes"])
        if num_img_preds == 0:
            continue
        # ground truth box
        gt_box = prediction["gt_bbox"]
        faux_gt_targets = Boxes(
            torch.tensor(gt_box, dtype=torch.float32, device=device)
        )

        # predictions
        scores = prediction["embeddingbox"]["scores"]
        pred_boxes = prediction["embeddingbox"]["pred_boxes"]

        # filter by score first,
        chosen = (scores > filter_score).nonzero().flatten()
        if len(chosen) == 0:
            selected_pred_box_id[str(i)] = torch.ones(len(gt_box)) * -1
            continue
        scores = scores[chosen]
        pred_boxes = pred_boxes[chosen].to(device)
        boxiou = pairwise_iou(pred_boxes, faux_gt_targets)
        iou_sorted, idx_sorted = torch.sort(boxiou, dim=0, descending=True)
        selected_pred_box_id[str(i)] = chosen[idx_sorted[0, :]]
        selected_pred_box_id[str(i)][iou_sorted[0, :] < filter_iou] = -1
    for gt_idx1, predbox_idx1 in enumerate(selected_pred_box_id["0"]):
        if predbox_idx1 != -1:
            for gt_idx2, predbox_idx2 in enumerate(selected_pred_box_id["1"]):
                if predbox_idx2 != -1:
                    if [gt_idx1, gt_idx2] in prediction_pair["gt_corrs"]:
                        labels.append(1)
                    else:
                        labels.append(0)
                    preds.append(pred_aff[predbox_idx1, predbox_idx2])
    ipaa_by_threshold = {}
    for th in hungarian_threshold:
        assignment = get_assignment(1 - pred_aff, threshold=1 - th)
        ipaa = compute_IPAA(assignment, prediction_pair["gt_corrs"])
        ipaa_by_threshold[th] = ipaa
    return {"labels": labels, "preds": preds, "ipaa_by_threshold": ipaa_by_threshold}


def compute_IPAA(pred_assignment_m, gt_assignment_list):
    wrong_count = 0
    gt_assignment_list = np.array(gt_assignment_list)
    common_row_idxs = gt_assignment_list[:, 0]
    common_colomn_idxs = gt_assignment_list[:, 1]
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
    return 1 - p


def get_assignment(distance_matrix, threshold=0.3):
    """
    km: Hungarian Algo
    if the distance > threshold, even it is smallest, it is also false.
    """
    cost_matrix = (distance_matrix.numpy() * 1000).astype(np.int)
    prediction_matrix_km = KMSolver.solve(cost_matrix, threshold=int(threshold * 1000))
    return prediction_matrix_km


def compute_auc(IPAA_dict):
    try:
        return (np.array(list(IPAA_dict.values())) / IPAA_dict["0"]).mean()
    except:
        return (np.array(list(IPAA_dict.values())) / IPAA_dict[0]).mean()
