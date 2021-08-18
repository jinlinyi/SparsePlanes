import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling import (
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
)
from detectron2.modeling import META_ARCH_REGISTRY
from sparseplane.modeling.postprocessing import detector_postprocess
from sparseplane.modeling.depth_net import build_depth_head
from sparseplane.modeling.camera_net import build_camera_head
from sparseplane.modeling.roi_heads.embedding_loss import (
    OnlineTripletLoss,
    CooperativeTripletLoss,
)


__all__ = ["SiamesePlaneRCNN"]


@META_ARCH_REGISTRY.register()
class SiamesePlaneRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.mask_threshold = cfg.MODEL.ROI_MASK_HEAD.MASK_THRESHOLD
        self.nms = cfg.MODEL.ROI_MASK_HEAD.NMS
        self.depth_head_on = cfg.MODEL.DEPTH_ON
        if self.depth_head_on:
            self.depth_head = build_depth_head(cfg)
        self.camera_on = cfg.MODEL.CAMERA_ON
        if self.camera_on:
            self.camera_head = build_camera_head(cfg)
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.embedding_on = cfg.MODEL.EMBEDDING_ON
        if self.embedding_on:
            self._asnet_on = (
                cfg.MODEL.ROI_EMBEDDING_HEAD.NAME == "EmbeddingRCNNASNetHead"
            )
            self.embedding_loss_weight = cfg.MODEL.ROI_EMBEDDING_HEAD.LOSS_WEIGHT
            if cfg.MODEL.ROI_EMBEDDING_HEAD.LOSS_TYPE == "TripletLoss":
                if not self._asnet_on:
                    self.embedding_loss = OnlineTripletLoss(
                        cfg.MODEL.ROI_EMBEDDING_HEAD.MARGIN,
                        cfg.MODEL.DEVICE,
                        selector_type=cfg.MODEL.ROI_EMBEDDING_HEAD.TRIPLET_SELECTOR_TYPE,
                    )
                else:
                    self.embedding_loss = CooperativeTripletLoss(
                        cfg.MODEL.ROI_EMBEDDING_HEAD.MARGIN,
                        cfg.MODEL.DEVICE,
                        selector_type=cfg.MODEL.ROI_EMBEDDING_HEAD.TRIPLET_SELECTOR_TYPE,
                    )
            else:
                raise NotImplementedError
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX
        self.to(self.device)
        self._freeze = cfg.MODEL.FREEZE
        for layers in self._freeze:
            layer = layers.split(".")
            final = self
            for l in layer:
                final = getattr(final, l)
            for params in final.parameters():
                params.requires_grad = False

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        batched_inputs_single = {"0": [], "1": []}
        for batched_input in batched_inputs:
            for i in ["0", "1"]:
                batched_inputs_single[i].append(batched_input[i])
        pred_instances1, losses1, features1 = self.forward_single(
            batched_inputs_single["0"]
        )
        pred_instances2, losses2, features2 = self.forward_single(
            batched_inputs_single["1"]
        )

        losses = {}
        for key in losses1.keys():
            losses[key] = (losses1[key] + losses2[key]) / 2

        if self.embedding_on:
            if "roi_heads.embedding_head" not in self._freeze:
                losses.update(
                    self.embedding_loss(
                        batched_inputs,
                        pred_instances1,
                        pred_instances2,
                        loss_weight=self.embedding_loss_weight,
                    )
                )
        if self.camera_on:
            gt_cls = self.process_camera(batched_inputs)
            losses.update(self.camera_head(features1, features2, gt_cls))
        return losses

    def forward_single(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        pred_instances, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )
        depth_losses = {}
        if self.depth_head_on:
            if "depth_head" not in self._freeze:
                gt_depth = self.process_depth(batched_inputs)
                depth_losses = self.depth_head(features, gt_depth)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(depth_losses)
        if self.camera_on:
            return pred_instances, losses, features
        else:
            return pred_instances, losses, None

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        assert detected_instances is None

        batched_inputs_single = {"0": [], "1": []}
        for batched_input in batched_inputs:
            for i in ["0", "1"]:
                batched_inputs_single[i].append(batched_input[i])
        pred_instances1, pred_depth1, features1 = self.inference_single(
            batched_inputs_single["0"], do_postprocess
        )
        pred_instances2, pred_depth2, features2 = self.inference_single(
            batched_inputs_single["1"], do_postprocess
        )
        if self.camera_on:
            pred_camera = self.camera_head(features1, features2)
        else:
            pred_camera = {
                "tran": [None] * len(pred_instances1),
                "rot": [None] * len(pred_instances1),
            }
        if self.embedding_on:
            emb_distance_matrix, numPlanes1, numPlanes2 = self.embedding_loss.inference(
                pred_instances1, pred_instances2
            )
            emb_affinity_matrix = 1 - emb_distance_matrix / 2  # [0, 1]
            results = []
            for pre1, pre2, m_emb, n1, n2, d1, d2, cam_tran, cam_rot in zip(
                pred_instances1,
                pred_instances2,
                emb_affinity_matrix,
                numPlanes1,
                numPlanes2,
                pred_depth1,
                pred_depth2,
                pred_camera["tran"],
                pred_camera["rot"],
            ):
                results.append(
                    {
                        "0": pre1,
                        "1": pre2,
                        "pred_aff": m_emb[: n1[0], : n2[0]],
                        "depth": {"0": d1, "1": d2},
                        "camera": {"tran": cam_tran, "rot": cam_rot},
                    }
                )
        else:
            results = []
            for pre1, pre2, d1, d2, cam_tran, cam_rot in zip(
                pred_instances1,
                pred_instances2,
                pred_depth1,
                pred_depth2,
                pred_camera["tran"],
                pred_camera["rot"],
            ):
                results.append(
                    {
                        "0": pre1,
                        "1": pre2,
                        "depth": {"0": d1, "1": d2},
                        "camera": {"tran": cam_tran, "rot": cam_rot},
                    }
                )
        return results

    def inference_single(self, batched_inputs, do_postprocess=True):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if self._eval_gt_box:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            elif "targets" in batched_inputs[0]:
                log_first_n(
                    logging.WARN,
                    "'targets' in the model inputs is now renamed to 'instances'!",
                    n=10,
                )
                gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            for inst in gt_instances:
                inst.proposal_boxes = inst.gt_boxes
                inst.objectness_logits = torch.ones(len(inst.gt_boxes))
            proposals = gt_instances
        else:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        pred_depth = [None] * len(proposals)
        if self.depth_head_on:
            if "depth_head" not in self._freeze:
                pred_depth = self.depth_head(features, None)

        results, _ = self.roi_heads(images, features, proposals, None)

        if self.camera_on:
            rtn_feature = features
        else:
            rtn_feature = None
        if do_postprocess:
            return (
                SiamesePlaneRCNN._postprocess(
                    results,
                    batched_inputs,
                    images.image_sizes,
                    mask_threshold=self.mask_threshold,
                    nms=self.nms,
                ),
                pred_depth,
                rtn_feature,
            )
        else:
            return results, pred_depth, rtn_feature

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = {}
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def process_depth(self, batched_inputs):
        depth = [x["depth"].to(self.device) for x in batched_inputs]
        depth = torch.stack(depth)
        return depth

    def process_camera(self, batched_inputs):
        tran_cls = [x["rel_pose"]["tran_cls"].to(self.device) for x in batched_inputs]
        rot_cls = [x["rel_pose"]["rot_cls"].to(self.device) for x in batched_inputs]
        tran_cls = torch.stack(tran_cls)
        rot_cls = torch.stack(rot_cls)
        return {"tran_cls": tran_cls, "rot_cls": rot_cls}

    @staticmethod
    def _postprocess(
        instances, batched_inputs, image_sizes, mask_threshold=0.5, nms=False
    ):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(
                results_per_image, height, width, mask_threshold, nms=nms
            )
            processed_results.append({"instances": r})
        return processed_results
