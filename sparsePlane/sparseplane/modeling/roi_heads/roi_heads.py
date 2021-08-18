from typing import Dict
import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import (
    StandardROIHeads,
    select_foreground_proposals,
)
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss
from detectron2.structures import Boxes, Instances

from sparseplane.modeling.roi_heads.embedding_head import build_embedding_head
from sparseplane.modeling.roi_heads.plane_head import (
    build_plane_head,
)


@ROI_HEADS_REGISTRY.register()
class PlaneRCNNROIHeads(StandardROIHeads):
    """
    The ROI specific heads for Mesh R-CNN
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self._init_embedding_head(cfg, input_shape)
        self._init_plane_head(cfg, input_shape)
        self._plane_roi_box_ratio = (
            cfg.MODEL.ROI_PLANE_HEAD.PLANE_POOLER_SCALE
        )  # Receptive Field
        self._embedding_roi_box_ratio = (
            cfg.MODEL.ROI_EMBEDDING_HEAD.EMBEDDING_POOLER_SCALE
        )  # Receptive Field
        self._embedding_gt_box = cfg.MODEL.ROI_EMBEDDING_HEAD.TRAIN_WITH_GT_BOX
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX
        self._freeze = cfg.MODEL.FREEZE

    def _init_embedding_head(self, cfg, input_shape):
        self.embedding_on = cfg.MODEL.EMBEDDING_ON
        self._asnet_on = cfg.MODEL.ROI_EMBEDDING_HEAD.NAME == "EmbeddingRCNNASNetHead"
        if not self.embedding_on:
            return
        embedding_pooler_resolution = cfg.MODEL.ROI_EMBEDDING_HEAD.POOLER_RESOLUTION
        embedding_pooler_scales = tuple(
            1.0 / input_shape[k].stride for k in self.in_features
        )
        embedding_sampling_ratio = cfg.MODEL.ROI_EMBEDDING_HEAD.POOLER_SAMPLING_RATIO
        embedding_pooler_type = cfg.MODEL.ROI_EMBEDDING_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in self.in_features][0]
        if not self._asnet_on:
            self.embedding_pooler = ROIPooler(
                output_size=embedding_pooler_resolution,
                scales=embedding_pooler_scales,
                sampling_ratio=embedding_sampling_ratio,
                pooler_type=embedding_pooler_type,
            )
        else:
            self.embedding_pooler = ROIPooler(
                output_size=embedding_pooler_resolution * 2,
                scales=embedding_pooler_scales,
                sampling_ratio=embedding_sampling_ratio,
                pooler_type=embedding_pooler_type,
            )
            self.H_size = 14
            self.W_size = 14
            self.H_min = 7
            self.H_max = 21
            self.W_min = 7
            self.W_max = 21
        shape = ShapeSpec(
            channels=in_channels,
            width=embedding_pooler_resolution,
            height=embedding_pooler_resolution,
        )
        self.embedding_head = build_embedding_head(cfg, shape)

    def _init_plane_head(self, cfg, input_shape):
        self.plane_on = cfg.MODEL.PLANE_ON
        if not self.plane_on:
            return

        plane_pooler_resolution = cfg.MODEL.ROI_PLANE_HEAD.POOLER_RESOLUTION
        plane_pooler_scales = tuple(
            1.0 / input_shape[k].stride for k in self.in_features
        )
        plane_sampling_ratio = cfg.MODEL.ROI_PLANE_HEAD.POOLER_SAMPLING_RATIO
        plane_pooler_type = cfg.MODEL.ROI_PLANE_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.plane_pooler = ROIPooler(
            output_size=plane_pooler_resolution,
            scales=plane_pooler_scales,
            sampling_ratio=plane_sampling_ratio,
            pooler_type=plane_pooler_type,
        )
        shape = ShapeSpec(
            channels=in_channels,
            width=plane_pooler_resolution,
            height=plane_pooler_resolution,
        )
        self.plane_head = build_plane_head(cfg, shape)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        feature: feature from FPN
        """
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self.training:
            losses = {}
            if "roi_heads.box_head" not in self._freeze:
                losses = self._forward_box(features, proposals)
            pred_instances, roi_losses = self.forward_with_selected_boxes(
                features, proposals, targets
            )
            losses.update(roi_losses)
            del targets
            return pred_instances, losses
        else:
            if self._eval_gt_box:
                pred_instances = [Instances(p._image_size) for p in proposals]
                for ins, p in zip(pred_instances, proposals):
                    ins.pred_boxes = p.gt_boxes
                    ins.scores = torch.ones(len(p.gt_boxes)).to("cuda")
                    ins.pred_classes = torch.zeros(len(p.gt_boxes)).to("cuda")
            else:
                pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_selected_boxes(self, features, instances, targets=None):
        assert self.training
        losses = {}
        if self._embedding_gt_box:
            proposals = targets
        else:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
        pred_instances = proposals
        if "roi_heads.mask_head" not in self._freeze:
            losses.update(self._forward_mask(features, pred_instances))
        if "roi_heads.plane_head" not in self._freeze:
            plane_loss, pred_instances = self._forward_plane(features, pred_instances)
            losses.update(plane_loss)
        if "roi_heads.embedding_head" not in self._freeze:
            pred_instances = self._forward_embedding(features, pred_instances)
        return pred_instances, losses

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances): the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_voxels`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        instances = self._forward_mask(features, instances)
        instances = self._forward_plane(features, instances)
        instances = self._forward_embedding(features, instances)
        return instances

    def increase_receptive_field(self, proposal_boxes, ratio):
        """
        scale bbox to have larger reception field.
        """
        centers = [box.get_centers() for box in proposal_boxes]
        dxs = [box.tensor[:, 2] - box.tensor[:, 0] for box in proposal_boxes]
        dys = [box.tensor[:, 3] - box.tensor[:, 1] for box in proposal_boxes]
        new_boxes = []
        for center, dx, dy in zip(centers, dxs, dys):
            new_boxes.append(
                Boxes(
                    torch.stack(
                        [
                            center[:, 0] - dx * ratio / 2,
                            center[:, 1] - dy * ratio / 2,
                            center[:, 0] + dx * ratio / 2,
                            center[:, 1] + dy * ratio / 2,
                        ],
                        axis=1,
                    )
                )
            )
        return new_boxes

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str,Tensor]): mapping from names to backbone features
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = instances
            if self._embedding_gt_box:
                # Use GT box to train
                proposal_boxes = [x.gt_boxes for x in proposals]
            else:
                # Use Pred box to train
                proposal_boxes = [x.proposal_boxes for x in proposals]
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head.layers(mask_features)
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_plane(self, features, instances):
        if not self.plane_on:
            return {}, instances if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = instances
            if self._embedding_gt_box:
                # Use GT box to train
                proposal_boxes = [x.gt_boxes for x in proposals]
            else:
                # Use Pred box to train
                proposal_boxes = [x.proposal_boxes for x in proposals]
        else:
            proposal_boxes = [x.pred_boxes for x in instances]
        new_boxes = self.increase_receptive_field(
            proposal_boxes, ratio=self._plane_roi_box_ratio
        )
        plane_features = self.plane_pooler(features, new_boxes)

        if self.training:
            return self.plane_head(plane_features, proposals)
        else:
            return self.plane_head(plane_features, instances)

    def _forward_embedding(self, features, instances):
        """
        Forward logic of the embedding prediction branch.
        targets: gt boxes
        instances: predicted boxes
        """
        if not self.embedding_on:
            return instances
        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = instances
            if self._embedding_gt_box:
                # Use GT box to train
                proposal_boxes = [x.gt_boxes for x in proposals]
            else:
                # Use Pred box to train
                proposal_boxes = [x.proposal_boxes for x in proposals]
        else:
            proposal_boxes = [x.pred_boxes for x in instances]
        if not self._asnet_on:
            new_boxes = self.increase_receptive_field(
                proposal_boxes, ratio=self._embedding_roi_box_ratio
            )
            embedding_features = self.embedding_pooler(features, new_boxes)
        else:
            new_boxes = self.increase_receptive_field(
                proposal_boxes, ratio=self._embedding_roi_box_ratio * 2.0
            )
            x = self.embedding_pooler(features, new_boxes)
            B, C, H, W = x.size()
            center = x[:, :, self.H_min : self.H_max, self.W_min : self.W_max].clone()
            surrnd = x
            surrnd[
                :, :, self.H_min : self.H_max, self.W_min : self.W_max
            ] = torch.zeros((B, C, self.H_size, self.W_size)).to(x.device)
            embedding_features = {"center": center, "surrnd": surrnd}

        if self.training:
            return self.embedding_head(embedding_features, proposals)
        else:
            return self.embedding_head(embedding_features, instances)
