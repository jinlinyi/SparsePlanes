import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

ROI_PLANE_HEAD_REGISTRY = Registry("ROI_PLANE_HEAD")


@ROI_PLANE_HEAD_REGISTRY.register()
class PlaneRCNNConvFCHead(nn.Module):
    """
    A head with several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_fc: the number of fc layers
            fc_dim: the dimension of the fc layers
        """
        super().__init__()

        # fmt: off
        num_conv        = cfg.MODEL.ROI_EMBEDDING_HEAD.NUM_CONV
        conv_dim        = cfg.MODEL.ROI_EMBEDDING_HEAD.CONV_DIM
        num_fc          = cfg.MODEL.ROI_PLANE_HEAD.NUM_FC
        fc_dim          = cfg.MODEL.ROI_PLANE_HEAD.FC_DIM
        param_dim       = cfg.MODEL.ROI_PLANE_HEAD.PARAM_DIM
        norm            = cfg.MODEL.ROI_PLANE_HEAD.NORM
        # fmt: on
        self._plane_normal_only = cfg.MODEL.ROI_PLANE_HEAD.NORMAL_ONLY
        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("plane_conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("plane_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        self.param_pred = nn.Linear(fc_dim, param_dim)

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        self._loss_weight = cfg.MODEL.ROI_PLANE_HEAD.LOSS_WEIGHT

        # self.debug = self.conv_norm_relus[0].weight.data.cpu().numpy()

    def forward(self, x, instances):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))

        x = self.param_pred(x)
        if self._plane_normal_only:
            x = F.normalize(x, p=2, dim=1)

        if self.training:
            plane_rcnn_inference(x, instances)
            return {
                "loss_plane": plane_rcnn_loss(
                    x,
                    instances,
                    loss_weight=self._loss_weight,
                    plane_normal_only=self._plane_normal_only,
                )
            }, instances
        else:
            plane_rcnn_inference(x, instances)
            return instances

    @property
    def output_size(self):
        return self._output_size


def plane_rcnn_loss(
    plane_pred, instances, loss_weight=1.0, smooth_l1_beta=0.0, plane_normal_only=False
):
    """
    Compute the plane_param loss.
    Args:
        z_pred (Tensor): A tensor of shape (B, C) or (B, 1) for class-specific or class-agnostic,
            where B is the total number of foreground regions in all images, C is the number of foreground classes,
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        loss (Tensor): A scalar tensor containing the loss.
    """
    gt_param = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        gt_param.append(instances_per_image.gt_planes)

    if len(gt_param) == 0:
        return plane_pred.sum() * 0

    gt_param = cat(gt_param, dim=0)
    if plane_normal_only:
        gt_param = F.normalize(gt_param, p=2, dim=1)
    assert len(plane_pred) > 0

    loss_plane_reg = smooth_l1_loss(
        plane_pred, gt_param, smooth_l1_beta, reduction="sum"
    )
    loss_plane_reg = loss_weight * loss_plane_reg / len(plane_pred)
    return loss_plane_reg


def plane_rcnn_inference(plane_pred, pred_instances):
    num_boxes_per_image = [len(i) for i in pred_instances]
    plane_pred = plane_pred.split(num_boxes_per_image, dim=0)

    for plane, instances in zip(plane_pred, pred_instances):
        instances.pred_plane = plane


def build_plane_head(cfg, input_shape):
    name = cfg.MODEL.ROI_PLANE_HEAD.NAME
    return ROI_PLANE_HEAD_REGISTRY.get(name)(cfg, input_shape)
