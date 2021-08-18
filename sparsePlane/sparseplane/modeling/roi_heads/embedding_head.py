import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F

ROI_EMBEDDING_HEAD_REGISTRY = Registry("ROI_EMBEDDING_HEAD")


def embedding_rcnn_inference(embeddings, pred_instances):
    """
    For each box, the embedding of the same class is attached to the instance by adding a
    new "embedding" field to pred_instances.
    Args:
        embeddings (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "embedding" field storing an embedding for each plane.
    """
    num_boxes_per_image = [len(i) for i in pred_instances]
    embeddings = embeddings.split(num_boxes_per_image, dim=0)

    for embedding, instances in zip(embeddings, pred_instances):
        instances.embedding = embedding


@ROI_EMBEDDING_HEAD_REGISTRY.register()
class EmbeddingRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
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
        num_fc          = cfg.MODEL.ROI_EMBEDDING_HEAD.NUM_FC
        fc_dim          = cfg.MODEL.ROI_EMBEDDING_HEAD.FC_DIM
        emb_dim         = cfg.MODEL.ROI_EMBEDDING_HEAD.EMBEDDING_DIM
        norm            = cfg.MODEL.ROI_EMBEDDING_HEAD.NORM
        # fmt: on

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
            self.add_module("embedding_conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("embedding_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        self.embedding_pred = nn.Linear(fc_dim, emb_dim)

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))

        x = self.embedding_pred(x)
        # normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        embedding_rcnn_inference(x, instances)

        return instances

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


@ROI_EMBEDDING_HEAD_REGISTRY.register()
class EmbeddingRCNNASNetHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
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
        num_fc          = cfg.MODEL.ROI_EMBEDDING_HEAD.NUM_FC
        fc_dim          = cfg.MODEL.ROI_EMBEDDING_HEAD.FC_DIM
        emb_dim         = cfg.MODEL.ROI_EMBEDDING_HEAD.EMBEDDING_DIM
        norm            = cfg.MODEL.ROI_EMBEDDING_HEAD.NORM
        # fmt: on

        self._output_size_app = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )
        self.app = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size_app[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("embedding_app{}".format(k + 1), conv)
            self.app.append(conv)
            self._output_size_app = (
                conv_dim,
                self._output_size_app[1],
                self._output_size_app[2],
            )
        self.fcs_app = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size_app), fc_dim)
            self.add_module("embedding_fc_app{}".format(k + 1), fc)
            self.fcs_app.append(fc)

        self._output_size_sur = (
            input_shape.channels,
            input_shape.height * 2,
            input_shape.width * 2,
        )
        self.sur = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size_sur[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("embedding_sur{}".format(k + 1), conv)
            self.sur.append(conv)
            self._output_size_sur = (
                conv_dim,
                self._output_size_sur[1],
                self._output_size_sur[2],
            )
        self.sur_pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        )
        self.fcs_sur = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size_app), fc_dim)  # because of maxpool
            self.add_module("embedding_fc_sur{}".format(k + 1), fc)
            self.fcs_sur.append(fc)

        self.embedding_pred_app = nn.Linear(fc_dim, emb_dim)
        self.embedding_pred_sur = nn.Linear(fc_dim, emb_dim)
        for layer in self.app:
            weight_init.c2_msra_fill(layer)
        for layer in self.sur:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs_app:
            weight_init.c2_xavier_fill(layer)
        for layer in self.fcs_sur:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances):
        # center
        for layer in self.app:
            x["center"] = layer(x["center"])
        if len(self.fcs_app):
            if x["center"].dim() > 2:
                x["center"] = torch.flatten(x["center"], start_dim=1)
            for layer in self.fcs_app:
                x["center"] = F.relu(layer(x["center"]))
        # surrnd
        for layer in self.sur:
            x["surrnd"] = layer(x["surrnd"])
        x["surrnd"] = self.sur_pool(x["surrnd"])
        if len(self.fcs_sur):
            if x["surrnd"].dim() > 2:
                x["surrnd"] = torch.flatten(x["surrnd"], start_dim=1)
            for layer in self.fcs_sur:
                x["surrnd"] = F.relu(layer(x["surrnd"]))
        x["center"] = self.embedding_pred_app(x["center"])
        x["surrnd"] = self.embedding_pred_sur(x["surrnd"])
        x["center"] = F.normalize(x["center"], p=2, dim=1)
        x["surrnd"] = F.normalize(x["surrnd"], p=2, dim=1)
        num_boxes_per_image = [len(i) for i in instances]
        center_embeddings = x["center"].split(num_boxes_per_image, dim=0)
        surrnd_embeddings = x["surrnd"].split(num_boxes_per_image, dim=0)

        for c, s, instance in zip(center_embeddings, surrnd_embeddings, instances):
            instance.embedding_c = c
            instance.embedding_s = s
        return instances

    @property
    def output_size(self):
        return self._output_size_sur

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


def build_embedding_head(cfg, input_shape):
    """
    Build an embedding head defined by `cfg.MODEL.ROI_EMBEDDING_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_EMBEDDING_HEAD.NAME
    return ROI_EMBEDDING_HEAD_REGISTRY.get(name)(cfg, input_shape)
