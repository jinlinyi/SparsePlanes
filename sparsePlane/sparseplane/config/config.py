# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def get_sparseplane_cfg_defaults(cfg):
    """
    Customize the detectron2 cfg to include some new keys and default values
    for SparsePlane
    """
    cfg.MODEL.FREEZE = []
    cfg.MODEL.EMBEDDING_ON = True
    cfg.MODEL.PLANE_ON = True
    cfg.MODEL.DEPTH_ON = False
    cfg.MODEL.CAMERA_ON = False
    cfg.MODEL.META_ARCHITECTURE = "SiamesePlaneRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    # ------------------------------------------------------------------------ #
    # ROI Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_HEADS.NAME = "PlaneRCNNROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # ------------------------------------------------------------------------ #
    # Box Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7

    # aspect ratio grouping has no difference in performance
    # but might reduce memory by a little bit
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.DATALOADER.AUGMENTATION = False
    cfg.TEST.SAVE_VIS = False
    cfg.TEST.EVAL_GT_BOX = False

    # ------------------------------------------------------------------------ #
    # Embedding Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_EMBEDDING_HEAD = CN()
    cfg.MODEL.ROI_EMBEDDING_HEAD.NAME = "EmbeddingRCNNConvFCHead"
    cfg.MODEL.ROI_EMBEDDING_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_EMBEDDING_HEAD.POOLER_SAMPLING_RATIO = 2
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_EMBEDDING_HEAD.POOLER_TYPE = "ROIAlign"
    cfg.MODEL.ROI_EMBEDDING_HEAD.EMBEDDING_DIM = 128
    cfg.MODEL.ROI_EMBEDDING_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_EMBEDDING_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_EMBEDDING_HEAD.NUM_FC = 1
    cfg.MODEL.ROI_EMBEDDING_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_EMBEDDING_HEAD.NORM = ""
    cfg.MODEL.ROI_EMBEDDING_HEAD.MARGIN = 0.2
    cfg.MODEL.ROI_EMBEDDING_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_EMBEDDING_HEAD.EMBEDDING_POOLER_SCALE = 1.0
    cfg.MODEL.ROI_EMBEDDING_HEAD.TRIPLET_SELECTOR_TYPE = "random"
    cfg.MODEL.ROI_EMBEDDING_HEAD.RELATION_NET = False
    cfg.MODEL.ROI_EMBEDDING_HEAD.LOSS_TYPE = "TripletLoss"
    cfg.MODEL.ROI_EMBEDDING_HEAD.TRAIN_WITH_GT_BOX = False
    # ------------------------------------------------------------------------ #
    # Plane Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_PLANE_HEAD = CN()
    cfg.MODEL.ROI_PLANE_HEAD.NAME = "PlaneRCNNConvFCHead"
    cfg.MODEL.ROI_PLANE_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_PLANE_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_PLANE_HEAD.PLANE_POOLER_SCALE = 1.0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_PLANE_HEAD.POOLER_TYPE = "ROIAlign"
    cfg.MODEL.ROI_PLANE_HEAD.EMBEDDING_DIM = 128
    cfg.MODEL.ROI_PLANE_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_PLANE_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_PLANE_HEAD.NUM_FC = 1
    cfg.MODEL.ROI_PLANE_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_PLANE_HEAD.NORM = ""
    cfg.MODEL.ROI_PLANE_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_PLANE_HEAD.PARAM_DIM = 3
    cfg.MODEL.ROI_PLANE_HEAD.NORMAL_ONLY = True

    # ------------------------------------------------------------------------ #
    # Camera Predict Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.CAMERA_BRANCH = "CACHED"  # For inference

    cfg.MODEL.CAMERA_HEAD = CN()
    cfg.MODEL.CAMERA_HEAD.NAME = "PlaneRCNNCameraHead"
    cfg.MODEL.CAMERA_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH = "./models/kmeans_trans_32.pkl"
    cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH = "./models/kmeans_rots_32.pkl"
    cfg.MODEL.CAMERA_HEAD.TRANS_CLASS_NUM = 32
    cfg.MODEL.CAMERA_HEAD.ROTS_CLASS_NUM = 32
    cfg.MODEL.CAMERA_HEAD.FEATURE_SIZE = 64
    cfg.MODEL.CAMERA_HEAD.BACKBONE_FEATURE = "p3"

    # ------------------------------------------------------------------------ #
    # Depth Predict Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.DEPTH_HEAD = CN()
    cfg.MODEL.DEPTH_HEAD.NAME = "PlaneRCNNDepthHead"
    cfg.MODEL.DEPTH_HEAD.LOSS_WEIGHT = 1.0
    # ------------------------------------------------------------------------ #
    # Mask Predict Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_MASK_HEAD.MASK_THRESHOLD = 0.5
    cfg.MODEL.ROI_MASK_HEAD.NMS = False
    cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 2
    cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlign"

    return cfg
