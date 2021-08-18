import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.visualizer import GenericMask, ColorMode

sns.set()

cmap = [
    [79, 129, 189],
    [155, 187, 89],
    [128, 100, 162],
    [192, 80, 77],
    [75, 172, 198],
    [31, 73, 125],
    [247, 150, 70],
    [238, 236, 225],
]

reds = [
    [242, 220, 219],
    [230, 184, 183],
    [218, 150, 148],
    [150, 54, 52],
    [99, 37, 35],
]

oranges = [
    [253, 233, 217],
    [252, 213, 180],
    [250, 191, 143],
    [247, 150, 70],
    [226, 107, 10],
]

purples = [
    [204, 192, 218],
    [176, 163, 190],
    [148, 134, 163],
    [120, 106, 135],
    [64, 49, 80],
]

sns_color = [(np.array(c) * 255).astype(int) for c in sns.color_palette("OrRd", 5)]


def save_affinity_after_stitch(affinity_pred, sz_i, sz_j, matching, mesh_dir):
    try:
        max_sz = max(sz_i, sz_j)
        if max_sz < 5:
            max_sz = 5
        elif max_sz < 10:
            max_sz = 10
        text = np.array([[""] * sz_j] * sz_i)
        for i, j in enumerate(matching):
            if j != -1:
                text[i][j] = "*"
        affinity_vis = affinity_pred[:max_sz, :max_sz]
        labels = (
            np.asarray(
                [
                    "{}\n{:.2f}".format(text, data)
                    for text, data in zip(
                        text.flatten(), affinity_pred[:sz_i, :sz_j].flatten()
                    )
                ]
            )
        ).reshape(text.shape)
        labels_full = np.array([[""] * max_sz] * max_sz).astype("<U6")
        labels_full[:sz_i, :sz_j] = labels
        plt.figure()
        sns.heatmap(affinity_vis, annot=labels_full, fmt="s", vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, "affinity_pred.png"))
        plt.close()
    except:
        plt.figure()
        sns.heatmap(affinity_pred[:max_sz, :max_sz], vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, "affinity_pred.png"))
        plt.close()
        pass


def save_aff(aff, save_path):
    pass


def get_loc_white(bbox):
    x1, y1, x2, y2 = bbox
    return [x1 + 4, y1 + 4, x2 - 4, y2 - 4]


def load_img(mat_path, img_path):
    mat = loadmat(mat_path)
    bbox = mat["bboxes"]
    im = Image.open(img_path)
    return im, bbox


def draw_bbox(img1, img2, bbox1, bbox2, matching_proposals):
    try:
        d1 = ImageDraw.Draw(img1)
        d2 = ImageDraw.Draw(img2)
        cmap_idx = 0
        for idx1, idx2 in enumerate(matching_proposals):
            if idx2 == -1:
                d1.rectangle(bbox1[idx1], fill=None, outline=(0, 0, 0), width=5)
            else:
                d1.rectangle(
                    bbox1[idx1], fill=None, outline=tuple(cmap[cmap_idx]), width=10
                )
                d1.rectangle(
                    get_loc_white(bbox1[idx1]),
                    fill=None,
                    outline=(255, 255, 255),
                    width=2,
                )
                d2.rectangle(
                    bbox2[idx2], fill=None, outline=tuple(cmap[cmap_idx]), width=10
                )
                d2.rectangle(
                    get_loc_white(bbox2[idx2]),
                    fill=None,
                    outline=(255, 255, 255),
                    width=2,
                )
                cmap_idx += 1
        for idx, box in enumerate(bbox2):
            if idx not in matching_proposals:
                d2.rectangle(box, fill=None, outline=(0, 0, 0), width=5)
        return img1, img2
    except:
        import pdb; pdb.set_trace()
        pass


def get_concat_v(im1, im2, distance=50, vertical=True):
    if vertical:
        dst = Image.new(
            "RGBA", (im1.width, im1.height + distance + im2.height), (255, 0, 0, 0)
        )
        dst.paste(im2, (0, distance + im1.height))
    else:
        dst = Image.new(
            "RGBA", (im1.width + distance + im2.width, im1.height), (255, 0, 0, 0)
        )
        dst.paste(im2, (distance + im1.width, 0))
    dst.paste(im1, (0, 0))
    return dst


def get_centers(bbox):
    return bbox.get_centers().numpy()


def draw_dot(d, center, color, factor, dotsize=20):
    outer_offset = int(dotsize * factor)
    inner_offset = int(dotsize / 20 * 16 * factor)
    outer_bbox = (
        center[0] - outer_offset,
        center[1] - outer_offset,
        center[0] + outer_offset,
        center[1] + outer_offset,
    )
    inner_bbox = (
        center[0] - inner_offset,
        center[1] - inner_offset,
        center[0] + inner_offset,
        center[1] + inner_offset,
    )
    d.ellipse(
        outer_bbox,
        fill=tuple(color),
        outline=tuple(color),
        width=int(dotsize / 20 * 5 * factor),
    )
    d.ellipse(
        inner_bbox,
        fill=None,
        outline=(255, 255, 255),
        width=int(dotsize / 20 * 4 * factor),
    )


def draw_match(
    img1_path,
    img2_path,
    box1,
    box2,
    matching_proposals,
    correct_list,
    pred_aff=None,
    before=True,
    th=0.5,
    distance=45,
    factor=4,
    vertical=True,
    dotsize=20,
    outlier_color=None,
):
    # factor: resize the image before drawing, resume after finishing. This avoids artifacts and draw high resolution lines.
    if isinstance(img1_path, str) and isinstance(img2_path, str):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
    else:
        img1 = Image.fromarray(img1_path)
        img2 = Image.fromarray(img2_path)
    img1 = img1.resize((img1.width * factor, img1.height * factor))
    img2 = img2.resize((img2.width * factor, img2.height * factor))

    if type(box1) == np.ndarray:
        centers1 = box1
        centers2 = box2
    else:
        centers1 = get_centers(box1)
        centers2 = get_centers(box2)

    centers1 = [np.floor(center * factor).astype(np.int32) for center in centers1]
    centers2 = [np.floor(center * factor).astype(np.int32) for center in centers2]
    distance *= factor

    concat = get_concat_v(img1, img2, distance, vertical)
    d = ImageDraw.Draw(concat)

    if vertical:
        offset = distance + img1.height
    else:
        offset = distance + img1.width

    # Draw black bbox
    # d.rectangle((0,0,img1.width,img1.height), fill=None, outline=(0,0,0), width=7*factor)
    # if vertical:
    #     d.rectangle((0,offset,img2.width,img2.height+offset), fill=None, outline=(0,0,0), width=7*factor)
    # else:
    #     d.rectangle((offset, 0, img2.width + offset, img2.height), fill=None, outline=(0,0,0), width=7*factor)
    # Plot black dots for non matching objs
    for i in range(len(centers1)):
        if len(matching_proposals):
            notblack = matching_proposals[:, 0]
        else:
            notblack = []
        if i not in notblack:
            center = centers1[i]
            draw_dot(d, center, (0, 0, 0), factor, dotsize=dotsize)

    for j in range(len(centers2)):
        if len(matching_proposals):
            notblack = matching_proposals[:, 1]
        else:
            notblack = []
        if j not in notblack:
            if vertical:
                center = centers2[j] + np.array([0, offset])
            else:
                center = centers2[j] + np.array([offset, 0])
            draw_dot(d, center, (0, 0, 0), factor, dotsize=dotsize)
    # Draw line
    for [i, j], color_id in zip(matching_proposals, correct_list):
        if color_id == 1:
            # color = [0,176,80] # Green
            color = [26, 133, 255]  # Blue
        else:
            if outlier_color is None:
                # color = [255,0,0] # Red
                color = [212, 17, 89]  # Red
            else:
                color = outlier_color
        width_factor = 1
        center1 = centers1[i]
        if vertical:
            line = (
                centers1[i][0],
                centers1[i][1],
                centers2[j][0],
                centers2[j][1] + offset,
            )
        else:
            line = (
                centers1[i][0],
                centers1[i][1],
                centers2[j][0] + offset,
                centers2[j][1],
            )
        d.line(line, fill=tuple(color), width=int(7 * width_factor * factor))
        d.line(line, fill=(255, 255, 255), width=int(2 * width_factor * factor))

    # Plot colored dots for matching objs
    for [i, j] in matching_proposals:
        color = purples[-1]
        center1 = centers1[i]
        draw_dot(d, center1, color, factor, dotsize=dotsize)
        if vertical:
            center2 = centers2[j] + np.array([0, offset])
        else:
            center2 = centers2[j] + np.array([offset, 0])
        draw_dot(d, list(center2), color, factor, dotsize=dotsize)
    # concat = concat.resize((int(concat.width/factor), int(concat.height/factor)))
    return concat


def create_instances(predictions, image_size, pred_planes=None, conf_threshold=0.7):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    labels = np.asarray([predictions[i]["category_id"] for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels
    ret.pred_planes = np.asarray([pred_planes[i] for i in chosen])

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except:
        pass
    return ret


def get_labeled_seg(
    predictions, score_threshold, vis, assigned_colors=None, paper_img=False
):
    boxes = predictions.pred_boxes
    scores = predictions.scores
    classes = predictions.pred_classes
    chosen = (scores > score_threshold).nonzero()[0]
    boxes = boxes[chosen]
    scores = scores[chosen]
    classes = classes[chosen]
    # labels = list(range(len(predictions)))
    labels = [f"{idx}: {score:.2f}" for idx, score in enumerate(scores)]
    masks = np.asarray(predictions.pred_masks)
    masks = [GenericMask(x, vis.output.height, vis.output.width) for x in masks]
    alpha = 0.5

    if vis._instance_mode == ColorMode.IMAGE_BW:
        vis.output.img = vis._create_grayscale_image(
            (predictions.pred_masks.any(dim=0) > 0).numpy()
        )
        alpha = 0.3
    if paper_img:
        boxes = None
        labels = None
    vis.overlay_instances(
        masks=masks,
        assigned_colors=assigned_colors,
        boxes=boxes,
        labels=labels,
        alpha=alpha,
    )
    seg_pred = vis.output.get_image()
    return seg_pred


def get_gt_labeled_seg(dic, vis, assigned_colors=None, paper_img=False):
    """
    Draw annotations/segmentaions in Detectron2 Dataset format.

    Args:
        dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

    Returns:
        output (VisImage): image object with visualizations.
    """
    annos = dic.get("annotations", None)
    if annos:
        if "segmentation" in annos[0]:
            masks = [x["segmentation"] for x in annos]
        else:
            masks = None

        boxes = [
            BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos
        ]

        labels = [f"{idx}: gt" for idx in range(len(annos))]
        if paper_img:
            labels = None
            boxes = None
        vis.overlay_instances(
            labels=labels,
            boxes=boxes,
            masks=masks,
            assigned_colors=assigned_colors,
        )
    seg_gt = vis.output.get_image()
    return seg_gt
