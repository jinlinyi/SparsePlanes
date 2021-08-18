import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import quaternion
import os
import time

from scipy.special import huber
from detectron2.data import DatasetCatalog, MetadataCatalog
from sparseplane.data import PlaneRCNNMapper
from sparseplane.utils.vis import get_pcd


def load_input_dataset(dataset):
    dataset_dict = {}
    dataset_list = list(DatasetCatalog.get(dataset))
    for dic in dataset_list:
        key0 = dic["0"]["image_id"]
        key1 = dic["1"]["image_id"]
        key = key0 + "__" + key1
        dataset_dict[key] = dic
    return dataset_dict


def segm2mask(segm, image_height, image_width):
    """
    convert coco format segmentation mask to binary mask
    """
    if isinstance(segm, list):
        polygons = [np.array(p, dtype=np.float64) for p in segm]
        rles = mask_util.frPyObjects(polygons, image_height, image_width)
        rle = mask_util.merge(rles)
    elif isinstance(segm, dict):  # RLE
        rle = segm
    else:
        raise TypeError(f"Unknown segmentation type {type(segm)}!")
    return mask_util.decode(rle)


def get_sift_from_normalized_image(img, box, segm, plane, pixel_per_meter=500):
    """
    Input: original image, bbox, segmentation, planeParameter
    Output: keypoints (in an arbitrary normalized frame), descriptor, and projection matrix H kp = H@kp_original_cam

    Extract sift feature from textures captured when camera is facing directly towards the plane.
    http://frahm.web.unc.edu/files/2014/01/3D-Model-Matching-with-Viewpoint-Invariant-Patches-VIP.pdf
    1. Warp the image texture onto the local tangential plane.
    2. Project the texture into an orthographic camera
    with viewing direction parallel to the local tangential
    plane’s normal.
    3. Extract the sift descriptor from the orthographic
    patch projection.
    """
    image_height, image_width, _ = img.shape
    mask = segm2mask(segm, image_height, image_width) == 0
    offset = np.linalg.norm(plane)
    normal = plane / (offset + 1e-5)
    verts = np.array(
        [
            [box[0], box[1]],
            [box[0] + box[2], box[1]],
            [box[0], box[1] + box[3]],
            [box[0] + box[2], box[1] + box[3]],
        ]
    )
    pcd = get_pcd(verts, normal, offset)
    # get rectified vertices location
    x_hat = (pcd[1] - pcd[0]) / np.linalg.norm(pcd[1] - pcd[0] + 1e-5)
    z_hat = normal
    y_hat = np.cross(z_hat, x_hat)
    verts_rect = (
        np.vstack((np.dot(pcd - pcd[0], x_hat), np.dot(pcd - pcd[0], y_hat))).T
        * pixel_per_meter
    )
    verts_rect = verts_rect - np.array([min(verts_rect[:, 0]), min(verts_rect[:, 1])])
    if verts_rect.reshape(-1).max() > pixel_per_meter * 10:
        # print(f"Large planes: {verts_rect.reshape(-1).max()}")
        verts_rect = verts_rect / (verts_rect.reshape(-1).max() / pixel_per_meter / 10)
    # Find homography
    H = cv2.getPerspectiveTransform(
        np.array(verts, np.float32), np.array(verts_rect, np.float32)
    )
    # Warp image
    src_img = img.copy()
    im_dst = cv2.warpPerspective(
        src_img, H, (int(max(verts_rect[:, 0])), int(max(verts_rect[:, 1])))
    )
    # Area of Interest based on segmentation mask
    mask_dst = cv2.warpPerspective(
        (mask == False).astype("uint8"),
        H,
        (int(max(verts_rect[:, 0])), int(max(verts_rect[:, 1]))),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask_dst = cv2.erode(
        mask_dst, kernel, iterations=5
    )  # do not want keypoints at boundary
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.01, sigma=0.8)
    (kps, descs) = sift.detectAndCompute(im_dst, mask=mask_dst)
    return kps, descs, H


def draw_matches(img1, kp1, img2, kp2, matches, color=None):
    """
    Copy from https://gist.github.com/isker/11be0c50c4f78cad9549
    Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (
            max(img1.shape[0], img2.shape[0]),
            img1.shape[1] + img2.shape[1],
            img1.shape[2],
        )
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0 : img1.shape[0], 0 : img1.shape[1]] = img1
    new_img[0 : img2.shape[0], img1.shape[1] : img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 5
    thickness = 1
    # Draw keypoints.
    for k in kp1:
        end1 = tuple(np.round(k.pt).astype(int))
        cv2.circle(
            new_img, end1, r, np.array([0, 0, 0], dtype="uint8").tolist(), thickness
        )
    for k in kp2:
        end2 = tuple(np.round(k.pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.circle(
            new_img, end2, r, np.array([0, 0, 0], dtype="uint8").tolist(), thickness
        )

    r = 15
    thickness = 2

    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = (
                np.random.randint(0, 256, 3)
                if len(img1.shape) == 3
                else np.random.randint(0, 256)
            )
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(
            np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0])
        )

        cv2.line(new_img, end1, end2, np.array(c, dtype="uint8").tolist(), thickness)
        cv2.circle(new_img, end1, r, np.array(c, dtype="uint8").tolist(), thickness)
        cv2.circle(new_img, end2, r, np.array(c, dtype="uint8").tolist(), thickness)

    plt.figure(figsize=(15, 15))
    plt.imshow(new_img)
    plt.show()


def get_match(kp1, des1, kp2, des2):
    if des1 is None or des2 is None:
        return []
    # BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    if len(matches) == 0 or len(des1) == 1 or len(des2) == 1:
        return []

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 3  # Affine is 6 DoF

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = (
            np.array([kp1[m.queryIdx].pt for m in good])
            .reshape(-1, 1, 2)
            .astype("int32")
        )
        dst_pts = (
            np.array([kp2[m.trainIdx].pt for m in good])
            .reshape(-1, 1, 2)
            .astype("int32")
        )
        th = max(
            src_pts[:, :, 0].max() - src_pts[:, :, 0].min(),
            src_pts[:, :, 1].max() - src_pts[:, :, 1].min(),
        )
        M, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=th * 0.01
        )
        matchesMask = mask.ravel().tolist()
    else:
        # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = []
    good = [good[i] for i in range(len(matchesMask)) if matchesMask[i] == 1]
    return good


def transform_keypoint(kps, M):
    xy1 = np.array([[kp.pt[0], kp.pt[1], 1] for kp in kps])
    xy1 = (M @ xy1.T).T
    xys = (xy1 / xy1[:, 2].reshape(-1, 1))[:, :2]
    for kp, xy in zip(kps, xys):
        kp.pt = tuple(xy)
    return kps


def get_pixel_matching(img1, box1, segm1, plane1, img2, box2, segm2, plane2):
    kp1, des1, H1 = get_sift_from_normalized_image(img1, box1, segm1, plane1)
    kp2, des2, H2 = get_sift_from_normalized_image(img2, box2, segm2, plane2)
    if des1 is None or des2 is None:
        return [], []
    good = get_match(kp1, des1, kp2, des2)
    kp1_origin = transform_keypoint(kp1, np.linalg.inv(H1))
    kp2_origin = transform_keypoint(kp2, np.linalg.inv(H2))
    xy_origin1 = [np.round(kp1_origin[m.queryIdx].pt).astype(int) for m in good]
    xy_origin2 = [np.round(kp2_origin[m.trainIdx].pt).astype(int) for m in good]
    return xy_origin1, xy_origin2


def draw_matches_xy(img1, kp1, img2, kp2, color=None, save_path="./debug", prefix=None):
    """
    Copy from https://gist.github.com/isker/11be0c50c4f78cad9549
    Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (
            max(img1.shape[0], img2.shape[0]),
            img1.shape[1] + img2.shape[1],
            img1.shape[2],
        )
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0 : img1.shape[0], 0 : img1.shape[1]] = img1
    new_img[0 : img2.shape[0], img1.shape[1] : img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2

    if color:
        c = color
    for i in range(len(kp1)):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = (
                np.random.randint(0, 256, 3)
                if len(img1.shape) == 3
                else np.random.randint(0, 256)
            )
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[i]).astype(int))
        end2 = tuple(np.round(kp2[i]).astype(int) + np.array([img1.shape[1], 0]))

        cv2.line(new_img, end1, end2, np.array(c, dtype="uint8").tolist(), thickness)
        cv2.circle(new_img, end1, r, np.array(c, dtype="uint8").tolist(), thickness)
        cv2.circle(new_img, end2, r, np.array(c, dtype="uint8").tolist(), thickness)
    cv2.imwrite(os.path.join(save_path, prefix + ".png"), new_img)


def plot_pixel_matches(
    img1, boxes1, segms1, planes1, img2, boxes2, segms2, planes2, plane_corr
):
    xys1, xys2 = [], []
    for [i, j] in plane_corr:
        xy1, xy2 = get_pixel_matching(
            img1,
            boxes1[i],
            segms1[i],
            planes1[i],
            img2,
            boxes2[j],
            segms2[j],
            planes2[j],
        )
        if len(xy1) == 0 or len(xy2) == 0:
            continue
        xys1.append(np.array(xy1))
        xys2.append(np.array(xy2))
    xys1 = np.vstack(xys1)
    xys2 = np.vstack(xys2)
    draw_matches_xy(
        img1,
        xys1,
        img2,
        xys2,
        save_path="./debug",
        prefix=time.strftime("%Y%m%d-%H%M%S"),
    )


# def get_pixel_error_predetermined_match(img1, planes1, xys1,
#                                         img2, planes2, xys2,
#                                         R, T):
#     offsets1 = np.linalg.norm(planes1, axis=1)
#     offsets2 = np.linalg.norm(planes2, axis=1)

#     normals1 = planes1 / (offsets1.reshape(-1,1) + 1e-5)
#     normals2 = planes2 / (offsets2.reshape(-1,1) + 1e-5)
#     pcd1s, pcd2s = [], []
#     # for i in range()


def get_pixel_error_online_sift(
    img1, boxes1, segms1, planes1, img2, boxes2, segms2, planes2, R, T, plane_corr
):
    offsets1 = np.linalg.norm(planes1, axis=1)
    offsets2 = np.linalg.norm(planes2, axis=1)

    normals1 = planes1 / (offsets1.reshape(-1, 1) + 1e-5)
    normals2 = planes2 / (offsets2.reshape(-1, 1) + 1e-5)
    pcd1s, pcd2s = [], []
    for [i, j] in plane_corr:
        xy1, xy2 = get_pixel_matching(
            img1,
            boxes1[i],
            segms1[i],
            planes1[i],
            img2,
            boxes2[j],
            segms2[j],
            planes2[j],
        )
        if len(xy1) == 0 or len(xy2) == 0:
            continue
        # project 2 3d
        pcd1 = get_pcd(xy1, normals1[i], offsets1[i]) * np.array([1.0, -1.0, -1.0])
        pcd2 = get_pcd(xy2, normals2[j], offsets2[j]) * np.array([1.0, -1.0, -1.0])

        pcd1s.append(pcd1)
        pcd2s.append(pcd2)
    pcd1s = np.vstack(pcd1s)
    pcd2s = np.vstack(pcd2s)

    if len(pcd1s) == 0:
        err = 0
    else:
        pcd1_glob = (R @ pcd1s.T).T + T
        pcd2_glob = pcd2s
        err = np.linalg.norm(pcd1_glob - pcd2_glob, axis=1).mean()
    return err


def get_pixel_error_precalculated_sift(img1, xys1, planes1, img2, xys2, planes2, R, T):
    assert len(xys1) == len(planes1)
    assert len(xys1) == len(xys2)
    assert len(xys2) == len(planes2)
    offsets1 = np.linalg.norm(planes1, axis=1)
    offsets2 = np.linalg.norm(planes2, axis=1)

    normals1 = planes1 / (offsets1.reshape(-1, 1) + 1e-5)
    normals2 = planes2 / (offsets2.reshape(-1, 1) + 1e-5)
    pcd1s, pcd2s = [], []
    for i in range(len(xys1)):
        xy1, xy2 = xys1[i], xys2[i]
        if len(xy1) == 0 or len(xy2) == 0:
            continue
        # project 2 3d
        pcd1 = get_pcd(xy1, normals1[i], offsets1[i]) * np.array([1.0, -1.0, -1.0])
        pcd2 = get_pcd(xy2, normals2[i], offsets2[i]) * np.array([1.0, -1.0, -1.0])

        pcd1s.append(pcd1)
        pcd2s.append(pcd2)
    if len(pcd1s) == 0 and len(pcd2s) == 0:
        return 0
    pcd1s = np.vstack(pcd1s)
    pcd2s = np.vstack(pcd2s)

    if len(pcd1s) == 0:
        err = 0
    else:
        pcd1_glob = (R @ pcd1s.T).T + T
        pcd2_glob = pcd2s
        err = np.linalg.norm(pcd1_glob - pcd2_glob, axis=1).mean()
        # err = np.minimum(np.linalg.norm(pcd1_glob-pcd2_glob, axis=1), 1.0).mean()
    return err


def main():
    # Load dataset
    dataset = "mp3d_test"
    dataset_dict = load_input_dataset(dataset)

    # Ground truth correspondence
    gt_corrs = [
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 8],
        [9, 11],
        [7, 10],
        [6, 9],
        [11, 9],
        [10, 12],
    ]

    # Testing matching on ground truth correspondence patches
    key1 = "ARNzJeq3xxb_0_0_0__ARNzJeq3xxb_0_0_0"
    img1_file = dataset_dict[key1]["0"]["file_name"]
    img1 = cv2.imread(img1_file, cv2.IMREAD_COLOR)[:, :, ::-1]

    key2 = "ARNzJeq3xxb_0_0_1__ARNzJeq3xxb_0_0_1"
    img2_file = dataset_dict[key2]["0"]["file_name"]
    img2 = cv2.imread(img2_file, cv2.IMREAD_COLOR)[:, :, ::-1]

    with open(
        "/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20/cameras/ARNzJeq3xxb.pkl",
        "rb",
    ) as f:
        cam_info = pickle.load(f)
    R1 = quaternion.as_rotation_matrix(cam_info[0]["rotation"])
    T1 = cam_info[0]["position"]
    R2 = quaternion.as_rotation_matrix(cam_info[1]["rotation"])
    T2 = cam_info[1]["position"]

    pcd1s = []
    pcd2s = []
    for [i, j] in gt_corrs:
        ann1 = dataset_dict[key1]["0"]["annotations"][i]
        ann2 = dataset_dict[key2]["0"]["annotations"][j]

        plane1 = ann1["plane"]
        offset1 = np.linalg.norm(plane1)
        normal1 = plane1 / (offset1 + 1e-5)

        plane2 = ann2["plane"]
        offset2 = np.linalg.norm(plane2)
        normal2 = plane2 / (offset2 + 1e-5)

        xy1, xy2 = get_pixel_matching(
            img1,
            ann1["bbox"],
            ann1["segmentation"],
            plane1,
            img2,
            ann2["bbox"],
            ann2["segmentation"],
            plane2,
        )
        if len(xy1) == 0 or len(xy2) == 0:
            continue
        # project 2 3d
        pcd1 = get_pcd(xy1, normal1, offset1) * np.array([1.0, -1.0, -1.0])
        pcd2 = get_pcd(xy2, normal2, offset2) * np.array([1.0, -1.0, -1.0])

        pcd1s.append(pcd1)
        pcd2s.append(pcd2)
    pcd1s = np.vstack(pcd1s)
    pcd2s = np.vstack(pcd2s)

    if len(pcd1s) == 0:
        err = 0
    else:
        pcd1_glob = (R1 @ pcd1s.T).T + T1
        pcd2_glob = (R2 @ pcd2s.T).T + T2

        err = np.linalg.norm(pcd1_glob - pcd2_glob, axis=1).mean()
    print(err)


def so3ToVec6d(so3):
    return np.array(so3).T.flatten()[:6]


def vec6dToSo3(vec6d):
    assert len(vec6d) == 6
    a1 = np.array(vec6d[:3])
    a2 = np.array(vec6d[3:])
    b1 = a1 / np.max([np.linalg.norm(a1), 1e-8])
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.max([np.linalg.norm(b2), 1e-8])
    b3 = np.cross(b1, b2)
    return np.vstack((b1, b2, b3)).T


def quaternion_from_array(float_array):
    assert len(float_array) == 4
    float_array = np.array(float_array)
    q = float_array / (np.linalg.norm(float_array) + 1e-5)
    return quaternion.from_float_array(q)


def rotation_matrix_from_array(float_array):
    q = quaternion_from_array(float_array)
    R = quaternion.as_rotation_matrix(q)
    return R


def project(R, T, x):
    Rx = R @ x
    Rx_norm = np.linalg.norm(Rx, axis=0)
    return (np.dot(T, Rx) / (Rx_norm ** 2) + 1) * Rx


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.abs(np.arccos(cos))


def fun_with_precalculated_sift_reduce_rot(
    x0, numPlane, img1, xys1, normals1, img2, xys2, normals2, init_R, weight
):
    assert numPlane == len(xys1)
    assert numPlane == len(xys2)
    assert numPlane == len(normals1)
    assert numPlane == len(normals2)
    R = vec6dToSo3(x0[:6])
    T = np.array(x0[6:9])
    offsets1 = x0[9 : 9 + numPlane].reshape(-1, 1)
    offsets2 = x0[9 + numPlane : 9 + 2 * numPlane].reshape(-1, 1)
    planes1_suncg = offsets1 * normals1
    planes2_suncg = offsets2 * normals2
    planes1_habitat = (planes1_suncg * np.array([1.0, -1.0, -1.0])).T
    planes2_habitat = (planes2_suncg * np.array([1.0, -1.0, -1.0])).T

    err_plane = huber(
        weight["huber_delta"],
        np.linalg.norm(project(R, T, planes1_habitat) - planes2_habitat, axis=0),
    ).sum()
    err_pixel = get_pixel_error_precalculated_sift(
        img1, xys1, planes1_suncg, img2, xys2, planes2_suncg, R, T
    )
    change_R = angle_error_mat(R, init_R)
    err = err_plane + err_pixel + change_R * weight["lambda_R"]
    # err = huber(0.01, np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum() + lambda_R * huber(1, change_R) #+ huber(1, change_T)
    return err


if __name__ == "__main__":
    main()
