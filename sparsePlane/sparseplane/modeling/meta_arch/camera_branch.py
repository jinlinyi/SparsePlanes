import pickle
import json
import numpy as np
import os
from collections import namedtuple


class Camera_Branch:
    def __init__(self, d2_cfg, rpnet_args=None):
        self.cfg = d2_cfg
        if self.cfg.MODEL.CAMERA_BRANCH == "GT":
            return
        elif self.cfg.MODEL.CAMERA_BRANCH == "CACHED":
            assert "summary.pkl" in rpnet_args.camera_cached_file
            assert os.path.exists(rpnet_args.camera_cached_file)
            print(f"Loading camera from {rpnet_args.camera_cached_file}")
            cached_file = os.path.join(rpnet_args.camera_cached_file)
            with open(cached_file, "rb") as f:
                self.cached_data = pickle.load(f)
            if self.cfg.MODEL.CAMERA_ON:
                with open(self.cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH, "rb") as f:
                    self.kmeans_trans = pickle.load(f)
                with open(self.cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH, "rb") as f:
                    self.kmeans_rots = pickle.load(f)
            else:
                flags = self.get_flags(rpnet_args)
                with open(flags.kmeans_trans_path, "rb") as f:
                    self.kmeans_trans = pickle.load(f)
                with open(flags.kmeans_rots_path, "rb") as f:
                    self.kmeans_rots = pickle.load(f)
        else:
            raise NotImplementedError

    def get_flags(self, rpnet_args):
        assert os.path.exists(rpnet_args.camera_cached_file)
        _, _, phase = rpnet_args.camera_cached_file.split("/")[-2].split("_")
        if "val" in rpnet_args.dataset_phase:
            assert phase == "val"
        if "test" in rpnet_args.dataset_phase:
            assert phase == "test"
        rpnet_config_path = os.path.join(
            rpnet_args.camera_cached_file.rsplit("/", 2)[0], "README.txt"
        )
        with open(rpnet_config_path, "r") as f:
            lines = f.read().splitlines()[1:]
            flags = json.loads(
                "".join(lines),
                object_hook=lambda d: namedtuple("X", d.keys())(*d.values()),
            )
        flags = flags._replace(train_test_phase=phase)
        return flags

    def get_rel_camera(self, batched_inputs, tran_topk=0, rot_topk=0):
        if self.cfg.MODEL.CAMERA_BRANCH == "GT":
            return [x["rel_pose"] for x in batched_inputs]
        elif self.cfg.MODEL.CAMERA_BRANCH == "CACHED":
            camera_infos = self.cached_camera_pose_pred(
                batched_inputs, tran_topk, rot_topk
            )
        elif self.cfg.MODEL.CAMERA_BRANCH == "ONLINE":
            camera_infos = self.online_camera_pose_pred(batched_inputs)
        return camera_infos

    def get_gt_cam_topk(self, batched_inputs):
        gt_cam_topks = []
        for batched_input in batched_inputs:
            key = batched_input["0"]["file_name"] + batched_input["1"]["file_name"]
            idx = self.cached_data["keys"].index(key)
            gt_tran_cls = self.cached_data["gts"]["tran_cls"][idx][0]
            gt_rot_cls = self.cached_data["gts"]["rot_cls"][idx][0]
            gt_tran_topk = list(
                np.argsort(self.cached_data["logits_sms"]["tran"][idx])[::-1]
            ).index(gt_tran_cls)
            gt_rot_topk = list(
                np.argsort(self.cached_data["logits_sms"]["rot"][idx])[::-1]
            ).index(gt_rot_cls)
            gt_cam_topks.append(
                {"gt_tran_topk": gt_tran_topk, "gt_rot_topk": gt_rot_topk}
            )
        return gt_cam_topks

    def cached_camera_pose_pred(self, batched_inputs, tran_topk=0, rot_topk=0):
        assert self.cfg.MODEL.CAMERA_BRANCH == "CACHED"
        camera_infos = []
        sorted_idx_tran = np.argsort(self.cached_data["logits_sms"]["tran"], axis=1)[:, ::-1]
        sorted_idx_rot = np.argsort(self.cached_data["logits_sms"]["rot"], axis=1)[:, ::-1]
        for batched_input in batched_inputs:
            key = batched_input["0"]["file_name"] + batched_input["1"]["file_name"]
            idx = self.cached_data["keys"].index(key)
            if tran_topk == -1:
                # GT
                tran = self.cached_data["gts"]["tran"][idx]
                tran_p = 1
            else:
                tran = self.class2xyz(sorted_idx_tran[idx][tran_topk])
                tran_p = self.cached_data["logits_sms"]["tran"][idx][
                    sorted_idx_tran[idx][tran_topk]
                ]
            if rot_topk == -1:
                # GT
                rot = self.cached_data["gts"]["rot"][idx]
                rot_p = 1
            else:
                rot = self.class2quat(sorted_idx_rot[idx][rot_topk])
                rot_p = self.cached_data["logits_sms"]["rot"][idx][
                    sorted_idx_rot[idx][rot_topk]
                ]
            camera_info = {
                "position": tran,
                "position_prob": tran_p,
                "rotation": rot,
                "rotation_prob": rot_p,
            }
            camera_infos.append(camera_info)
        return camera_infos

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


if __name__ == "__main__":
    cb = Camera_Branch()
