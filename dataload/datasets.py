import copy
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

from dataload.utils import compute_rel_poses_letent, generate_condition_indices, calculate_half_fov


class Camera:
    def __init__(self, extrinsic_matrix, intrinsic_array):
        # The released pose txt keeps the original NPZ layout:
        # line 1 is normalized intrinsics [fx, fy, cx, cy],
        # remaining lines are flattened 4x4 world-to-camera matrices.
        self.w2c_mat = np.array(extrinsic_matrix, dtype=np.float32)
        self.c2w_mat = np.linalg.inv(self.w2c_mat)

        self.fx = intrinsic_array[0]
        self.fy = intrinsic_array[1]
        self.cx = intrinsic_array[2]
        self.cy = intrinsic_array[3]


class MemDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        sample_n_frames=49,
        zero_t_first_frame=True,
        image_size=(480, 720),
    ):
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        if not isinstance(self.dataset, list):
            raise ValueError(
                f"{self.metadata_path} must contain a JSON list of samples, got {type(self.dataset).__name__}."
            )

        self.zero_t_first_frame = zero_t_first_frame
        self.sample_n_frames = sample_n_frames
        self.sample_size = tuple(image_size) if not isinstance(image_size, int) else (image_size, image_size)
        self.sample_wh_ratio = self.sample_size[1] / self.sample_size[0]

    def __len__(self):
        return len(self.dataset)

    def _resolve_path(self, value):
        path = Path(value)
        if not path.is_absolute():
            path = self.metadata_path.parent / path
        return path.resolve()

    def _load_pose_txt(self, pose_path):
        lines = [line.strip() for line in pose_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError(f"Pose file must contain 1 intrinsic line and at least 1 extrinsic line: {pose_path}")

        intrinsics = np.fromstring(lines[0].replace(",", " "), sep=" ", dtype=np.float32)
        if intrinsics.size != 4:
            raise ValueError(
                f"Expected 4 intrinsic values [fx, fy, cx, cy] in the first line of {pose_path}, got {intrinsics.size}."
            )

        extrinsics = []
        for line_idx, line in enumerate(lines[1:], start=2):
            values = np.fromstring(line.replace(",", " "), sep=" ", dtype=np.float32)
            if values.size != 16:
                raise ValueError(
                    f"Expected 16 values for a flattened 4x4 extrinsic matrix at line {line_idx} in {pose_path}, got {values.size}."
                )
            extrinsics.append(values.reshape(4, 4))

        return [Camera(extrinsic, intrinsics) for extrinsic in extrinsics]

    def _load_item(self, idx):
        sample = self.dataset[idx]
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {idx} in {self.metadata_path} must be a JSON object.")

        missing_keys = [key for key in ("img", "prompt", "pose") if key not in sample]
        if missing_keys:
            raise ValueError(f"Sample {idx} is missing required keys: {missing_keys}")

        image_path = self._resolve_path(sample["img"])
        pose_path = self._resolve_path(sample["pose"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        image = read_image(str(image_path))
        prompt = sample["prompt"]
        clip_name = image_path.stem
        cam_params = self._load_pose_txt(pose_path)

        return clip_name, image, prompt, cam_params

    def get_relative_pose_first_origin(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]

        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -cam_to_origin],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        return np.array(ret_poses, dtype=np.float32)

    def get_relative_pose_last_origin(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

        source_cam_c2w = abs_c2ws[-1]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])

        target_cam_c2w = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -cam_to_origin],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        abs2rel = target_cam_c2w @ abs_w2cs[-1]
        ret_poses = [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[:-1]] + [target_cam_c2w]
        return np.array(ret_poses, dtype=np.float32)

    def get_batch(self, idx):
        clip_name, image, prompt, cam_params = self._load_item(idx)

        if len(cam_params) < self.sample_n_frames:
            raise ValueError(
                f"Sample '{clip_name}' only has {len(cam_params)} camera poses, but sample_n_frames={self.sample_n_frames}."
            )

        cam_params = [copy.deepcopy(cam_params[i]) for i in range(len(cam_params))]

        ori_h, ori_w = image.shape[-2:]
        ori_wh_ratio = ori_w / ori_h
        if ori_wh_ratio > self.sample_wh_ratio:
            resized_ori_w = self.sample_size[0] * ori_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
        else:
            resized_ori_h = self.sample_size[1] / ori_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]

        intrinsics = np.asarray(
            [
                [
                    cam_param.fx * self.sample_size[1],
                    cam_param.fy * self.sample_size[0],
                    cam_param.cx * self.sample_size[1],
                    cam_param.cy * self.sample_size[0],
                ]
                for cam_param in cam_params
            ],
            dtype=np.float32,
        )
        pose_conditions = np.asarray([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)

        total_frames = len(pose_conditions)
        window_size = self.sample_n_frames

        all_intrinsics = []
        all_c2w = []
        all_frame_ids_image = []
        start_idx = 0

        while start_idx + window_size <= total_frames:
            end_idx = start_idx + window_size

            if start_idx == 0:
                frame_ids_image = None
                c2w_poses = self.get_relative_pose_first_origin(cam_params[start_idx:end_idx])
                intrinsics_tensor = torch.as_tensor(intrinsics[:end_idx])
                c2w = torch.as_tensor(c2w_poses)
            else:
                current_window = pose_conditions[start_idx:end_idx]
                pose_conditions_current = torch.from_numpy(compute_rel_poses_letent(current_window))

                history_window = pose_conditions[: start_idx + 1]
                pose_conditions_history_image = torch.from_numpy(history_window)
                pose_conditions_com_image = torch.cat([pose_conditions_history_image, pose_conditions_current]).unsqueeze(1)

                fov_x_deg, fov_y_deg = calculate_half_fov(cam_params[0].fx, cam_params[0].fy)

                frame_ids_image = generate_condition_indices(
                    pose_conditions_history_image.shape[0],
                    1,
                    pose_conditions_com_image,
                    fov_x_deg,
                    fov_y_deg,
                )
                frame_ids_image = torch.stack(frame_ids_image, dim=0)

                c2w_poses_0 = self.get_relative_pose_last_origin(cam_params[: start_idx + 1])
                c2w_poses_1 = self.get_relative_pose_first_origin(cam_params[start_idx:end_idx])
                c2w_poses = np.concatenate([c2w_poses_0, c2w_poses_1], axis=0)
                intrinsics_tensor = torch.as_tensor(intrinsics[: end_idx + 1])
                c2w = torch.as_tensor(c2w_poses)

            all_frame_ids_image.append(frame_ids_image)
            all_intrinsics.append(intrinsics_tensor)
            all_c2w.append(c2w)

            start_idx = end_idx - 1

        return image, prompt, all_intrinsics, all_c2w, clip_name, all_frame_ids_image

    def __getitem__(self, idx):
        image, prompt, intrinsics, c2w, clip_name, frame_ids_image = self.get_batch(idx)
        image = F.resize(image, self.sample_size)

        return {
            "prompt": prompt,
            "caption": prompt,
            "image": image,
            "intrinsics": intrinsics,
            "c2w": c2w,
            "clip_name": clip_name,
            "frame_ids_image": frame_ids_image,
        }
