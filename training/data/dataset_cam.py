import os
import random
import json
import torch

import torchvision.transforms as transforms
import numpy as np
import copy
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver
from safetensors.torch import load_file, save_file
from torchvision.transforms.functional import to_pil_image
import time


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



class RealEstate10KCameraControlnetDataset(Dataset):
    def __init__(
            self,
            video_root_dir,
            npz_path,
            stride=(2, 2),
            sample_n_frames=49,
            relative_pose=True,
            zero_t_first_frame=True,
            image_size=[480, 720],
            rescale_fxy=True,
            mode ="train"
    ):
        minimum_sample_stride, sample_stride = stride
        root_path = video_root_dir
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames

        self.dataset = np.load(npz_path, allow_pickle=True)['arr_0']
        self.length = len(self.dataset)
        sample_size = image_size
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        pixel_transforms = [transforms.Resize(sample_size),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.mode = mode


    def get_relative_pose_first_origin(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
       
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_video_reader_from_npz(self, idx):
        video_dict = self.dataset[idx]
        video_path = os.path.join(self.root_path, video_dict['video_path'])
        video_reader = VideoReader(video_path)

        video_path_parts = video_dict['video_path'].split('/')
        clip_name = f"{video_path_parts[-2]}_{video_path_parts[-1].replace('.mp4', '')}"

        prompt_files = f"RealEstate10K/prompt/train/{clip_name}.safetensors"

        return clip_name, video_reader,   video_dict['long_caption']


    def load_cameras_from_npz(self, idx):
        video_dict = self.dataset[idx]
        extrinsics = video_dict['camera_extrinsics']       # shape [T, 4, 4]
        intrinsics = video_dict['camera_intrinsics']       # shape [4,] -> [fx, fy, cx, cy]

        cam_params = [Camera(extrinsic, intrinsics) for extrinsic in extrinsics]
        return cam_params

    def get_batch(self, idx):

        clip_name, video_reader, prompt = self.load_video_reader_from_npz(idx)


        cam_params = self.load_cameras_from_npz(idx)
        total_frames = len(cam_params)

        current_sample_stride = self.sample_stride
        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        if self.mode == "val":
            start_frame_ind = 0 
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)


        frames_np = video_reader.get_batch(frame_indices).asnumpy()

        pixel_values = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.


        cam_params = [copy.deepcopy(cam_params[indice]) for indice in frame_indices]

        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:                                         # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]


        intrinsics_np = np.asarray(
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


        if self.relative_pose:
            c2w_poses = self.get_relative_pose_first_origin(cam_params)   # current
        else:
            c2w_poses = np.array(
                [cam_param.c2w_mat for cam_param in cam_params],
                dtype=np.float32,
            )


        intrinsics = torch.as_tensor(intrinsics_np)  # [n_frame, 4]
        c2w = torch.as_tensor(c2w_poses)             # [n_frame, 4, 4]


        return pixel_values, prompt, intrinsics, c2w, clip_name
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video, prompt, intrinsics, c2w, clip_name = self.get_batch(idx)

        for transform in self.pixel_transforms:
            video = transform(video)
        image = video[0]
        first_image = image.mul(0.5).add(0.5).clamp(0, 1)
        clip_image_pil = to_pil_image(first_image)
        data = {
            'prompts': prompt,
            'videos': video,  
            'intrinsics' : intrinsics,
            'c2w': c2w,
            'image': video[0],
            'clip_name': clip_name,
            'clip_image_pils': clip_image_pil,
        }
        return data
