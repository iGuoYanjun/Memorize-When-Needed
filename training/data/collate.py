"""Batch collation for the camera-control training dataset."""

import torch


def collate_fn(examples):
    """Stack tensor fields while keeping prompts and PIL images as lists."""
    videos = [ex["videos"] for ex in examples]
    prompts = [ex["prompts"] for ex in examples]

    clip_names = [ex["clip_name"] for ex in examples]
    images = [ex["image"] for ex in examples]
    clip_image_pils = [ex["clip_image_pils"] for ex in examples]


    intrinsics = torch.stack([ex["intrinsics"] for ex in examples], dim=0)  # [B, n_frame, 4]
    c2w = torch.stack([ex["c2w"] for ex in examples], dim=0)                # [B, n_frame, 4, 4]

    videos = torch.stack(videos).to(memory_format=torch.contiguous_format)
    images = torch.stack(images).to(memory_format=torch.contiguous_format)

    intrinsics = intrinsics.to(memory_format=torch.contiguous_format)
    c2w = c2w.to(memory_format=torch.contiguous_format)

    return {
        "videos": videos,
        "prompts": prompts,
        "clip_names": clip_names,
        "images": images,
        "clip_image_pils": clip_image_pils,
        "intrinsics": intrinsics,
        "c2w": c2w,
    }
