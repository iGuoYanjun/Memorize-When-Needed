"""Tensor and memory helpers shared by training and validation."""

import gc

import torch
import torch.nn.functional as F


def free_memory() -> None:
    """Release cached CUDA allocations between training and validation."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def resize_mask(mask, latent_size):
    """Resize the first latent frame separately from the remaining frames."""
    batch_size, channels, num_frames, height, width = mask.shape


    target_size = list(latent_size[2:])
    target_size[0] = 1
    first_frame_resized = F.interpolate(
        mask[:, :, 0:1, :, :],
        size=target_size,
        mode='trilinear',
        align_corners=False
    )

    target_size = list(latent_size[2:])
    target_size[0] = target_size[0] - 1
    if target_size[0] != 0:
        remaining_frames_resized = F.interpolate(
            mask[:, :, 1:, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
    else:
        resized_mask = first_frame_resized

    return resized_mask


def sigma_shift(sigma, shift: float):
    # sigma: tensor in (0,1]
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def sample_high_noise_timesteps_keep_linear_time_dist(
    noise_scheduler,
    bsz: int,
    mean: float,
    std: float,
    a: float,
    b: float,
    device,
    dtype=torch.float32,
    n_dim=4,
):
    """Sample high-noise sigmas and map their shifted values to the scheduler grid."""
    sigma_lin = torch.nn.init.trunc_normal_(
        torch.empty(bsz, device=device),
        mean=mean, std=std, a=a, b=b
    ).clamp(min=1e-6, max=1.0)


    shift = float(noise_scheduler.config.shift)
    sigma_used = sigma_shift(sigma_lin, shift=shift).to(dtype=dtype)


    sched_sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)      # sigma' grid
    sched_timesteps = noise_scheduler.timesteps.to(device=device)                        # = sigma' * N

    dist = (sched_sigmas[None, :] - sigma_used[:, None]).abs()
    idx = dist.argmin(dim=1)

    timesteps = sched_timesteps[idx]  # [bsz]
    sigmas = sched_sigmas[idx]        # [bsz]  (sigma' used in training)
    while len(sigmas.shape) < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return timesteps, sigmas
