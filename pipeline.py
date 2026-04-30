import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from models.wan_modules import (AutoencoderKLWan, AutoTokenizer,CLIPModel,
                      WanTransformer3DModel, WanT5EncoderModel)


logger = logging.get_logger(__name__)  


# following AC3D (CVPR 2025): https://github.com/snap-research/ac3d
CAMERA_CONTROLNET_SIGMA_LIN_START = 0.6


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
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
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def build_controlnet_token_mask(
    normalized_frame_ids_image: torch.Tensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a token-level mask for history controlnet states.

    Args:
        normalized_frame_ids_image: [B, F] long tensor. Each value is a frame id;
            values < 0 mark invalid frames.
        seq_len: token sequence length of history_controlnet_states.
        device: output mask device.
        dtype: output mask dtype.

    Returns:
        [1, seq_len, 1] float mask broadcastable to history_controlnet_states.
    """
    frame_ids_slice = normalized_frame_ids_image[0:1]
    valid_per_frame = frame_ids_slice >= 0

    latent_ids = torch.where(
        valid_per_frame,
        torch.where(
            frame_ids_slice == 0,
            frame_ids_slice,
            (frame_ids_slice - 1) // 4 + 1,
        ),
        torch.full_like(frame_ids_slice, -1),
    )

    frame_diff = latent_ids[:, 1:] - latent_ids[:, :-1]
    too_close_mask = frame_diff.abs() < 1
    too_close_mask = torch.cat(
        [
            torch.zeros(too_close_mask.shape[0], 1, dtype=torch.bool, device=too_close_mask.device),
            too_close_mask,
        ],
        dim=1,
    )

    valid_per_frame[:, 0] = False
    valid_per_frame = valid_per_frame & (~too_close_mask)

    tokens_per_frame = seq_len // valid_per_frame.shape[1]
    frame_mask = valid_per_frame.to(device=device, dtype=dtype)
    return frame_mask.repeat_interleave(tokens_per_frame, dim=1).unsqueeze(-1)


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class WanMemPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer:WanTransformer3DModel,
        clip_image_encoder: CLIPModel,
        controlnet,
        history_controlnet,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
             tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer,
              clip_image_encoder=clip_image_encoder, scheduler=scheduler, controlnet=controlnet,history_controlnet=history_controlnet,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            # mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            # masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents
    

    def prepare_history_chunk_latents(
        self,
        history_video,
        frame_ids_image,
        height,
        width,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.FloatTensor:
        # Inputs:
        #   history_video: [B, F, C, H, W] or [B, C, F, H, W]
        #   frame_ids_image: [B, N], each value is the center history-frame index
        # Output:
        #   chunk_latents: [B, N, C_lat, 1, H_lat, W_lat]
        if history_video is None or frame_ids_image is None:
            return None

        history_video = history_video if history_video.ndim == 5 else history_video.unsqueeze(0)
        if history_video.ndim != 5:
            raise ValueError(
                f"`history_video` must be 5D before chunk VAE encoding, got shape={tuple(history_video.shape)}"
            )
        if history_video.shape[1] == 3 and history_video.shape[2] != 3:
            history_video = history_video.permute(0, 2, 1, 3, 4).contiguous()

        history_video_proc = self.image_processor.preprocess(
            rearrange(history_video, "b f c h w -> (b f) c h w"),
            height=height,
            width=width,
        )
        batch_size, history_frames = history_video.shape[:2]
        history_video_proc = rearrange(history_video_proc, "(b f) c h w -> b f c h w", b=batch_size, f=history_frames)

        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio
        zero_latent = torch.zeros(
            self.vae.config.latent_channels,
            1,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )

        chunk_latents = []
        with torch.no_grad():
            for b_idx in range(batch_size):
                batch_latents = []
                for center_idx in frame_ids_image[b_idx].tolist():
                    center_idx = int(center_idx)
                    if center_idx < 0 or center_idx >= history_frames:
                        batch_latents.append(zero_latent.clone())
                        continue

                    start_idx = max(0, center_idx - 2)
                    end_idx = min(history_frames, center_idx + 3)
                    chunk_frames = history_video_proc[b_idx : b_idx + 1, start_idx:end_idx]
                    chunk_frames = rearrange(chunk_frames, "b f c h w -> b c f h w")
                    chunk_frames = chunk_frames.to(device=device, dtype=self.vae.dtype)

                    chunk_latent = self.vae.encode(chunk_frames)[0].mode()
                    batch_latents.append(chunk_latent[0, :, -1:].to(device=device, dtype=dtype))

                chunk_latents.append(torch.stack(batch_latents, dim=0))

        return torch.stack(chunk_latents, dim=0)


    def prepare_cond_image_latents(
        self,
        pixel_values_bfchw: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.FloatTensor:
        """Prepare cond_image latents: [B,F,C,H,W] -> [B,C_lat,F,h,w]."""
        if pixel_values_bfchw is None:
            return None
        b, f, c, h, w = pixel_values_bfchw.shape
        pv = rearrange(pixel_values_bfchw, "b f c h w -> (b f) c 1 h w")
        bs = 1
        out = []
        with torch.no_grad():
            pv = pv.to(device=device, dtype=self.vae.dtype)
            for i in range(0, pv.shape[0], bs):
                pv_bs = pv[i : i + bs]
                pv_bs = self.vae.encode(pv_bs)[0].mode()
                out.append(pv_bs)
        latents = torch.cat(out, dim=0)
        latents = rearrange(latents, "(b f) c 1 h w -> b c f h w", b=b, f=f)
        return latents.to(device=device, dtype=dtype)


    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames
    def decode_latents_wonorm(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        # frames = (frames / 2 + 0.5).clamp(0, 1)
        # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        # frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)


    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Optional[torch.FloatTensor] = None,
        mask_video: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        c2ws: Optional[torch.FloatTensor] = None,
        history_video: Optional[torch.FloatTensor] = None,
        cond_image: Optional[torch.FloatTensor] = None,
        frame_ids_image: Optional[Any] = None,
        controlnet_stride: int = 1,
        clip_image: Image = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        shift: int = 5,
        start_image: Optional[torch.FloatTensor] = None,
        timesteps: Optional[List[int]] = None,
        eta: float = 0.0,
        num_videos_per_prompt: int = 1,
        comfyui_progressbar: bool = False,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """
        if start_image is not None:
            raise ValueError("This simplified __call__ does not support start_image. Your test code passes None.")
        if timesteps is not None:
            raise ValueError("This simplified __call__ does not support custom timesteps.")
        if comfyui_progressbar:
            raise ValueError("This simplified __call__ removed comfyui_progressbar branch.")

        if video is None or mask_video is None:
            raise ValueError("This simplified __call__ requires video and mask_video (your test code always provides them).")
    
        # 1) Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        num_videos_per_prompt = 1

        # 2) batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            if prompt_embeds is None:
                raise ValueError("prompt and prompt_embeds cannot both be None.")
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        do_cfg = self._guidance_scale > 1.0

        # 3) encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        in_prompt_embeds = (negative_prompt_embeds + prompt_embeds) if do_cfg else prompt_embeds
        
        # 4) timesteps
        from diffusers import FlowMatchEulerDiscreteScheduler
        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError(
                "This simplified __call__ only supports FlowMatchEulerDiscreteScheduler (sampler_name=Flow in your script)."
            )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps=None,
            mu=1,
        )
        self._num_timesteps = len(timesteps)

        # 5) preprocess init_video
        video_length = video.shape[2]
        init_video = self.image_processor.preprocess(
            rearrange(video, "b c f h w -> (b f) c h w"),
            height=height,
            width=width,
        )
        init_video = init_video.to(dtype=torch.float32)
        init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        
        # 6) prepare latents
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )

        # 7) prepare mask latents
        bs, _, video_length, H, W = video.size()

        mask_condition = self.mask_processor.preprocess(
            rearrange(mask_video, "b c f h w -> (b f) c h w"),
            height=H,
            width=W,
        )
        mask_condition = mask_condition.to(dtype=torch.float32)
        mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

        masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)

        _, masked_video_latents = self.prepare_mask_latents(
            None,
            masked_video,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_cfg,
            noise_aug_strength=None,
        )

        # Align the mask tensor to the latent layout.
        mask_condition = torch.concat(
            [
                torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
                mask_condition[:, :, 1:],
            ],
            dim=2,
        )
        mask_condition = mask_condition.view(bs, mask_condition.shape[2] // 4, 4, H, W)
        mask_condition = mask_condition.transpose(1, 2)

        mask_latents = resize_mask(1 - mask_condition, masked_video_latents, True).to(device, weight_dtype)

        # Keep the first-frame mask aligned with the latent grid.
        mask = None
        if self.vae.spatial_compression_ratio >= 16:
            mask = F.interpolate(
                mask_condition[:, :1],
                size=latents.size()[-3:],
                mode="trilinear",
                align_corners=True,
            ).to(device, weight_dtype)

            # If the first frame has no editable region, keep it anchored.
            if not mask[:, :, 0, :, :].any():
                mask[:, :, 1:, :, :] = 1
                latents = (1 - mask) * masked_video_latents + mask * latents

        # 8) prepare camera control latents 


        def _as_bfchw(x: torch.Tensor) -> torch.Tensor:
            """
            Normalize video-like tensors to shape [B, F, C, H, W].
            Accepts [B, F, C, H, W] or [B, C, F, H, W] or [B, C, H, W] (single frame).
            """
            if x is None:
                return x
            if x.ndim == 4:
                # [B, C, H, W] -> [B, 1, C, H, W]
                return x.unsqueeze(1)
            if x.ndim != 5:
                raise ValueError(f"Expected 4D or 5D tensor for video/image input, got shape={tuple(x.shape)}")
            # Heuristic: if second dim is 3 and third dim is not 3, treat as [B, C, F, H, W]
            if x.shape[1] == 3 and x.shape[2] != 3:
                return x.permute(0, 2, 1, 3, 4).contiguous()
            return x

        normalized_frame_ids_image = None
        if frame_ids_image is not None:
            normalized_frame_ids_image = torch.as_tensor(frame_ids_image, device=device, dtype=torch.long)
            normalized_frame_ids_image = normalized_frame_ids_image.squeeze(-1)
            if normalized_frame_ids_image.ndim != 2:
                raise ValueError(
                    f"`frame_ids_image` must have shape [B, F] or [B, F, 1], got shape={tuple(normalized_frame_ids_image.shape)}"
                )

        cond_latents_chunks = None
        if history_video is not None and normalized_frame_ids_image is not None:
            cond_latents_chunks = self.prepare_history_chunk_latents(
                history_video=_as_bfchw(history_video),
                frame_ids_image=normalized_frame_ids_image,
                height=height,
                width=width,
                dtype=weight_dtype,
                device=device,
            )

        # cond_image (aka cond_images_latents in training, can be multi-frame) -> cond_images_latents latents
        cond_images_latents = None
        if cond_image is not None:
            cond_image = _as_bfchw(cond_image)
            cond_images_latents = self.prepare_cond_image_latents(
                cond_image,
                dtype=weight_dtype,
                device=device,
            )
        

        if clip_image is None:
            raise ValueError("`clip_image` is required for this pipeline.")
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype) 
        clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        clip_context_input = (
                    torch.cat([clip_context] * 2) if do_cfg else clip_context
                )

        # 9) extra kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio,
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])
            * target_shape[1]
        )

        # 10) denoise loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps

        # intrinsics, c2ws: normalize shape then move to correct device.
        # expected shapes:
        #   intrinsics: [B, V, 4] (fx, fy, cx, cy)
        #   c2ws:       [B, V, 4, 4]
        intrinsics = torch.cat([ intrinsics] * 2) if do_cfg else intrinsics 
        c2ws = torch.cat([ c2ws] * 2) if do_cfg else c2ws

        if normalized_frame_ids_image is not None and do_cfg:
            normalized_frame_ids_image = torch.cat([normalized_frame_ids_image] * 2, dim=0)
        if cond_latents_chunks is not None and do_cfg:
            cond_latents_chunks = torch.cat([cond_latents_chunks] * 2, dim=0)
        if cond_images_latents is not None and do_cfg:
            cond_images_latents = torch.cat([cond_images_latents] * 2, dim=0)


        mask_in = torch.cat([mask_latents] * 2) if do_cfg else mask_latents
        masked_latents_in = torch.cat([masked_video_latents] * 2) if do_cfg else masked_video_latents
        y = torch.cat([mask_in, masked_latents_in], dim=1).to(device, weight_dtype)

        def sigma_shift(sigma, shift: float):
            return shift * sigma / (1.0 + (shift - 1.0) * sigma)

        N = self.scheduler.config.num_train_timesteps
        shift = float(self.scheduler.config.shift)  # =5

        sigma_used_start = float(sigma_shift(torch.tensor(CAMERA_CONTROLNET_SIGMA_LIN_START), shift).item())  # ≈0.882
        t_boundary = sigma_used_start * N

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                self.transformer.current_steps = i
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep = t.expand(latent_model_input.shape[0])
                

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):

                    controlnet_states = None
                    history_controlnet_states = None
                    if t >= t_boundary:
                        controlnet_states = self.controlnet(
                            hidden_states=latent_model_input,
                            y=y,
                            encoder_hidden_states=in_prompt_embeds,
                            intrinsics=intrinsics[:,-num_frames:],
                            c2ws=c2ws[:,-num_frames:],
                            timestep=timestep,
                            frame_num=num_frames,
                            return_dict=False,
                        )[0]
                        

                        if isinstance(controlnet_states, (tuple, list)):
                            controlnet_states = [x.to(dtype=weight_dtype) for x in controlnet_states]
                        else:
                            controlnet_states = controlnet_states.to(dtype=weight_dtype)

                    if (
                        self.history_controlnet is not None
                        and (cond_latents_chunks is not None)
                        and (cond_images_latents is not None)
                        and (intrinsics is not None)
                        and (c2ws is not None)
                        and (normalized_frame_ids_image is not None)
                    ):
                        history_controlnet_states = self.history_controlnet(
                            hidden_states=latent_model_input,
                            y=y,
                            timestep=timestep,
                            frame_num=num_frames,
                            cond_latents_chunks=cond_latents_chunks,
                            cond_images=cond_images_latents,
                            intrinsics=intrinsics,
                            c2ws=c2ws,
                            frame_ids_image=normalized_frame_ids_image,
                            return_dict=False,
                        )[0]

                        if isinstance(history_controlnet_states, (tuple, list)):
                            _, seq_len, _ = history_controlnet_states[0].shape    
                            token_mask_device = history_controlnet_states[0].device
                            token_mask_dtype = history_controlnet_states[0].dtype
                        else:
                            _, seq_len, _ = history_controlnet_states.shape
                            token_mask_device = history_controlnet_states.device
                            token_mask_dtype = history_controlnet_states.dtype

                        token_mask = build_controlnet_token_mask(
                            normalized_frame_ids_image=normalized_frame_ids_image,
                            seq_len=seq_len,
                            device=token_mask_device,
                            dtype=token_mask_dtype,
                        )
    
                    
                        if isinstance(history_controlnet_states, (tuple, list)):
                            history_controlnet_states = [
                                x * token_mask for x in history_controlnet_states
                            ]
                        else:
                           history_controlnet_states = history_controlnet_states * token_mask

                        if isinstance(history_controlnet_states, (tuple, list)):
                            history_controlnet_states = [x.to(dtype=weight_dtype) for x in history_controlnet_states]
                        else:
                            history_controlnet_states = history_controlnet_states.to(dtype=weight_dtype)
                   
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=y,
                        clip_fea=clip_context_input,
                        controlnet_states=controlnet_states,
                        controlnet_stride=controlnet_stride,
                        history_controlnet_states=history_controlnet_states,
                    )
                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 11) decode
        if output_type == "latent":
            video_out = latents
        else:
            if output_type != "numpy":
                video_out = self.decode_latents_wonorm(latents)
                video_out = self.video_processor.postprocess_video(video=video_out, output_type=output_type)
            else:
                video_out = self.decode_latents(latents)


        # Offload all models
        self.maybe_free_model_hooks()

        if (output_type == "numpy") and (not return_dict):
            video_out = torch.from_numpy(video_out)

        return WanPipelineOutput(videos=video_out)
