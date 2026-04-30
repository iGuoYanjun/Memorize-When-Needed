from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers,deprecate
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_wan import (
    WanRotaryPosEmbed, 
   
)
from .attn_models.WanTransformerBlock_ca import WanTransformerBlock_ca
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from einops import rearrange
from packaging import version as pver

# `ray_condition()` is adapted from CameraCtrl:
# https://github.com/hehao13/CameraCtrl
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H, W):
    device = c2w.device
    dtype = c2w.dtype
    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)

    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, V, H, W, 6)
    plucker = plucker.permute(0, 1, 4, 2, 3)

    return plucker

class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states : torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb, timestep_proj
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class WanRotaryPosEmbed_his(WanRotaryPosEmbed):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        num_frames = 2*num_frames####only change this line
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


class WanHistoryControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A Controlnet Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the controlnet input.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        downscale_coef (`int`, *optional*, defaults to `8`):
            Coeficient for downscale controlnet input video.
        out_proj_dim (`int`, *optional*, defaults to `128 * 12`):
            Output projection dimention for last linear layers.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock_ca"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    
    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 12,
        attention_head_dim: int = 128,
        in_channels: int = 6,
        hidden_channels: int = 36,
        vae_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 20,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        downscale_coef: int = 8,
        out_proj_dim: int = 128 * 40,
    ) -> None:
        super().__init__()
        self.downscale_coef = downscale_coef
        inner_dim = num_attention_heads * attention_head_dim

        start_channels = in_channels * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4, vae_channels ]
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)
        
        self.controlnet_encode_first = nn.Sequential(
            nn.Conv2d(input_channels[0], input_channels[1], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[1]),
            nn.ReLU(),
        )

        self.controlnet_encode_second = nn.Sequential(
            nn.Conv2d(input_channels[1], input_channels[2], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[2]),
            nn.ReLU(),
        )

        self.controlnet_encode_third = nn.Sequential(
            nn.Conv2d(input_channels[2], input_channels[3], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[3]),
            nn.ReLU(),
        )

        
        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.rope_his = WanRotaryPosEmbed_his(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_current = nn.Conv3d(hidden_channels + vae_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_history_video = nn.Conv3d(vae_channels + vae_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_history_image = nn.Conv3d(vae_channels + vae_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
        )
        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock_ca(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4 Controlnet modules
        self.controlnet_blocks = nn.ModuleList([])

        for _ in range(len(self.blocks)):
            controlnet_block = nn.Linear(inner_dim, out_proj_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
            
        self.gradient_checkpointing = False
    def compress_time(self, x, num_frames):
            x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
            batch_size, frames, channels, height, width = x.shape
            x = rearrange(x, 'b f c h w -> (b h w) c f')
            
            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
            else:
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
            x = rearrange(x, '(b h w) c f -> (b f) c h w', b=batch_size, h=height, w=width)
            return x
    def forward(
        self,
        hidden_states: torch.Tensor,
        y: torch.Tensor,
        timestep: torch.LongTensor,
        frame_num,
        cond_latents_chunks=None,
        cond_images=None,
        intrinsics=None,
        c2ws=None,
        frame_ids_image=None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        if y is not None:
            hidden_states = torch.cat([hidden_states, y], dim=1)#torch.Size([B, 16+4+16, F, 60, 104])
        rotary_emb = self.rope(hidden_states)
        rotary_emb_his = self.rope_his(hidden_states)
        device = hidden_states.device
        dtype = hidden_states.dtype
        # 0. Controlnet encoder

        camera_states = ray_condition(intrinsics[:,(-frame_num):], 
                                c2ws[:,(-frame_num):], 
                                hidden_states.shape[3]*self.downscale_coef,
                                hidden_states.shape[4]*self.downscale_coef).to(dtype)

        batch_size, num_frames, channels, height, width = camera_states.shape

        camera_states = rearrange(camera_states, 'b f c h w -> (b f) c h w')
        camera_states = self.unshuffle(camera_states)
        camera_states = self.controlnet_encode_first(camera_states)
        camera_states = self.compress_time(camera_states, num_frames=num_frames) 
        num_frames = camera_states.shape[0] // batch_size

        camera_states = self.controlnet_encode_second(camera_states)
        camera_states = self.compress_time(camera_states, num_frames=num_frames)

        camera_states = self.controlnet_encode_third(camera_states)  
        camera_states = rearrange(camera_states, '(b f) c h w -> b c f h w', b=batch_size)

        camera_states_history = ray_condition(intrinsics[:,:(-frame_num)], 
                        c2ws[:,:(-frame_num)], 
                        hidden_states.shape[3]*self.downscale_coef,
                        hidden_states.shape[4]*self.downscale_coef).to(dtype)
        batch_size, num_frames_h, channels_h, height_h, width_h = camera_states_history.shape

        camera_states_history = rearrange(camera_states_history, 'b f c h w -> (b f) c h w')

        camera_states_history = self.unshuffle(camera_states_history)
        camera_states_history = self.controlnet_encode_first(camera_states_history)

        camera_states_history = self.compress_time(camera_states_history, num_frames=num_frames_h)
        num_frames_h = camera_states_history.shape[0] // batch_size

        camera_states_history = self.controlnet_encode_second(camera_states_history)
        camera_states_history = self.compress_time(camera_states_history, num_frames=num_frames_h)

        camera_states_history= self.controlnet_encode_third(camera_states_history) 
        camera_states_history = rearrange(camera_states_history, '(b f) c h w -> b c f h w', b=batch_size)
    
        # `cond_latents_chunks` is precomputed in the pipeline:
        # [B, N, C_lat, 1, h, w] -> [B, C_lat, N, h, w]
        if cond_latents_chunks is None:
            raise ValueError("`cond_latents_chunks` is required for WanHistoryControlnet.")
        cond_latents_chunks = cond_latents_chunks.squeeze(3).permute(0, 2, 1, 3, 4).contiguous()

        if frame_ids_image is None:
            raise ValueError("`frame_ids_image` is required for WanHistoryControlnet.")
        frame_ids_image = frame_ids_image.to(device=device, dtype=torch.long)
        if frame_ids_image.ndim != 2:
            raise ValueError(
                f"`frame_ids_image` must have shape [B, F], got shape={tuple(frame_ids_image.shape)}"
            )

        valid_frame_mask = frame_ids_image >= 0
        frame_ids_history = torch.where(
            valid_frame_mask,
            torch.where(frame_ids_image == 0, frame_ids_image, (frame_ids_image - 1) // 4 + 1),
            torch.zeros_like(frame_ids_image),
        ) 
        frame_ids_history = frame_ids_history.clamp(min=0, max=camera_states_history.shape[2] - 1)

        index = frame_ids_history[:, None, :, None, None]
        b, c, _, h, w = camera_states_history.shape
        index = index.expand(b, c, frame_ids_history.size(1), h, w).long()
        camera_states_history = torch.gather(camera_states_history, dim=2, index=index)  # [B, 16, N, h, w]

        history_video_latents = torch.cat([cond_latents_chunks, camera_states_history], dim=1)
        history_image_latents =  torch.cat([cond_images , camera_states], dim=1) #b  2*16 13 h w channel cat
        hidden_states  =  torch.cat([hidden_states , camera_states], dim=1)



        hidden_states = self.patch_embedding_current(hidden_states)
        history_image_latents = self.patch_embedding_history_image (history_image_latents)
        history_video_latents = self.patch_embedding_history_video (history_video_latents)
        history_states = torch.cat([history_video_latents ,  history_image_latents], dim=2)#frame cat Size([1, 1536, 26, 30, 52])

        history_states =  history_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)


        temb, timestep_proj= self.condition_embedder(
            timestep, hidden_states,  timestep_seq_len= None)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        num_latent_frames = history_video_latents.shape[2]
        history_video_block_indices = torch.arange(num_latent_frames, device=history_states.device)
        history_image_block_indices = history_video_block_indices + num_latent_frames
        history_block_indices = torch.stack(
            [history_video_block_indices, history_image_block_indices], dim=-1
        )
        history_block_indices = history_block_indices[None].expand(batch_size, -1, -1)
        # 4. Transformer blocks
    
        controlnet_hidden_states = ()
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):

                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    history_states,
                    timestep_proj,
                    rotary_emb,
                    rotary_emb_his,
                    history_block_indices,
                )
                controlnet_hidden_states += (controlnet_block(hidden_states),)
        else:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):
                hidden_states = block(
                    hidden_states,
                    history_states,
                    timestep_proj,
                    rotary_emb,
                    rotary_emb_his,
                    history_block_indices,
                )
                controlnet_hidden_states += (controlnet_block(hidden_states),)


        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_hidden_states,)

        return Transformer2DModelOutput(sample=controlnet_hidden_states)

