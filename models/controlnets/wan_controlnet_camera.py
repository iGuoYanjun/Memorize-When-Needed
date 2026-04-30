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
    # WanTransformerBlock
)
from .attn_models.CamBlock import CamBlock
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from einops import rearrange
from packaging import version as pver
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
        text_embed_dim: int,

        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class WanControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
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
    _no_split_modules = ["CamBlock"]
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
        start_channels = in_channels * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]

        start_channels = in_channels * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]
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

        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(hidden_channels + input_channels[2], inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )
        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                CamBlock(
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
        encoder_hidden_states: torch.Tensor,
        frame_num,
        intrinsics=None,
        c2ws=None,
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
        device = hidden_states.device
        dtype = hidden_states.dtype
        # 0. Controlnet encoder
        controlnet_states = ray_condition(intrinsics[:,-frame_num:], 
                                c2ws[:,-frame_num:], 
                                hidden_states.shape[3]*self.downscale_coef,
                                hidden_states.shape[4]*self.downscale_coef).to(dtype)
        batch_size, num_frames, channels, height, width = controlnet_states.shape

        controlnet_states = rearrange(controlnet_states, 'b f c h w -> (b f) c h w')
        controlnet_states = self.unshuffle(controlnet_states)
        controlnet_states = self.controlnet_encode_first(controlnet_states)
        controlnet_states = self.compress_time(controlnet_states, num_frames=num_frames) 
        num_frames = controlnet_states.shape[0] // batch_size

        controlnet_states = self.controlnet_encode_second(controlnet_states)
        controlnet_states = self.compress_time(controlnet_states, num_frames=num_frames) 
        controlnet_states = rearrange(controlnet_states, '(b f) c h w -> b c f h w', b=batch_size)



        # print("+" * 50, hidden_states.shape, controlnet_states.shape)
        hidden_states = torch.cat([hidden_states, controlnet_states], dim=1)
        
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        max_sequence_length = 512
        encoder_hidden_states = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in encoder_hidden_states], dim=0
        )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states,  timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        
        # 4. Transformer blocks
        controlnet_hidden_states = ()
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):

                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                controlnet_hidden_states += (controlnet_block(hidden_states),)
        else:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                controlnet_hidden_states += (controlnet_block(hidden_states),)


        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_hidden_states,)

        return Transformer2DModelOutput(sample=controlnet_hidden_states)


if __name__ == "__main__":
    parameters = {
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "image_dim": None,
        "in_channels": 3,
        "num_attention_heads": 12,
        "num_layers": 2,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
        "downscale_coef": 8,
        "out_proj_dim": 12 * 128,
        "vae_channels": 16
    }
    controlnet = WanControlnet(**parameters)

    hidden_states = torch.rand(1, 16, 13, 60, 90)
    timestep = torch.tensor([1000]).repeat(17550).unsqueeze(0) #torch.randint(low=0, high=1000, size=(1,), dtype=torch.long)
    encoder_hidden_states = torch.rand(1, 512, 4096)
    controlnet_states = torch.rand(1, 3, 49, 480, 720)

    controlnet_hidden_states = controlnet(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_states=controlnet_states,
        return_dict=False
    )
    print("Output states count", len(controlnet_hidden_states[0]))
    for out_hidden_states in controlnet_hidden_states[0]:
        print(out_hidden_states.shape)
    