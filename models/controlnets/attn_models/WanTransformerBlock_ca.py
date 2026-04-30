# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def sdpa_with_block_gather_val(
    query,  # [B, H, 13*HW, D]
    key,    # [B, H, 50*HW, D]
    value,  # [B, H, 50*HW, D]
    sel_k_blocks,  # [B, 13, k] —— 来自 build_topk_block_index
    height: int,
    width: int,
    enable_flash: bool = True,
):  
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    B, Hh, Qtot, D = query.shape
    _, _, Ktot, _ = key.shape
    HW = height * width
    Tq_blk = sel_k_blocks.shape[1]   # 13
    k_per_q = sel_k_blocks.shape[-1] # 3
    Tk_blk = Ktot // HW              # 50

    # 形状检查
    assert Qtot == Tq_blk * HW, f"Qtot={Qtot}, but Tq_blk*HW={Tq_blk*HW}"
    assert Ktot == Tk_blk * HW, f"Ktot={Ktot}, but Tk_blk*HW={Tk_blk*HW}"

    q_blk = query.view(B, Hh, Tq_blk, HW, D).contiguous()   # [B,H,13,HW,D]
    k_blk = key.view(B, Hh, Tk_blk, HW, D).contiguous()     # [B,H,50,HW,D]
    v_blk = value.view(B, Hh, Tk_blk, HW, D).contiguous()   # [B,H,50,HW,D]

    # 1) 先在 k_blk/v_blk 上扩出“查询块维”，形成 [B,H,13,Tk,HW,D]
    k_blk_exp = k_blk.unsqueeze(2).expand(B, Hh, Tq_blk, Tk_blk, HW, D)
    v_blk_exp = v_blk.unsqueeze(2).expand(B, Hh, Tq_blk, Tk_blk, HW, D)

    # 2) 构造块索引，并用 take_along_dim 沿 dim=3（块维）选取
    # 2) 构造 mask：-1 表示无效
    valid_mask = (sel_k_blocks >= 0)  # [B, 13, k]

    # 将 -1 替换为 0（任意有效索引），后续用 mask 屏蔽
    idx_blocks = sel_k_blocks.clamp(min=0)  # [B, 13, k]
    idx_blocks = idx_blocks[:, None, :, :, None, None]                # [B,1,13,k,1,1]
    idx_blocks = idx_blocks.expand(B, Hh, Tq_blk, k_per_q, HW, D)       # [B,H,13,k,HW,D]
    idx_blocks = idx_blocks.to(dtype=torch.long)

    k_sel = torch.take_along_dim(k_blk_exp, idx_blocks, dim=3)          # [B,H,13,k,HW,D]
    v_sel = torch.take_along_dim(v_blk_exp, idx_blocks, dim=3)          # [B,H,13,k,HW,D]

    # 3) 把选出来的 k 个块拼成紧凑 K'/V'（每个查询块的 K' = k*HW）
    k_small = k_sel.reshape(B, Hh, Tq_blk, k_per_q * HW, D).contiguous() # [B,H,13,k*HW,D]
    v_small = v_sel.reshape(B, Hh, Tq_blk, k_per_q * HW, D).contiguous() # 同上
    q_small = q_blk      #(B, Hh, Tq_blk, HW, D)                                               # [B,H,13,HW,D]

    # 5) 构造 attention mask: [B, Tq_blk, 1, HW, k*HW]
    # valid_mask: [B, 13, k] -> [B, 13, k, HW] -> [B, 13, k*HW]
    attn_mask = valid_mask[:, :, :, None].expand(B, Tq_blk, k_per_q, HW)
    attn_mask = attn_mask.reshape(B, Tq_blk, k_per_q * HW)  # [B, 13, k*HW]
    attn_mask = attn_mask[:, :, None, None, :]  # [B, 13, 1, 1, k*HW]
    attn_mask = attn_mask.expand(B, Tq_blk, 1, HW, k_per_q * HW)  # [B, 13, 1, HW, k*HW]
    # 4) 合批跑无 mask 的 SDPA
    B2 = B * Tq_blk

    q_small = q_small.permute(0, 2, 1, 3, 4)
    k_small = k_small.permute(0, 2, 1, 3, 4)
    v_small = v_small.permute(0, 2, 1, 3, 4)

    q_ = q_small.reshape(B2, Hh, HW, D).contiguous()
    k_ = k_small.reshape(B2, Hh, k_per_q * HW, D).contiguous()
    v_ = v_small.reshape(B2, Hh, k_per_q * HW, D).contiguous()

    attn_mask = attn_mask.reshape(B2, 1, HW, k_per_q * HW)

    # with torch.backends.cuda.sdp_kernel(enable_flash=enable_flash,
    #                                     enable_math=not enable_flash,
    #                                     enable_mem_efficient=False):


    out_ = F.scaled_dot_product_attention(q_, k_, v_,attn_mask=attn_mask,dropout_p=0.0, is_causal=False)  # [B2, Hh, HW, D]

    out = out_.reshape(B, Tq_blk, Hh, HW, D).permute(0, 2, 1, 3, 4).reshape(B, Hh, Tq_blk * HW, D)                # [B,H,13*HW,D]
    return out.permute(0, 2, 1, 3) #[B, 13*HW,,H,D]
def sdpa_with_block_gather(
    query,  # [B, H, 13*HW, D]
    key,    # [B, H, 50*HW, D]
    value,  # [B, H, 50*HW, D]
    sel_k_blocks,  # [B, 13, k]，其中可能有 -1
    height: int,
    width: int,
    enable_flash: bool = True,
):
    ##dropout###
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    B, Hh, Qtot, D = query.shape
    _, _, Ktot, _ = key.shape
    HW = height * width
    Tq_blk = sel_k_blocks.shape[1]
    k_per_q = sel_k_blocks.shape[-1]
    Tk_blk = Ktot // HW

    assert Qtot == Tq_blk * HW, f"Qtot={Qtot}, but Tq_blk*HW={Tq_blk*HW}"
    assert Ktot == Tk_blk * HW, f"Ktot={Ktot}, but Tk_blk*HW={Tk_blk*HW}"

    q_blk = query.view(B, Hh, Tq_blk, HW, D).contiguous()   # [B,H,Tq,HW,D]
    k_blk = key.view(B, Hh, Tk_blk, HW, D).contiguous()     # [B,H,Tk,HW,D]
    v_blk = value.view(B, Hh, Tk_blk, HW, D).contiguous()   # [B,H,Tk,HW,D]

    k_blk_exp = k_blk.unsqueeze(2).expand(B, Hh, Tq_blk, Tk_blk, HW, D)
    v_blk_exp = v_blk.unsqueeze(2).expand(B, Hh, Tq_blk, Tk_blk, HW, D)

    # 关键点 1：记录哪些位置是 -1
    invalid = (sel_k_blocks < 0)  # [B,Tq,k] bool

    # 关键点 2：gather 前把 -1 clamp 掉，避免 take_along_dim 取错
    sel_safe = sel_k_blocks.clamp(min=0).to(dtype=torch.long)  # [B,Tq,k]

    idx_blocks = sel_safe[:, None, :, :, None, None]                # [B,1,Tq,k,1,1]
    idx_blocks = idx_blocks.expand(B, Hh, Tq_blk, k_per_q, HW, D)   # [B,H,Tq,k,HW,D]

    k_sel = torch.take_along_dim(k_blk_exp, idx_blocks, dim=3)      # [B,H,Tq,k,HW,D]
    v_sel = torch.take_along_dim(v_blk_exp, idx_blocks, dim=3)      # [B,H,Tq,k,HW,D]

    # 关键点 3：把 invalid 的 K 替换成对应的 Q
    # 对应的 Q 是同一 (B,H,Tq,HW,D)，扩展到 k 维
    q_rep = q_blk.unsqueeze(3).expand(B, Hh, Tq_blk, k_per_q, HW, D)  # [B,H,Tq,k,HW,D]
    invalid_exp = invalid[:, None, :, :, None, None].expand_as(k_sel) # [B,H,Tq,k,HW,D]
    k_sel = torch.where(invalid_exp, q_rep, k_sel)

    # 如果你也希望 V 在 invalid 时一致（更稳定），可以打开这一行
    v_sel = torch.where(invalid_exp, q_rep, v_sel)

    k_small = k_sel.reshape(B, Hh, Tq_blk, k_per_q * HW, D).contiguous()
    v_small = v_sel.reshape(B, Hh, Tq_blk, k_per_q * HW, D).contiguous()
    q_small = q_blk

    B2 = B * Tq_blk
    q_small = q_small.permute(0, 2, 1, 3, 4)
    k_small = k_small.permute(0, 2, 1, 3, 4)
    v_small = v_small.permute(0, 2, 1, 3, 4)

    q_ = q_small.reshape(B2, Hh, HW, D).contiguous()
    k_ = k_small.reshape(B2, Hh, k_per_q * HW, D).contiguous()
    v_ = v_small.reshape(B2, Hh, k_per_q * HW, D).contiguous()

    out_ = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=0.0, is_causal=False)

    out = out_.reshape(B, Tq_blk, Hh, HW, D).permute(0, 2, 1, 3, 4).reshape(B, Hh, Tq_blk * HW, D)
    return out.permute(0, 2, 1, 3) #[B, 13*HW,,H,D]


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, history_states: torch.Tensor):

    query = attn.to_q(hidden_states)
    key = attn.to_k(history_states)
    value = attn.to_v(history_states)
    return query, key, value



class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        history_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_emb_his: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        history_block_indices=None,

    ) -> torch.Tensor:

        query, key, value = _get_qkv_projections(attn, hidden_states,  history_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))#torch.Size([1, 20280, 12, 128])
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb_his)

        # hidden_states = dispatch_attention_fn(
        #     query,
        #     key,
        #     value,
        #     attn_mask=attention_mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        #     backend=self._attention_backend,
        #     # Reference: https://github.com/huggingface/diffusers/pull/12909 30, 52
        #     parallel_config=None,
        # )

        hidden_states = sdpa_with_block_gather_val(
            query, key, value,
            sel_k_blocks=history_block_indices, height=30, width=52, enable_flash=True
        ) 
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)



        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The WanAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use WanAttnProcessor instead. "
        )
        deprecate("WanAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return WanAttnProcessor(*args, **kwargs)


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.kv_inner_dim = self.inner_dim 

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.is_cross_attention = True

        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        history_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_emb_his: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        history_block_indices=None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            history_states,
            attention_mask,
            rotary_emb,
            rotary_emb_his,
            history_block_indices,
            **kwargs,
        )

class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
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


@maybe_allow_in_graph
class WanTransformerBlock_ca(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. cross-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
    
        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        history_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        rotary_emb_his: torch.Tensor,
        history_block_indices,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            history_states,
            None,
            rotary_emb,
            rotary_emb_his,
            history_block_indices,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

