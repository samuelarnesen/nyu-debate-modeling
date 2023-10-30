# copied from @philschmid
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/utils/llama_patch.py

# flash decoder work copied from
# https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/

from typing import List, Optional, Tuple
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
import warnings
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from peft.tuners.lora import LoraLayer

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_with_kvcache
    from flash_attn.bert_padding import unpad_input, pad_input
except Exception:
    raise ModuleNotFoundError(
        "Please install FlashAttention first, e.g., with pip install flash-attn --no-build-isolation, Learn more at https://github.com/Dao-AILab/flash-attention#installation-and-features"
    )

try:
    from einops import rearrange
except Exception:
    raise ModuleNotFoundError("Please install einops first, e.g., with pip install einops")

from utils.logger_utils import LoggerUtils


LOGGER = LoggerUtils.get_default_logger(__name__)


# ADAPTED from https://github.com/allenai/open-instruct/blob/main/open_instruct/llama_flash_attn_monkey_patch.py
# AND https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py
# AND https://github.com/LAION-AI/Open-Assistant/blob/04fa9a24b2a58c8885b8aa6a2eb02b18de6b4961/model/model_training/models/patching_llama.py
# AND Sourabh https://github.com/huggingface/transformers/commit/ee81bf5aee0d65f005d157c013777e3d27d8d6bf
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if output_attentions:
        warnings.warn("Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
        output = flash_attn_varlen_qkvpacked_func(qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
        output = rearrange(
            pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
            "b s (h d) -> b s h d",
            h=nheads,
        )
    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        print(
            "Flash attention is only supported on Ampere or Hopper GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward


def unplace_flash_attn_with_attn():
    import importlib
    import transformers

    print("Reloading llama model, unpatching flash attention")
    importlib.reload(transformers.models.llama.modeling_llama)


# Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch_dtype)
        if "norm" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    return model


def flash_decoding_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)

    kv_seq_len = key_states.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[-2]
        kv_seq_len += past_kv_len

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    rotary_dim = cos.shape[-1]
    cos, sin = cos.squeeze(0, 1)[:, : rotary_dim // 2].contiguous(), sin.squeeze(0, 1)[:, : rotary_dim // 2].contiguous()

    if past_key_value is not None:
        key_cache = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
        value_cache = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
    else:
        key_cache = key_states
        value_cache = value_states

    assert not output_attentions, "output_attentions is not supported"

    q = query_states  # [bsz, q_len, nh, hd]
    k, v = key_states, value_states  # [bsz, q_len, nh, hd]

    output = flash_attn_with_kvcache(
        q,
        key_cache,
        value_cache,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=past_kv_len,
        softmax_scale=None,
        causal=True,
        rotary_interleaved=False,
    )
    output = rearrange(output, "b s h d -> b s (h d)", b=bsz)

    past_key_value = (
        (key_cache[:, :kv_seq_len].transpose(1, 2), value_cache[:, :kv_seq_len].transpose(1, 2)) if use_cache else None
    )

    output = self.o_proj(output)

    return output, None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    return attention_mask


def replace_with_flash_decoding():
    if flash_attn_with_kvcache != None:
        print("USE_FLASH_ATTENTION: ", True)
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = flash_decoding_forward
    else:
        print("USE_FLASH_ATTENTION: ", False)
