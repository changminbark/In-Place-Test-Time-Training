# coding=utf-8                                                                                                                 
# Copyright 2025 Google LLC and the HuggingFace Inc. team. All rights reserved.
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
#                                                                                                                              
# This file is a modified version of transformers/models/gemma3/modeling_gemma3.py.
# Modifications Copyright 2026 Chang Min Bark and Hung Ngo.                                                                    
# Modifications add In-Place Test-Time Training (TTT) adapter modules                                                          
# (Conv1D + W_target) to the MLP and a frozen-base training mode.
from typing import Optional, Union

import torch
from torch import nn

from transformers import PreTrainedModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import (
    Cache,
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
    TransformersKwargs,
    Unpack,
    _bidirectional_window_overlay,
    auto_docstring,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from einops import rearrange, repeat
from opt_einsum import contract

from .config_gemma3 import Gemma3TTTConfig


# Marker subclasses so _init_weights can identify TTT modules unambiguously
# (without relying on shape heuristics that collide with q_proj/o_proj).
class TTTLinear(nn.Linear):
    pass


class TTTConv1d(nn.Conv1d):
    pass


class Gemma3MLPTTT(Gemma3MLP):
    def __init__(self, config: Gemma3TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config)
        
        # TTT Add-on
        self.layer_idx = -1 if layer_idx is None else layer_idx
        if getattr(config, "use_ttt", False) and self.layer_idx in getattr(config, "ttt_layers", []):
            self.ttt_chunk = getattr(config, "ttt_chunk", 8192)
            if getattr(config, "ttt_proj", True):
                self.ttt_proj = TTTLinear(self.hidden_size, self.hidden_size, bias=False)
            else:
                self.ttt_proj = None
            self.ttt_lr = getattr(config, "ttt_lr", 0.3)
            self.ttt_conv = TTTConv1d(
                self.hidden_size, self.hidden_size, kernel_size=5, padding=2,
                groups=self.hidden_size, bias=False,
            )
            
    # TTT chunk padding
    def padding(self, x: torch.Tensor):
        if not hasattr(self, "ttt_chunk"):
            return x
        if x.shape[1] % self.ttt_chunk != 0:
            padding_embeddings = torch.zeros(
                [x.shape[0], self.ttt_chunk - x.shape[1] % self.ttt_chunk, x.shape[2]],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, padding_embeddings], dim=1)
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)
        
    # TTT forward
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        # Input embedding
        z = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        
        # Vanilla Path
        if t is None or not hasattr(self, "ttt_conv"):
            return self.down_proj(z)
        
        # TTT Path
        t = self.padding(t) # shape: (batch_size, chunk_num, chunk_size, d_model)
        z_padded = self.padding(z) # shape: (batch_size, chunk_num, chunk_size, d_ff)
        bs, chunk_num, chunk_size, _ = t.shape
        t = (
            self.ttt_conv(t.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size)) # conv across d_model channels for chunk_size 
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )
        if self.ttt_proj is not None:
            delta_down_proj = contract(
                "b t c h, b t c d, d e -> b t e h",
                z_padded[:, :-1], t[:, :-1], self.ttt_proj.weight,
            )
        else:
            delta_down_proj = contract(
                "b t c h, b t c d -> b t d h",
                z_padded[:, :-1], t[:, :-1],
            )
        delta_down_proj = torch.cat(
            [repeat(self.down_proj.weight, "d h -> b 1 d h", b=bs), delta_down_proj * self.ttt_lr],
            dim=1,
        )
        delta_down_proj_sum = delta_down_proj.cumsum(dim=1)
        down_proj = contract("b t d h, b t c h -> b t c d", delta_down_proj_sum, z_padded)
        return rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :] # shape: (batch, seq_len, d_model)

class Gemma3DecoderLayerTTT(Gemma3DecoderLayer):
    def __init__(self, config: Gemma3TTTConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLPTTT(config, layer_idx=layer_idx)
        ttt_layers = getattr(config, "ttt_layers", []) or []
        self.is_ttt_layer = getattr(config, "use_ttt", False) and layer_idx in ttt_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        target_states: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        # TTT MLP 
        if self.is_ttt_layer and target_states is None:
            target_states = hidden_states
        hidden_states = self.mlp(hidden_states, t=target_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3PreTrainedModelTTT(Gemma3PreTrainedModel):
    config: Gemma3TTTConfig
    config_class = Gemma3TTTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Gemma3DecoderLayerTTT",
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Gemma3DecoderLayerTTT,
        "attentions": Gemma3Attention,
    }
    input_modalities = ("text",)
    
    @torch.no_grad()  # will not affect TTT layers since they bypass autograd
    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)

        # TTT modules: custom init so the TTT branch starts as a near-identity
        # perturbation of the frozen base (diagonal W_target, zero conv).
        if isinstance(module, TTTLinear):
            if module.weight.device.type == "meta":
                return
            # Square matrix (ttt_proj is d_model x d_model)
            diag_size = module.weight.shape[0]
            weight_data = module.weight.data
            if hasattr(weight_data, "_local_tensor"):
                # DTensor: operate on the local shard only
                import torch.distributed as dist

                local_tensor = weight_data._local_tensor
                local_tensor.zero_()

                local_rows = local_tensor.shape[0]
                num_cols = local_tensor.shape[1]
                rank = dist.get_rank()
                start_row = rank * local_rows

                g = torch.Generator(device=local_tensor.device)
                g.manual_seed(42)
                all_diag_values = torch.randn(
                    diag_size, generator=g, device=local_tensor.device, dtype=local_tensor.dtype,
                ) * std

                local_row_indices = torch.arange(local_rows, device=local_tensor.device)
                global_col_indices = start_row + local_row_indices

                valid_mask = global_col_indices < num_cols
                local_row_indices = local_row_indices[valid_mask]
                global_col_indices = global_col_indices[valid_mask]

                if len(local_row_indices) > 0:
                    local_tensor[local_row_indices, global_col_indices] = all_diag_values[global_col_indices]
            else:
                weight_data.zero_()
                diag_values = torch.randn(diag_size, device=weight_data.device, dtype=weight_data.dtype) * std
                indices = torch.arange(diag_size, device=weight_data.device)
                weight_data[indices, indices] = diag_values
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            return
        if isinstance(module, TTTConv1d):
            # TTT conv: zero init so the TTT branch starts as a no-op perturbation of the frozen base
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
            return

        # Non-TTT modules: defer to upstream Gemma3 init (also honors the
        # _is_hf_initialized skip-flag so loaded checkpoints aren't overwritten).
        super()._init_weights(module)


class Gemma3TextModelTTT(Gemma3PreTrainedModelTTT):
    config: Gemma3TTTConfig
    input_modalities = ("text",)

    def __init__(self, config: Gemma3TTTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size ** 0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayerTTT(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False

        self.use_ttt = getattr(config, "use_ttt", False)
        self.ttt_layers = getattr(config, "ttt_layers", []) or []
        self.ttt_target = getattr(config, "ttt_target", "hidden_states")

        self.post_init()

    def _resolve_ttt_target_states(
        self,
        decoder_layer: Gemma3DecoderLayerTTT,
        inputs_embeds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_ttt or not decoder_layer.is_ttt_layer:
            return None
        if self.ttt_target == "input_embed":
            return inputs_embeds
        # ttt_target == "hidden_states": decoder layer falls back to its own pre-MLP hidden_states
        return None

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if getattr(self.config, "use_bidirectional_attention", False):
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                    self.config.sliding_window
                )

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # Embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Gemma3ForCausalLMTTT(Gemma3PreTrainedModelTTT, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3TTTConfig

    def __init__(self, config: Gemma3TTTConfig):
        super().__init__(config)
        self.model = Gemma3TextModelTTT(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        # adapter-only training: only ttt_proj + ttt_conv get gradients
        for name, param in self.named_parameters():
            param.requires_grad = ("ttt_proj" in name) or ("ttt_conv" in name)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Gemma3PreTrainedModelTTT",
    "Gemma3TextModelTTT",
    "Gemma3ForCausalLMTTT",
]
