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
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn

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
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from einops import rearrange, repeat
from opt_einsum import contract

from .config_gemma3 import Gemma3TTTConfig


# Output dataclasses extending HF outputs with a per-layer fast-weight snapshot.
# `fast_weights` is a dict {layer_idx: tensor of shape (B, d, d_ff)} containing
# the cumulative (un-scaled) per-chunk ΔW captured during a forward where
# `return_fast_weights=True`. The η factor is applied at consumption, not here.
@dataclass
class Gemma3TTTBaseModelOutput(BaseModelOutputWithPast):
    fast_weights: Optional[Dict[int, torch.Tensor]] = None


@dataclass
class Gemma3TTTCausalLMOutput(CausalLMOutputWithPast):
    fast_weights: Optional[Dict[int, torch.Tensor]] = None


# Marker subclasses so _init_weights can identify TTT modules unambiguously
# (without relying on shape heuristics that collide with q_proj/o_proj).
class TTTLinear(nn.Linear):
    pass


class TTTConv1d(nn.Conv1d):
    pass


class Gemma3MLPTTT(Gemma3MLP):
    def __init__(self, config: Gemma3TTTConfig, layer_idx: Optional[int] = None) -> None:
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
    def padding(self, x: torch.Tensor) -> torch.Tensor:
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
    #
    # Three call modes:
    #   1. Vanilla / paper-style (default): forward(x, t) -> Tensor.
    #      Identical to the original implementation. Used by ICL and ttt_paper.
    #   2. Snapshot producer (strict ingest): forward(x, t, return_fast_weights=True)
    #      -> (Tensor, fw). Runs paper-style chunked update, additionally returns
    #      `fw` of shape (B, d, d_ff) — the un-scaled cumulative ΔW across ALL
    #      chunks of this input (no causal exclusion of the last chunk, since for
    #      ingest there is no "future" beyond the doc). η is NOT pre-multiplied.
    #   3. Snapshot consumer (strict answer): forward(x, t, fast_weights=fw)
    #      -> Tensor (or (Tensor, fw_passthrough) if return_fast_weights=True).
    #      Skips per-chunk evolution; uses W_eff = W_down + η * fw uniformly for
    #      every position. Question's own tokens contribute zero new ΔW.
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        fast_weights: Optional[torch.Tensor] = None,
        return_fast_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input embedding
        z = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # Vanilla path (no TTT conv module, e.g. layer is not in ttt_layers).
        if t is None or not hasattr(self, "ttt_conv"):
            out = self.down_proj(z)
            return out, None

        # Snapshot consumer path: use the supplied fast_weights uniformly.
        # fast_weights expected shape (d, d_ff) or (B, d, d_ff). η applied here.
        # Always return a tuple in the strict path so the caller's unpack works
        # regardless of whether `return_fast_weights` is set.
        if fast_weights is not None:
            scaled = self.ttt_lr * fast_weights
            base = self.down_proj.weight  # (d, d_ff)
            if scaled.dim() == 2:
                W_eff = base + scaled                                # (d, d_ff)
                out = z @ W_eff.T                                    # (B, N, d)
            elif scaled.dim() == 3:
                # Per-batch effective W_down.
                W_eff = base.unsqueeze(0) + scaled                   # (B, d, d_ff)
                out = torch.einsum("bnf,bdf->bnd", z, W_eff)         # (B, N, d)
            else:
                raise ValueError(f"fast_weights must be 2D or 3D, got {scaled.dim()}D")
            # Pass the (un-modified) snapshot back when capture was requested,
            # else None — keeps the result schema consistent with the producer
            # path while signaling that no new snapshot was computed.
            return out, (fast_weights if return_fast_weights else None)

        # Paper-style chunk-wise update path.
        t = self.padding(t)        # (b, t, c, d) = (batch_size, chunk_num, chunk_size, d_model)
        z_padded = self.padding(z) # (b, t, c, d_ff) = (batch_size, chunk_num, chunk_size, d_ff)
        bs, chunk_num, chunk_size, _ = t.shape
        t = (
            self.ttt_conv(t.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size)) # conv across d_model channels for chunk_size 
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )
        if self.ttt_proj is not None:
            # Per-chunk un-scaled ΔW (b, t, d, d_ff). Causal exclusion of last
            # chunk is enforced by the [:, :-1] slice — preserves paper math.
            delta_per_chunk_excl = contract(
                "b t c h, b t c d, d e -> b t e h",
                z_padded[:, :-1], t[:, :-1], self.ttt_proj.weight,
            )
        else:
            delta_per_chunk_excl = contract(
                "b t c h, b t c d -> b t d h",
                z_padded[:, :-1], t[:, :-1],
            )
        delta_down_proj = torch.cat(
            [
                repeat(self.down_proj.weight, "d h -> b 1 d h", b=bs),
                delta_per_chunk_excl * self.ttt_lr,
            ],
            dim=1,
        )
        delta_down_proj_sum = delta_down_proj.cumsum(dim=1)
        down_proj = contract("b t d h, b t c h -> b t c d", delta_down_proj_sum, z_padded)
        out = rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]

        if not return_fast_weights:
            return out, None

        # Build the un-scaled snapshot summing ΔW across ALL chunks (incl. last).
        # delta_per_chunk_excl already covers chunks 0..T-2; compute chunk T-1
        # and add it. Skipping the last chunk's contribution would lose the tail
        # of the doc — undesirable for ingest where there is no "future".
        if self.ttt_proj is not None:
            last_delta = contract(
                "b c h, b c d, d e -> b e h",
                z_padded[:, -1], t[:, -1], self.ttt_proj.weight,
            )
        else:
            last_delta = contract(
                "b c h, b c d -> b d h",
                z_padded[:, -1], t[:, -1],
            )
        snapshot = delta_per_chunk_excl.sum(dim=1) + last_delta      # (B, d, d_ff)
        return out, snapshot

class Gemma3DecoderLayerTTT(Gemma3DecoderLayer):
    def __init__(self, config: Gemma3TTTConfig, layer_idx: int) -> None:
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
        fast_weights: torch.Tensor | None = None,
        return_fast_weights: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        hidden_states, fw_out = self.mlp(
            hidden_states,
            t=target_states,
            fast_weights=fast_weights,
            return_fast_weights=return_fast_weights,
        )

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, fw_out


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
    def _init_weights(self, module: nn.Module) -> None:
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

    def __init__(self, config: Gemma3TTTConfig) -> None:
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
        fast_weights: Optional[Dict[int, torch.Tensor]] = None,
        return_fast_weights: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma3TTTBaseModelOutput:
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

        # If the caller wants per-layer fast-weight snapshots back, allocate the
        # collector dict. Only TTT layers will populate it.
        out_fast_weights: Optional[Dict[int, torch.Tensor]] = (
            {} if return_fast_weights else None
        )

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Per-layer snapshot to consume (only meaningful on TTT layers).
            in_fw = None
            if fast_weights is not None and getattr(decoder_layer, "is_ttt_layer", False):
                in_fw = fast_weights.get(i)

            hidden_states, fw_out = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),
                fast_weights=in_fw,
                return_fast_weights=return_fast_weights,
                **kwargs,
            )
            if out_fast_weights is not None and fw_out is not None:
                out_fast_weights[i] = fw_out

        hidden_states = self.norm(hidden_states)

        return Gemma3TTTBaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            fast_weights=out_fast_weights,
        )


class Gemma3ForCausalLMTTT(Gemma3PreTrainedModelTTT, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3TTTConfig

    def __init__(self, config: Gemma3TTTConfig) -> None:
        super().__init__(config)
        self.model = Gemma3TextModelTTT(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self) -> None:
        # adapter-style training: ttt_proj (W_target), ttt_conv, and the MLP
        # down_proj (W_down) are trainable — but only the down_proj on TTT
        # layers, since W_down is the surface the per-chunk ΔW updates. On
        # non-TTT layers there is no fast-weight stream, so down_proj stays
        # frozen along with the rest of the base model.
        ttt_layers = set(getattr(self.config, "ttt_layers", None) or [])

        for name, param in self.named_parameters():
            if "ttt_proj" in name or "ttt_conv" in name:
                param.requires_grad = True
                continue
            if "down_proj" in name:
                # name is like "model.layers.{i}.mlp.down_proj.weight"
                parts = name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    layer_idx = -1
                param.requires_grad = layer_idx in ttt_layers
                continue
            param.requires_grad = False

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
        fast_weights: Optional[Dict[int, torch.Tensor]] = None,
        return_fast_weights: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma3TTTCausalLMOutput:
        outputs: Gemma3TTTBaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fast_weights=fast_weights,
            return_fast_weights=return_fast_weights,
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

        return Gemma3TTTCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            fast_weights=outputs.fast_weights,
        )


__all__ = [
    "Gemma3PreTrainedModelTTT",
    "Gemma3TextModelTTT",
    "Gemma3ForCausalLMTTT",
    "Gemma3TTTBaseModelOutput",
    "Gemma3TTTCausalLMOutput",
]
