"""
A KV cache implementation for DeepseekV3 model, based on the SinkCache from the `transformers` library.

Looks like transformers has deprecated SinkCache, (which was apparently bugged)
so would need to write a new cache implementation.

Dummy left as a placeholder.
"""

from typing import Dict, Any, Optional, Tuple

import torch
class DeepseekV3RollingCache():
    def __init__(self, window_length, qk_rope_head_dim):
        raise NotImplementedError("This is a placeholder implementation. The actual implementation of the DeepseekV3RollingCache out of service.")
        return
'''
from transformers import SinkCache

class DeepseekV3RollingCache(SinkCache):
    """
    Repurposed from `transformers` library to implement a rolling cache for the DeepseekV3 model.
    This cache is used to store the key and value states of the attention mechanism in a rolling manner,
    so that when old tokens are shifted out, the RoPE embeddings are recomputed for the remaining tokens.

    This keeps the positional embeddings always within the window length.
    """

    def __init__(self, window_length, qk_rope_head_dim):
        super().__init__(window_length=window_length, num_sink_tokens=0)
        self._qk_rope_head_dim = qk_rope_head_dim

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the sin/cos cache, which holds sin/cos values for all possible positions
        if using_rope and layer_idx == 0:
            # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
            # after all RoPE models have a llama-like cache utilization.
            if cos.dim() == 2:
                self._cos_cache = cos
                self._sin_cache = sin
            else:
                if self._cos_cache is None:
                    self._cos_cache = cos[0, ...]
                    self._sin_cache = sin[0, ...]
                elif self._cos_cache.shape[0] < self.window_length:
                    self._cos_cache = torch.cat([self._cos_cache, cos[0, ...]], dim=0)
                    self._sin_cache = torch.cat([self._sin_cache, sin[0, ...]], dim=0)

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

# Earhtrovers: In the original code, this was < self.window_length: which might be a bug?
        elif key_states.shape[-2] + self.get_seq_length(layer_idx) <= self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:

            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, self._cos_cache[: self.window_length], self._sin_cache[: self.window_length]
                )

# Earhtrovers: In the original code, the partial_rotation_size was used to slice the keys_to_keep tensor.
# With Deepseek, the dims corresponding to qk_rope_head are at the end, so this had to be modified.
                if self._qk_rope_head_dim is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., -self._qk_rope_head_dim:],
                        keys_to_keep[..., :-self._qk_rope_head_dim],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if self._qk_rope_head_dim is not None:
                    keys_to_keep = torch.cat((keys_pass, keys_to_keep), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)
            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
'''