"""
Test script for the DeepseekV3 model, focusing on forward pass and kv cache implementation.

The kv cache won't work with newer versions of transformers as SinkCache has been deprecated.
"""

import time

import lightning as L
import torch
from lightning.pytorch.utilities.model_summary import summarize

from configuration_deepseek import DeepseekV3Config
from kv_cache import DeepseekV3RollingCache
from navigation_deepseek import DeepseekV3ForRobotNavigation


class LightningDeepseekV3(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.example_input = {
            "inputs_embeds": torch.randn(1, model.config.max_position_embeddings, model.config.hidden_size),
            "attention_mask": torch.ones(1, model.config.max_position_embeddings, dtype=torch.float32),
        }


def test_deepseekv3():
    config = DeepseekV3Config(
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=16,  # num_attention_heads == num_key_value_heads means MHA
        hidden_size=1176,
        n_routed_experts=None,  # if None, utilize MLP in forward pass
        intermediate_size=2048,  # MLP hidden size if n_routed_experts == None
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        kv_lora_rank=256,
        q_lora_rank=512,
        use_cache=True,
        max_position_embeddings=100,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepseekV3ForRobotNavigation(config).to(device)
    model.eval()

    l = LightningDeepseekV3(model)
    print(summarize(l, max_depth=3))

    input_embeds = torch.randn(
        1,
        2 * config.max_position_embeddings,
        config.hidden_size,
        dtype=torch.float32,
        device=device,
    )

    attention_mask = torch.ones(1, config.max_position_embeddings, dtype=torch.float32, device=device)
    with torch.inference_mode():
        start = time.time()
        _ = model(
            inputs_embeds=input_embeds[:, config.max_position_embeddings:],
            attention_mask=attention_mask,
        )
        print(f"Time taken for full forward pass: {time.time() - start:.4f} seconds")

    # Test with past_key_values
    with torch.inference_mode():
        cache1 = DeepseekV3RollingCache(
            window_length=config.max_position_embeddings,
            qk_rope_head_dim=config.qk_rope_head_dim,
        )

        for i in range(input_embeds.shape[1]):
            emb = input_embeds[:, [i]]
            attention_mask = torch.ones(
                1,
                min(i + 1, config.max_position_embeddings),
                dtype=torch.float32,
                device=device,
            )
            start = time.time()
            output = model(
                inputs_embeds=emb,
                attention_mask=attention_mask,
                past_key_values=cache1,
            )
            print(f"Time taken for token {i}: {time.time() - start:.4f} seconds")
            cache1 = output.past_key_values

        last_part = input_embeds[:, config.max_position_embeddings:]
        cache2 = DeepseekV3RollingCache(
            window_length=config.max_position_embeddings,
            qk_rope_head_dim=config.qk_rope_head_dim,
        )
        for i in range(last_part.shape[1]):
            emb = last_part[:, [i]]
            attention_mask = torch.ones(
                1,
                min(i + 1, config.max_position_embeddings),
                dtype=torch.float32,
                device=device,
            )
            output = model(
                inputs_embeds=emb,
                attention_mask=attention_mask,
                past_key_values=cache2,
            )
            cache2 = output.past_key_values

        attention_mask = torch.ones(
            1,
            config.max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )
        cache3 = DeepseekV3RollingCache(
            window_length=config.max_position_embeddings,
            qk_rope_head_dim=config.qk_rope_head_dim,
        )
        output = model(
            inputs_embeds=last_part,
            attention_mask=attention_mask,
            past_key_values=cache3,
        )
        cache3 = output.past_key_values

        assert torch.allclose(
            cache1.key_cache[0][..., :config.qk_nope_head_dim],
            cache3.key_cache[0][..., :config.qk_nope_head_dim],
            atol=1e-6,
        ), (
            cache1.key_cache[0][0, 0, 0, :5],
            cache3.key_cache[0][0, 0, 0, :5],
        )

        # Check if the caches are equal
        assert torch.allclose(
            cache1.key_cache[0][..., :config.qk_nope_head_dim],
            cache2.key_cache[0][..., :config.qk_nope_head_dim],
        )

        assert torch.allclose(
            cache1.key_cache[0][..., config.qk_nope_head_dim:],
            cache2.key_cache[0][..., config.qk_nope_head_dim:],
            atol=1e-6,
        ), (
            cache1.key_cache[0][0, 0, 0, config.qk_nope_head_dim : config.qk_nope_head_dim + 5],
            cache3.key_cache[0][0, 0, 0, config.qk_nope_head_dim : config.qk_nope_head_dim + 5],
            cache2.key_cache[0][0, 0, 0, config.qk_nope_head_dim : config.qk_nope_head_dim + 5],
        )


if __name__ == "__main__":
    test_deepseekv3()
