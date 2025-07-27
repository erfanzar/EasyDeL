import unittest

from flax.nnx import Rngs

import easydel as ed


class TestLoraLayerOperations(unittest.TestCase):
    """Test suite for LoRA (Low-Rank Adaptation) operations on LlamaForCausalLM."""

    def setUp(self):
        """Initialize test environment with a basic model configuration."""
        self.config = ed.LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
        )
        self.model = ed.LlamaForCausalLM(
            config=self.config,
            rngs=Rngs(0),
        )
        self.lora_rank = 32
        self.layer_pattern = ".*(q_proj|k_proj).*"

    def test_lora_application(self):
        """Test successful application of LoRA to specified layers."""
        model_with_lora = self.model.apply_lora_to_layers(self.lora_rank, self.layer_pattern)

        # Test all layers have LoRA parameters where expected
        for layer_idx in range(self.config.num_hidden_layers):
            layer = model_with_lora.model.layers[layer_idx].self_attn

            # Check q_proj and k_proj have LoRA parameters
            for proj in ["q_proj", "k_proj"]:
                proj_layer = getattr(layer, proj)
                self.assertTrue(hasattr(proj_layer, "lora_a"), f"Layer {layer_idx} {proj} missing lora_a")
                self.assertTrue(hasattr(proj_layer, "lora_b"), f"Layer {layer_idx} {proj} missing lora_b")

            # Verify v_proj doesn't have LoRA parameters
            self.assertFalse(
                hasattr(layer.v_proj, "lora_a"),
                f"Layer {layer_idx} v_proj shouldn't have lora_a",
            )
            self.assertFalse(
                hasattr(layer.v_proj, "lora_b"),
                f"Layer {layer_idx} v_proj shouldn't have lora_b",
            )

    def test_lora_params_splitting_and_merging(self):
        """Test LoRA parameter splitting and merging operations."""
        # Apply LoRA and split parameters
        model_with_lora = self.model.apply_lora_to_layers(self.lora_rank, self.layer_pattern)
        lora_params = model_with_lora.split_lora_params()

        # Verify split parameters structure
        self.assertIsInstance(lora_params, dict, "Split parameters should be a dictionary")

        # Merge parameters back
        merged_model = model_with_lora.merge_lora_params(lora_params)

        # Verify merged model structure
        self.assertEqual(
            type(merged_model),
            type(self.model),
            "Merged model should maintain original model type",
        )

    def test_lora_unwrapping(self):
        """Test complete LoRA workflow including unwrapping."""
        # Apply LoRA
        model_with_lora = self.model.apply_lora_to_layers(self.lora_rank, self.layer_pattern)

        # Split and merge parameters
        lora_params = model_with_lora.split_lora_params()
        merged_model = model_with_lora.merge_lora_params(lora_params)

        # Unwrap LoRA
        final_model = merged_model.unwrap_lora_to_layers()

        # Verify LoRA parameters are removed
        for layer_idx in range(self.config.num_hidden_layers):
            layer = final_model.model.layers[layer_idx].self_attn
            for proj in ["q_proj", "k_proj"]:
                proj_layer = getattr(layer, proj)
                self.assertFalse(
                    hasattr(proj_layer, "lora_a"),
                    f"Layer {layer_idx} {proj} should not have lora_a after unwrapping",
                )
                self.assertFalse(
                    hasattr(proj_layer, "lora_b"),
                    f"Layer {layer_idx} {proj} should not have lora_b after unwrapping",
                )

    def test_invalid_lora_rank(self):
        """Test handling of invalid LoRA ranks."""
        with self.assertRaises(ValueError):
            self.model.apply_lora_to_layers(-1, self.layer_pattern)

        with self.assertRaises(ValueError):
            self.model.apply_lora_to_layers(0, self.layer_pattern)


if __name__ == "__main__":
    unittest.main()
