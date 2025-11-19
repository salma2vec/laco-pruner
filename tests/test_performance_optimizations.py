"""Test that performance optimizations don't break functionality."""
import pytest
import torch
import re
from laco.layer_merge import (
    find_layer_prefixes, 
    partition_keys_by_layer, 
    rdscl_merge_state_dict,
    _KNOWN_PATTERNS_CACHE
)
from laco.similarity import _get_hidden_states


class TestLayerMergeOptimizations:
    """Test optimizations in layer_merge.py."""
    
    def test_cached_patterns_exist(self):
        """Verify regex patterns are cached."""
        assert len(_KNOWN_PATTERNS_CACHE) > 0
        # Check they are compiled patterns
        for name, pattern, base in _KNOWN_PATTERNS_CACHE:
            assert hasattr(pattern, 'match')
    
    def test_find_layer_prefixes_with_cache(self):
        """Test find_layer_prefixes works with cached patterns."""
        # Create a mock state dict with GPT-2 style layers
        state_dict = {
            "transformer.h.0.attn.weight": torch.randn(10, 10),
            "transformer.h.0.mlp.weight": torch.randn(10, 10),
            "transformer.h.1.attn.weight": torch.randn(10, 10),
            "transformer.h.1.mlp.weight": torch.randn(10, 10),
        }
        
        base, num_layers = find_layer_prefixes(state_dict)
        assert base == "transformer.h"
        assert num_layers == 2
    
    def test_partition_keys_optimized(self):
        """Test partition_keys_by_layer with optimized regex."""
        state_dict = {
            "model.layers.0.attn.weight": torch.randn(10, 10),
            "model.layers.0.mlp.weight": torch.randn(10, 10),
            "model.layers.1.attn.weight": torch.randn(10, 10),
        }
        
        per_layer = partition_keys_by_layer(state_dict, "model.layers", 2)
        assert len(per_layer[0]) == 2
        assert len(per_layer[1]) == 1
    
    def test_rdscl_merge_shallow_copy(self):
        """Test that rdscl_merge uses shallow copy correctly."""
        state_dict = {
            "model.layers.0.attn.weight": torch.randn(10, 10),
            "model.layers.0.mlp.weight": torch.randn(10, 10),
            "model.layers.1.attn.weight": torch.randn(10, 10),
            "model.layers.1.mlp.weight": torch.randn(10, 10),
        }
        
        # Merge layer 1 into layer 0
        result = rdscl_merge_state_dict(state_dict, "model.layers", 2, l=0, K=1, partial="full")
        
        # Check that layer 1 keys are removed
        assert "model.layers.1.attn.weight" not in result
        assert "model.layers.1.mlp.weight" not in result
        
        # Check that layer 0 keys are updated
        assert "model.layers.0.attn.weight" in result
        assert "model.layers.0.mlp.weight" in result
        
        # Verify the merged tensor is different from original
        assert not torch.equal(result["model.layers.0.attn.weight"], 
                              state_dict["model.layers.0.attn.weight"])


class TestSimilarityOptimizations:
    """Test optimizations in similarity.py."""
    
    def test_get_hidden_states_helper(self):
        """Test the helper function for extracting hidden states."""
        
        # Mock output with hidden_states
        class MockOutput1:
            def __init__(self):
                self.hidden_states = [torch.randn(1, 5, 10), torch.randn(1, 5, 10)]
        
        out1 = MockOutput1()
        hidden = _get_hidden_states(out1)
        assert hidden.shape == (1, 5, 10)
        
        # Mock output with last_hidden_state
        class MockOutput2:
            def __init__(self):
                self.hidden_states = None
                self.last_hidden_state = torch.randn(1, 5, 10)
        
        out2 = MockOutput2()
        hidden = _get_hidden_states(out2)
        assert hidden.shape == (1, 5, 10)
        
        # Mock output with logits only
        class MockOutput3:
            def __init__(self):
                self.logits = torch.randn(1, 5, 10)
        
        out3 = MockOutput3()
        hidden = _get_hidden_states(out3)
        assert hidden.shape == (1, 5, 10)


class TestProfilingOptimizations:
    """Test optimizations in profiling.py."""
    
    def test_bytes_to_mb_constant(self):
        """Test that MB conversion constant is correct."""
        from laco.profiling import _BYTES_TO_MB
        assert _BYTES_TO_MB == 1024 ** 2
        assert _BYTES_TO_MB == 1048576


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
