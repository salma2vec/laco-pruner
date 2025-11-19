"""Test that performance optimizations don't break code structure."""
import pytest
import re
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLayerMergeStructure:
    """Test structure optimizations in layer_merge.py without running."""
    
    def test_cached_patterns_exist(self):
        """Verify regex patterns cache is defined."""
        from laco.layer_merge import _KNOWN_PATTERNS_CACHE
        assert len(_KNOWN_PATTERNS_CACHE) > 0
        # Check they contain pattern, name, and base
        for item in _KNOWN_PATTERNS_CACHE:
            assert len(item) == 3
            name, pattern, base = item
            assert isinstance(name, str)
            assert hasattr(pattern, 'match')  # Compiled regex
            assert isinstance(base, str)
    
    def test_no_deepcopy_import(self):
        """Verify copy.deepcopy is no longer imported."""
        with open("laco/layer_merge.py", "r") as f:
            content = f.read()
            # Should not import copy or deepcopy
            assert "import copy" not in content
            assert "copy.deepcopy" not in content
            # Should use shallow copy instead
            assert "{k: v for k, v in state_dict.items()}" in content


class TestSimilarityStructure:
    """Test structure optimizations in similarity.py."""
    
    def test_helper_function_exists(self):
        """Verify _get_hidden_states helper exists."""
        from laco.similarity import _get_hidden_states
        assert callable(_get_hidden_states)
    
    def test_batching_threshold_fixed(self):
        """Verify batching threshold is improved."""
        with open("laco/similarity.py", "r") as f:
            content = f.read()
            # Should use improved threshold
            assert "batch_size > 1 and len(texts) > 1" in content


class TestProfilingStructure:
    """Test structure optimizations in profiling.py."""
    
    def test_bytes_to_mb_constant_exists(self):
        """Verify MB conversion constant is defined."""
        from laco.profiling import _BYTES_TO_MB
        assert _BYTES_TO_MB == 1024 ** 2
        assert _BYTES_TO_MB == 1048576


class TestPrunePipelineStructure:
    """Test structure optimizations in prune_pipeline.py."""
    
    def test_no_unnecessary_cloning(self):
        """Verify unnecessary cloning is removed."""
        with open("laco/prune_pipeline.py", "r") as f:
            content = f.read()
            # The load_state_dict should not clone unnecessarily
            # Check that we create tmp_model once
            assert "tmp_model = AutoModelForCausalLM.from_config(model.config)" in content
            # Should reuse tmp_model
            assert "tmp_model.load_state_dict(tmp_state" in content


class TestBenchmarkStructure:
    """Test structure optimizations in benchmark.py."""
    
    def test_batch_generate_function_exists(self):
        """Verify batch generation helper exists."""
        from bench.benchmark import _batch_generate
        assert callable(_batch_generate)
    
    def test_uses_batched_generation(self):
        """Verify evaluation uses batched generation."""
        with open("bench/benchmark.py", "r") as f:
            content = f.read()
            # Should use _batch_generate
            assert "_batch_generate(model, tokenizer, sources" in content
            assert "_batch_generate(pruned_model, tokenizer, sources" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
