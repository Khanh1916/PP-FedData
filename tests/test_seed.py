"""Test deterministic seeding for reproducibility."""
import numpy as np
import pytest

from ppfeddata.utils import set_seed


class TestSetSeed:
    """Tests for set_seed reproducibility."""

    def test_numpy_deterministic(self):
        """Same seed produces same numpy random sequence."""
        set_seed(42)
        a = np.random.rand(100)
        set_seed(42)
        b = np.random.rand(100)
        np.testing.assert_array_equal(a, b)

    def test_numpy_different_seeds(self):
        """Different seeds produce different sequences."""
        set_seed(0)
        a = np.random.rand(100)
        set_seed(1)
        b = np.random.rand(100)
        assert not np.array_equal(a, b)

    def test_torch_deterministic(self):
        """Same seed produces same torch random sequence."""
        torch = pytest.importorskip("torch")
        set_seed(42)
        a = torch.randn(100)
        set_seed(42)
        b = torch.randn(100)
        assert torch.equal(a, b)

    def test_torch_different_seeds(self):
        """Different seeds produce different torch sequences."""
        torch = pytest.importorskip("torch")
        set_seed(0)
        a = torch.randn(100)
        set_seed(1)
        b = torch.randn(100)
        assert not torch.equal(a, b)

    def test_python_random_deterministic(self):
        """Same seed produces same python random sequence."""
        import random
        set_seed(7)
        a = [random.random() for _ in range(50)]
        set_seed(7)
        b = [random.random() for _ in range(50)]
        assert a == b
