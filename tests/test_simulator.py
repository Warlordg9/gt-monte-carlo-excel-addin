import numpy as np
from engine.distributions import triangular_sample

def test_triangular_mean():
    np.random.seed(42)
    samples = triangular_sample(0.48, 0.63, 0.78, 100000)
    expected = (0.48 + 0.63 + 0.78) / 3
    assert abs(np.mean(samples) - expected) < 0.005
