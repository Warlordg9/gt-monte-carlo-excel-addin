import numpy as np

def triangular_sample(min_val: float, mode_val: float, max_val: float, size: int) -> np.ndarray:
    if not (min_val <= mode_val <= max_val):
        raise ValueError("min <= mode <= max required")
    c = (mode_val - min_val) / (max_val - min_val)
    u = np.random.rand(size)
    x = np.where(
        u < c,
        min_val + np.sqrt(u * (max_val - min_val) * (mode_val - min_val)),
        max_val - np.sqrt((1 - u) * (max_val - min_val) * (max_val - mode_val))
    )
    return x
