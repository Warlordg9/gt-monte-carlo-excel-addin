import numpy as np
from scipy.stats.qmc import LatinHypercube

def triangular_sample(min_val: float, mode_val: float, max_val: float, size: int) -> np.ndarray:
    """
    Генерация выборки из треугольного распределения методом обратного преобразования.
    Используется стандартный генератор numpy.random (зависит от глобального seed).
    """
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


def lhs_triangular_sample(min_val: float, mode_val: float, max_val: float, size: int, seed: int = None) -> np.ndarray:
    """
    Генерация выборки из треугольного распределения методом Latin Hypercube Sampling.
    Равномерные квантили берутся из LHS, затем преобразуются через inverse CDF треугольного.
    Параметр seed обеспечивает воспроизводимость.
    """
    if not (min_val <= mode_val <= max_val):
        raise ValueError("min <= mode <= max required")
    sampler = LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=size).flatten()
    c = (mode_val - min_val) / (max_val - min_val)
    x = np.where(
        u < c,
        min_val + np.sqrt(u * (max_val - min_val) * (mode_val - min_val)),
        max_val - np.sqrt((1 - u) * (max_val - min_val) * (max_val - mode_val))
    )
    return x