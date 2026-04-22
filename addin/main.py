import xlwings as xw
import numpy as np

@xw.func
def mc_triangle(min_val: float, mode_val: float, max_val: float) -> float:
    c = (mode_val - min_val) / (max_val - min_val)
    u = np.random.rand()
    if u < c:
        return min_val + np.sqrt(u * (max_val - min_val) * (mode_val - min_val))
    else:
        return max_val - np.sqrt((1 - u) * (max_val - min_val) * (max_val - mode_val))

@xw.sub
def run_monte_carlo():
    xw.sheets.active.range("A1").value = "Monte Carlo simulation placeholder"
