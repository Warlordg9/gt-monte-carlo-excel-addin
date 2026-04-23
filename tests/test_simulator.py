import numpy as np
import json
import pandas as pd
from engine.simulator import MonteCarloEngine
from engine.distributions import triangular_sample, lhs_triangular_sample

def test_triangular_mean():
    np.random.seed(42)
    samples = triangular_sample(0.48, 0.63, 0.78, 100000)
    expected = (0.48 + 0.63 + 0.78) / 3
    assert abs(np.mean(samples) - expected) < 0.005

def test_monte_carlo_engine():
    with open("config/gt36_s5_config.json") as f:
        config = json.load(f)
    engine = MonteCarloEngine(config)
    results = engine.run()
    assert results.shape[0] == config["n_iterations"]
    assert results.shape[1] == len(config["outputs"])
    stats = engine.get_statistics()
    # Проверяем, что медиана не NaN
    assert not np.isnan(stats["GT_Gross_Power_MW"]["median"])

def test_lhs_seed_reproducibility():
    with open("config/gt36_s5_config.json") as f:
        config = json.load(f)
    config["sampling_method"] = "latin_hypercube"
    config["random_seed"] = 42
    config["n_iterations"] = 1000

    engine1 = MonteCarloEngine(config)
    engine1.run()
    samples1 = engine1.input_samples_df.copy()

    engine2 = MonteCarloEngine(config)
    engine2.run()
    samples2 = engine2.input_samples_df

    pd.testing.assert_frame_equal(samples1, samples2)