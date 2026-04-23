import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import spearmanr
from .distributions import triangular_sample, lhs_triangular_sample


class MonteCarloEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        config: словарь, содержащий:
            - n_iterations: int
            - random_seed: int
            - sampling_method: "random" или "latin_hypercube"
            - inputs: список словарей с полями:
                name, distribution, min, mode, max, ml, unit_step, coefficients
            - outputs: список словарей с полями name, ml
        """
        self.config = config
        self.n_iterations = config.get("n_iterations", 50000)
        self.random_seed = config.get("random_seed", 42)
        self.sampling_method = config.get("sampling_method", "random")
        self.inputs = config.get("inputs", [])
        self.outputs_config = config.get("outputs", [])
        self.results_df = None
        self.input_samples_df = None

    def _generate_input_samples(self) -> pd.DataFrame:
        """Генерирует DataFrame со всеми входными выборками согласно указанному методу."""
        samples_dict = {}
        for inp in self.inputs:
            name = inp["name"]
            dist = inp.get("distribution", "triangular")
            if dist != "triangular":
                raise ValueError(f"Unsupported distribution: {dist}")
            min_val, mode_val, max_val = inp["min"], inp["mode"], inp["max"]

            if self.sampling_method == "random":
                samples_dict[name] = triangular_sample(min_val, mode_val, max_val, self.n_iterations)
            elif self.sampling_method == "latin_hypercube":
                samples_dict[name] = lhs_triangular_sample(
                    min_val, mode_val, max_val, self.n_iterations, seed=self.random_seed
                )
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        return pd.DataFrame(samples_dict)

    def _apply_linear_model(self, input_samples_df: pd.DataFrame) -> pd.DataFrame:
        """Применяет линейную модель на основе обменных коэффициентов."""
        # Инициализируем DataFrame выходов с ML значениями
        outputs_df = pd.DataFrame(
            {out["name"]: out["ml"] for out in self.outputs_config},
            index=range(self.n_iterations)
        )
        for inp in self.inputs:
            name = inp["name"]
            ml_val = inp["ml"]
            step = inp["unit_step"]
            coeffs = inp["coefficients"]
            delta = input_samples_df[name] - ml_val
            delta_norm = delta / step
            for out_name, coeff in coeffs.items():
                outputs_df[out_name] += delta_norm * coeff
        return outputs_df

    def run(self) -> pd.DataFrame:
        """Запускает полную симуляцию. Возвращает DataFrame с выходными параметрами."""
        # Устанавливаем seed для случайного метода (LHS использует свой seed внутри функции)
        np.random.seed(self.random_seed)
        self.input_samples_df = self._generate_input_samples()
        self.results_df = self._apply_linear_model(self.input_samples_df)
        return self.results_df

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Возвращает статистики для каждого выходного параметра:
        mean, median, std, p5, p50, p85, min, max
        """
        if self.results_df is None:
            raise RuntimeError("Run simulation first")
        stats = {}
        for col in self.results_df.columns:
            data = self.results_df[col]
            stats[col] = {
                "mean": np.mean(data),
                "median": np.median(data),
                "std": np.std(data, ddof=1),
                "p5": np.percentile(data, 5),
                "p50": np.percentile(data, 50),
                "p85": np.percentile(data, 85),
                "min": np.min(data),
                "max": np.max(data),
            }
        return stats

    def get_sensitivity(self) -> Dict[str, pd.DataFrame]:
        """
        Возвращает для каждого выхода DataFrame с ранговой корреляцией Спирмена
        между входными параметрами и данным выходом.
        """
        if self.results_df is None or self.input_samples_df is None:
            raise RuntimeError("Run simulation first")
        sensitivity = {}
        for out_name in self.results_df.columns:
            corr = []
            for inp_name in self.input_samples_df.columns:
                r, _ = spearmanr(self.input_samples_df[inp_name], self.results_df[out_name])
                corr.append({"input": inp_name, "spearman_corr": r})
            df = pd.DataFrame(corr).sort_values("spearman_corr", key=abs, ascending=False)
            sensitivity[out_name] = df
        return sensitivity