import pandas as pd

class MonteCarloEngine:
    def __init__(self, config: dict):
        self.config = config
        self.results = None

    def run(self) -> pd.DataFrame:
        # временная заглушка
        return pd.DataFrame()
