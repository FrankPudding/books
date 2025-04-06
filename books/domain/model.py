from typing import List

import pandas as pd


class Model:
    def fit(self, data: pd.DataFrame, target: List[int]):
        raise NotImplementedError()

    def predict(self, data: pd.DataFrame) -> List[int]:
        raise NotImplementedError()
