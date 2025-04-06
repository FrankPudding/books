import abc
from typing import List

import pandas as pd


class Model(abc.ABC):
    def fit(self, data: pd.DataFrame, target: List[int]):
        raise NotImplementedError()

    def predict(self, data: pd.DataFrame) -> List[int]:
        raise NotImplementedError()
