from typing import List
import pandas as pd
import xgboost
from books.domain.model import Model


class XGBoostModel(Model):
    def __init__(self):
        self._model = None

    def fit(self, data: pd.DataFrame, target: List[int]):
        if self._model is not None:
            raise ValueError("Model already trained")
        dtrain = xgboost.DMatrix(data=data, label=target)
        self._model = xgboost.train(params={}, dtrain=dtrain)

    def predict(self, data: pd.DataFrame) -> List[int]:
        if self._model is None:
            raise ValueError("Model has not yet been trained")
        return self._model.predict(data=xgboost.DMatrix(data=data))
