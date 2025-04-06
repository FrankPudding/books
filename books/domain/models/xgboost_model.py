from typing import List

import numpy as np
from sklearn.calibration import LabelEncoder
import pandas as pd
import xgboost

from books.domain.model import Model


class XGBoostModel(Model):
    def __init__(self):
        self._model = None
        self._label_encoder = LabelEncoder()

    def fit(self, data: pd.DataFrame, target: List[int]):
        if self._model is not None:
            raise ValueError("Model already trained")
        transformed_target = self._label_encoder.fit_transform(target)
        self._model = xgboost.XGBClassifier()
        self._model.fit(data, transformed_target)

    def predict(self, data: pd.DataFrame) -> np.ndarray[np.int64]:
        if self._model is None:
            raise ValueError("Model has not yet been trained")
        prediction = self._model.predict(X=data)
        untransformed_prediction = self._label_encoder.inverse_transform(
            prediction
        )
        return untransformed_prediction
