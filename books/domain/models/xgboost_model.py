from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from xgboost import XGBClassifier

from books.domain.model import Model


class XGBoostModel(Model):
    def __init__(
        self,
        classifier: Optional[XGBClassifier] = None,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        self.classifier = classifier
        self.label_encoder = label_encoder or LabelEncoder()

    def fit(self, data: pd.DataFrame, target: List[int]):
        if self.classifier is not None:
            raise ValueError("Model already trained")
        transformed_target = self.label_encoder.fit_transform(target)
        self.classifier = XGBClassifier()
        self.classifier.fit(data, transformed_target)

    def predict(self, data: pd.DataFrame) -> np.ndarray[np.int64]:
        if self.classifier is None:
            raise ValueError("Model has not yet been trained")
        prediction = self.classifier.predict(X=data)
        untransformed_prediction = self.label_encoder.inverse_transform(
            prediction
        )
        return untransformed_prediction
