import numpy as np
import pandas as pd
import pytest

from books.application.container import Container
from books.domain.models.xgboost_model import XGBoostModel


class TestXGBoostModel:
    @pytest.fixture()
    def sut(self, container: Container) -> XGBoostModel:
        return container.xgboost_model()

    def test_fits(self, sut: XGBoostModel):
        # arrange
        data = pd.DataFrame({"a": [0.2, 0.3], "b": [0.8, 0.1]})
        target = [1, 2]

        # act
        sut.fit(data=data, target=target)

    def test_predicts_after_fit(self, sut: XGBoostModel):
        # arrange
        data = pd.DataFrame({"a": [0.2, 0.3], "b": [0.8, 0.1]})
        target = [1, 2]
        sut.fit(data=data, target=target)

        # act
        result = sut.predict(data=data)

        # assert
        assert isinstance(result, np.ndarray)
        assert isinstance(result[0], np.int64)
        assert len(result) == 2

    def test_errors_on_double_fit(self, sut: XGBoostModel):
        # arrange
        data = pd.DataFrame({"a": [0.2, 0.3], "b": [0.8, 0.1]})
        target = [1, 2]
        sut.fit(data=data, target=target)

        # act & assert
        with pytest.raises(ValueError):
            sut.fit(data=data, target=target)

    def test_doesnt_predict_without_fit(self, sut: XGBoostModel):
        # arrange
        data = pd.DataFrame({"a": [0.2, 0.3], "b": [0.8, 0.1]})

        # act & assert
        with pytest.raises(ValueError):
            sut.predict(data)
