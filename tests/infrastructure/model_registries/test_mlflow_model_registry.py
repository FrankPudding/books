import asyncio

import pandas as pd
import pytest

from books.application.container import Container
from books.domain.models.xgboost_model import XGBoostModel
from books.infrastructure.model_registries.mlflow_model_regsitry import (
    MlflowModelRegsitry,
)


class TestMlflowModelRegsitry:
    @pytest.fixture()
    def sut(self, container: Container) -> MlflowModelRegsitry:
        return container.mlflow_model_registry()

    @pytest.mark.asyncio
    async def test_logs_model(self, sut: MlflowModelRegsitry):
        # arrange
        data = pd.DataFrame({"a": [0.2, 0.3], "b": [0.8, 0.1]})
        target = [1, 2]

        model = XGBoostModel()
        model.fit(data=data, target=target)

        # act
        await sut.log_model(model=model)

    @pytest.mark.asyncio
    async def test_loads_model_after_logging(self, sut: MlflowModelRegsitry):
        # arrange
        data = pd.DataFrame({"a": [0.2, 0.3], "b": [0.8, 0.1]})
        target = [1, 2]

        model = XGBoostModel()
        model.fit(data=data, target=target)
        model_uri = await sut.log_model(model=model)

        # act
        result = await sut.load_model(model_uri=model_uri)
        asyncio.sleep(1)

        # assert
        test_data = pd.DataFrame({"a": [0.8, 0.1], "b": [0.2, 0.3]})
        assert all(
            model.predict(data=test_data) == result.predict(data=test_data)
        )
