import pytest

from books.application.container import Container
from books.domain.models.xgboost_model import XGBoostModel
from books.domain.training_service import TrainingService


class TestTrainingService:
    @pytest.fixture()
    def sut(self, container: Container) -> TrainingService:
        return container.training_service()

    @pytest.mark.asyncio
    async def test_trains_xgboost(self, sut: TrainingService):
        # arrange
        model = XGBoostModel()

        # act
        result = await sut.train_model(model=model)

        # assert
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_can_load_model_id(
        self, sut: TrainingService, container: Container
    ):
        # arrange
        model = XGBoostModel()
        model_registry = container.mlflow_model_registry()

        # act
        model_id = await sut.train_model(model=model)
        result = await model_registry.load_model(model_id=model_id)

        # assert
        assert isinstance(result, XGBoostModel)
