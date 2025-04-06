from typing import List
import pytest

from books.application.container import Container
from books.domain.entities.sentiment import Sentiment
from books.domain.inference_service import InferenceService
from books.domain.models.xgboost_model import XGBoostModel
from books.domain.training_service import TrainingService


class TestInferenceService:
    @pytest.fixture()
    def sut(self, container: Container) -> InferenceService:
        return container.inference_service()

    @pytest.fixture()
    def training_service(self, container: Container) -> TrainingService:
        return container.training_service()

    @pytest.mark.asyncio
    async def test_predicts(
        self, sut: InferenceService, training_service: TrainingService
    ):
        # arrange
        model = XGBoostModel()
        model_uri = await training_service.train_model(model=model)
        sentences = ["This is bad", "This is worse", "This is good though"]

        # act
        result = await sut.predict_batch(
            model_uri=model_uri, sentences=sentences
        )

        # assert
        assert isinstance(result, List)
        assert len(result) == 3
        assert isinstance(result[0], Sentiment)
