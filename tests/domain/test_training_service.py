import pytest

from books.application.container import Container
from books.domain.model import Model
from books.domain.training_service import TrainingService


class TestTrainingService:
    @pytest.fixture()
    def sut(self, container: Container) -> TrainingService:
        return container.training_service()

    @pytest.mark.asyncio
    async def test_trains(self, sut: TrainingService):
        # act
        result = await sut.train_model()

        # assert
        assert isinstance(result, Model)
