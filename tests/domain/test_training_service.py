import pytest

from books.domain.entities.review import Review
from books.domain.model import Model
from books.domain.repository import Repository
from books.domain.training_service import TrainingService
from books.infrastructure.jsonl_file_review_repository import (
    JsonlFileReviewRepository,
)
from tests import TEST_DATA_ROOT


class TestTrainingService:
    @pytest.fixture()
    def sut(self, review_repository: Repository[Review]) -> TrainingService:
        return TrainingService(review_repository=review_repository)

    @pytest.fixture()
    def review_repository(self) -> Repository[Review]:
        path = TEST_DATA_ROOT.joinpath("Books_sample.jsonl").as_posix()
        return JsonlFileReviewRepository(filepath=path)

    @pytest.mark.asyncio
    async def test_trains(self, sut: TrainingService):
        # act
        result = await sut.train_model()

        # assert
        assert isinstance(result, Model)
