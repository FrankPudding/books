from typing import AsyncGenerator
import pytest

from books.application.container import Container
from books.domain.entities.review import Review
from books.infrastructure.repositories.jsonl_file_review_repository import (
    JsonlFileReviewRepository,
)


class TestJsonlFileReviewRepository:
    @pytest.fixture()
    def sut(self, container: Container) -> JsonlFileReviewRepository:
        return container.jsonl_file_review_repository()

    @pytest.mark.asyncio
    async def test_gets_all_items(self, sut: JsonlFileReviewRepository):
        # act
        result = sut.get_all_items()

        # assert
        assert isinstance(result, AsyncGenerator)
        result_list = [review async for review in result]
        assert len(result_list) == 10
        for review in result_list:
            assert isinstance(review, Review)

    def test_doesnt_allow_non_jsonl_files(self):
        # act
        with pytest.raises(ValueError):
            JsonlFileReviewRepository(filepath="something.json")
