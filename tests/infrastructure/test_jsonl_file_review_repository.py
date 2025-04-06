from typing import AsyncGenerator
import pytest

from books.domain.entities.review import Review
from books.infrastructure.jsonl_file_review_repository import JsonlFileReviewRepository
from tests import TEST_DATA_ROOT


class TestJsonlFileReviewRepository:
    @pytest.fixture()
    def sut(self) -> JsonlFileReviewRepository:
        return JsonlFileReviewRepository(
            filepath=TEST_DATA_ROOT.joinpath("Books_sample.jsonl").as_posix()
        )

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

    @pytest.mark.asyncio
    async def test_dummy(self, sut: JsonlFileReviewRepository):
        from sklearn.feature_extraction.text import TfidfVectorizer

        result = sut.get_all_items()
        texts = [review.text async for review in result]
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(texts)
        print(x)
