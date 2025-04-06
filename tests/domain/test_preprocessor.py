from typing import AsyncGenerator
import pytest

from books.application.container import Container
from books.domain.entities.review import Review
from books.domain.entities.sentiment import Sentiment
from books.domain.preprocessor import Preprocessor


class TestPreprocessor:
    @pytest.fixture()
    def sut(self, container: Container) -> Preprocessor:
        return container.preprocessor()

    @pytest.mark.asyncio
    async def test_preprocesses(self, sut: Preprocessor):
        # arrange
        async def _generate_reviews():
            reviews = [
                Review(rating=1.0, text="This is really bad. So bad"),
                Review(rating=2.0, text="This is quite bad"),
                Review(rating=3.0, text="This is alright. I suppose"),
                Review(rating=4.0, text="This is quite good"),
                Review(rating=5.0, text="This is really good. So good. Wow."),
            ]
            for review in reviews:
                yield review

        reviews = _generate_reviews()

        # act
        sentences_result, sentiment_result = await sut.preprocess_reviews(
            reviews=reviews
        )

        # assert
        assert isinstance(sentences_result, AsyncGenerator)
        assert isinstance(sentiment_result, AsyncGenerator)
        sentences_list = [sentence async for sentence in sentences_result]
        sentiment_list = [sentiment async for sentiment in sentiment_result]
        assert len(sentences_list) == 9
        assert sentences_list[0] == "This is really bad."
        assert sentences_list[4] == "I suppose."
        assert sentences_list[-1] == "Wow."
        assert (
            sentiment_list
            == [Sentiment.NEGATIVE.value] * 3
            + [Sentiment.NEUTRAL.value] * 2
            + [Sentiment.POSITIVE.value] * 4
        )
