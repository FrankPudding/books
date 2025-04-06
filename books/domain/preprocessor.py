import aioitertools
from typing import AsyncGenerator, Tuple

from books.domain.entities.review import Review
from books.domain.entities.sentiment import Sentiment


class Preprocessor:

    async def preprocess_reviews(
        self, reviews: AsyncGenerator[Review, None]
    ) -> Tuple[AsyncGenerator[str, None], AsyncGenerator[int, None]]:
        reviews_1, reviews_2 = aioitertools.tee(reviews, n=2)
        sentences = self._generate_sentences(reviews=reviews_1)
        sentiments = self._generate_sentiments(reviews=reviews_2)
        return sentences, sentiments

    async def _generate_sentences(
        self, reviews: AsyncGenerator[Review, None]
    ) -> AsyncGenerator[str, None]:
        async for review in reviews:
            sentences = review.text.split(".")
            sentences = [
                sentence.strip() + "."
                for sentence in sentences
                if len(sentence.strip()) > 0
            ]
            for sentence in sentences:
                yield sentence

    async def _generate_sentiments(
        self, reviews: AsyncGenerator[Review, None]
    ) -> AsyncGenerator[str, None]:
        async for review in reviews:
            sentences = review.text.split(".")
            num_sentences = len(
                [0 for sentence in sentences if len(sentence.strip()) > 0]
            )

            sentiment = Sentiment.POSITIVE.value
            if review.rating == 3:
                sentiment = Sentiment.NEUTRAL.value
            elif review.rating < 3:
                sentiment = Sentiment.NEGATIVE.value

            for _ in range(num_sentences):
                yield sentiment
