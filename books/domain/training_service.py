import numpy as np
from sklearn.model_selection import train_test_split

from books.domain.entities.review import Review
from books.domain.feature_builder import FeatureBuilder
from books.domain.model_builders.xgboost_model_builder import (
    XGBoostModelBuilder,
)
from books.domain.preprocessor import Preprocessor
from books.domain.repository import Repository


class TrainingService:
    def __init__(
        self, review_repository: Repository[Review], batch_size: int = 1000
    ):
        self._review_repository = review_repository
        self._model_builder = XGBoostModelBuilder()
        self._preprocessor = Preprocessor()
        self._feature_builder = FeatureBuilder()
        self._batch_size = batch_size

    async def train_model(self):
        reviews = self._review_repository.get_all_items()
        sentences, sentiments = await self._preprocessor.preprocess_reviews(
            reviews=reviews
        )

        x_df = await self._feature_builder.build_features(sentences)
        sentiments = np.fromiter(
            iter=[sentiment async for sentiment in sentiments], dtype=int
        )
        model = self._model_builder.build_model()
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, sentiments, test_size=0.1, random_state=17
        )
        model.fit(x_train, y_train)

        return model
