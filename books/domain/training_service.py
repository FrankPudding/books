from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split

from books.domain.entities.review import Review
from books.domain.feature_builder import FeatureBuilder
from books.domain.model import Model
from books.domain.model_regsitry import ModelRegistry
from books.domain.preprocessor import Preprocessor
from books.domain.repository import Repository


class TrainingService:
    def __init__(
        self,
        review_repository: Repository[Review],
        preprocessor: Preprocessor,
        feature_builder: FeatureBuilder,
        model_registry: ModelRegistry,
        test_split: Optional[float] = None,
    ):
        self._review_repository = review_repository
        self._preprocessor = preprocessor
        self._feature_builder = feature_builder
        self._model_regsitry = model_registry
        if test_split is None:
            test_split = 0.1
        self._test_split = test_split

    async def train_model(self, model: Model) -> str:
        reviews = self._review_repository.get_all_items()
        sentences, sentiments = await self._preprocessor.preprocess_reviews(
            reviews=reviews
        )

        features_df = await self._feature_builder.build_features(sentences)
        sentiments = np.fromiter(
            iter=[sentiment async for sentiment in sentiments], dtype=int
        )
        x_train, _, y_train, _ = train_test_split(
            features_df,
            sentiments,
            test_size=self._test_split,
            random_state=17,
        )
        model.fit(x_train, y_train)

        model_uri = await self._model_regsitry.log_model(model)

        return model_uri
