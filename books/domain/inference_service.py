from typing import AsyncGenerator, List

from books.domain.entities.sentiment import Sentiment
from books.domain.feature_builder import FeatureBuilder
from books.domain.model_regsitry import ModelRegistry


class InferenceService:
    def __init__(
        self, model_registry: ModelRegistry, feature_builder: FeatureBuilder
    ):
        self._model_registry = model_registry
        self._feature_builder = feature_builder

    async def predict_batch(
        self, model_uri: str, sentences: List[str]
    ) -> List[Sentiment]:
        model = await self._model_registry.load_model(model_uri=model_uri)

        async def _generate_sentences() -> AsyncGenerator[str, None]:
            for sentence in sentences:
                yield sentence

        async_sentences = _generate_sentences()

        features_df = await self._feature_builder.build_features(
            sentences=async_sentences
        )
        results = model.predict(features_df)
        predictions = [Sentiment(int(result)) for result in results]
        return predictions
