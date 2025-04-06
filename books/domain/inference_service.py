from typing import AsyncGenerator, List

import numpy as np
from books.domain.feature_builder import FeatureBuilder
from books.domain.model_regsitry import ModelRegistry


class InferenceService:
    def __init__(
        self, model_registry: ModelRegistry, feature_builder: FeatureBuilder
    ):
        self._model_registry = model_registry
        self._feature_builder = feature_builder

    async def predict_batch(
        self, model_id: str, sentences: List[str]
    ) -> np.ndarray[np.int64]:
        model = await self._model_registry.load_model(model_id=model_id)

        async def _generate_sentences() -> AsyncGenerator[str, None]:
            for sentence in sentences:
                yield sentence

        async_sentences = _generate_sentences()

        features_df = await self._feature_builder.build_features(
            sentences=async_sentences
        )
        result = model.predict(features_df)
        return result
