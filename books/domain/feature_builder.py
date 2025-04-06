from typing import AsyncGenerator

import pandas as pd
from sentence_transformers import SentenceTransformer


class FeatureBuilder:
    def __init__(self, model: str):
        self._model = SentenceTransformer(model)

    async def build_features(
        self, sentences: AsyncGenerator[str, None]
    ) -> pd.DataFrame:
        vector_list = []
        async for sentence in sentences:
            vector = self._model.encode(sentence)
            vector_list.append(vector)
        features_df = pd.DataFrame(vector_list)
        return features_df
