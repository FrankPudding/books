import pandas as pd
import pytest

from books.application.container import Container
from books.domain.feature_builder import FeatureBuilder


class TestFeatureBuilder:
    @pytest.fixture()
    def sut(self, container: Container) -> FeatureBuilder:
        return container.feature_builder()

    @pytest.mark.asyncio
    async def test_builds_features(self, sut: FeatureBuilder):
        # arrange
        async def _generate_sentences():
            sentences = [
                "Is this the real life",
                "Is this just fantasy",
                "Caught in a landslide",
                "No escape from reality",
            ]
            for sentence in sentences:
                yield sentence

        sentences = _generate_sentences()

        # act
        result = await sut.build_features(sentences=sentences)

        # assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
