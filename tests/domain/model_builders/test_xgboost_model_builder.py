import pytest

from books.application.container import Container
from books.domain.model_builders.xgboost_model_builder import (
    XGBoostModelBuilder,
)
from books.domain.models.xgboost_model import XGBoostModel


class TestXGBoostModelBuilder:
    @pytest.fixture()
    def sut(self, container: Container) -> XGBoostModelBuilder:
        return container.xgboost_model_builder()

    def test_builds_model(self, sut: XGBoostModelBuilder):
        # act
        result = sut.build_model()

        # assert
        assert isinstance(result, XGBoostModel)
