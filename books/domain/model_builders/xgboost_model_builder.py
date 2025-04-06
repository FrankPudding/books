from books.domain.model_builder import ModelBuilder
from books.domain.models.xgboost_model import XGBoostModel


class XGBoostModelBuilder(ModelBuilder):
    def build_model(self) -> XGBoostModel:
        return XGBoostModel()
