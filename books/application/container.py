from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer

from books.application.config import Config
from books.domain.feature_builder import FeatureBuilder
from books.domain.model_builders.xgboost_model_builder import (
    XGBoostModelBuilder,
)
from books.domain.preprocessor import Preprocessor
from books.domain.training_service import TrainingService
from books.infrastructure.jsonl_file_review_repository import (
    JsonlFileReviewRepository,
)


class Container(DeclarativeContainer):
    config: Config = providers.Configuration()

    review_repository = providers.Factory(
        JsonlFileReviewRepository, filepath=config.reviews_jsonl_filepath
    )

    preprocessor = providers.Factory(Preprocessor)

    feature_builder = providers.Factory(
        FeatureBuilder, model=config.feature_builder_model
    )

    model_builder = providers.Factory(XGBoostModelBuilder)

    training_service = providers.Factory(
        TrainingService,
        review_repository=review_repository,
        preprocessor=preprocessor,
        feature_builder=feature_builder,
        model_builder=model_builder,
        test_split=config.test_split,
    )
