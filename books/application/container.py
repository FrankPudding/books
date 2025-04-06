from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer
from sentence_transformers import SentenceTransformer

from books.application.config import Config
from books.domain.feature_builder import FeatureBuilder
from books.domain.models.xgboost_model import XGBoostModel
from books.domain.preprocessor import Preprocessor
from books.domain.training_service import TrainingService
from books.infrastructure.model_registries.mlflow_model_regsitry import (
    MlflowModelRegsitry,
)
from books.infrastructure.repositories.jsonl_file_review_repository import (
    JsonlFileReviewRepository,
)


class Container(DeclarativeContainer):
    config: Config = providers.Configuration()

    jsonl_file_review_repository = providers.Factory(
        JsonlFileReviewRepository, filepath=config.reviews_jsonl_filepath
    )

    preprocessor = providers.Factory(Preprocessor)

    sentence_transformer = providers.Singleton(
        SentenceTransformer,
        model_name_or_path=config.sentence_transformer_model,
    )
    feature_builder = providers.Factory(
        FeatureBuilder, sentence_transformer=sentence_transformer
    )

    xgboost_model = providers.Factory(XGBoostModel)

    training_service = providers.Factory(
        TrainingService,
        review_repository=jsonl_file_review_repository,
        preprocessor=preprocessor,
        feature_builder=feature_builder,
        untrained_model=xgboost_model,
        test_split=config.test_split,
    )

    mlflow_model_regsitry = providers.Factory(
        MlflowModelRegsitry, tracking_uri=config.mlflow_tracking_uri
    )
