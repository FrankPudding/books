from pathlib import Path
import pickle
import tempfile
from typing import Dict, Optional

import mlflow

from books.domain.model import Model
from books.domain.model_regsitry import ModelRegistry
from books.domain.models.xgboost_model import XGBoostModel


class MlflowModelRegsitry(ModelRegistry):
    def __init__(self, tracking_uri: str):
        self._tracking_uri = tracking_uri

    async def log_model(
        self,
        model: Model,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        mlflow.set_tracking_uri(self._tracking_uri)
        with mlflow.start_run() as run:
            model_uri = run.info.artifact_uri
            if isinstance(model, XGBoostModel):
                mlflow.xgboost.log_model(
                    model.classifier,
                    artifact_path="xgboost",
                )
                model_uri = f"{model_uri}/xgboost"
                client = mlflow.MlflowClient()
                with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
                    pickle.dump(model.label_encoder, tmp)
                    tmp.flush()
                    client.log_artifact(
                        run_id=run.info.run_id,
                        local_path=tmp.name,
                        artifact_path="label_encoder",
                    )
            else:
                raise ValueError("Unrecognised model type")

            if metrics:
                mlflow.log_metrics(metrics=metrics)

            return model_uri

    async def load_model(self, model_uri: str) -> Model:
        mlflow.set_tracking_uri(self._tracking_uri)
        if model_uri.endswith("/xgboost"):
            return self._load_xgboost_model(model_uri=model_uri)
        else:
            raise ValueError("Unrecognised model_uri")

    def _load_xgboost_model(self, model_uri: str) -> Model:
        classifier = mlflow.xgboost.load_model(model_uri)
        client = mlflow.MlflowClient()
        run_id = Path(model_uri).parents[1].stem
        with tempfile.TemporaryDirectory() as tmpdir:
            client.download_artifacts(
                run_id=run_id, path="label_encoder", dst_path=tmpdir
            )
            encoder_dir = Path(tmpdir).joinpath("label_encoder")
            encoder_filepath = next(encoder_dir.glob("*")).as_posix()
            with open(encoder_filepath, "rb") as encoder_file:
                unpickler = pickle.Unpickler(encoder_file)
                label_encoder = unpickler.load()
        model = XGBoostModel(
            classifier=classifier, label_encoder=label_encoder
        )
        return model
