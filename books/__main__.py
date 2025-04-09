import asyncio
from typing import Optional
import click

from books.application.config import Config
from books.application.container import Container
from books.domain.models.xgboost_model import XGBoostModel
from tests import TEST_DATA_ROOT


@click.group()
def cli():
    pass


@cli.command()
def train():
    container = Container()
    config = Config(
        reviews_jsonl_filepath=TEST_DATA_ROOT.joinpath(
            "Books_sample.jsonl"
        ).as_posix()
    )
    container.config.from_pydantic(config)
    training_service = container.training_service()
    model = XGBoostModel()
    model_uri = asyncio.run(training_service.train_model(model))
    print(f"Model URI: {model_uri}")


@cli.command()
@click.option("--text", help="The text to classify")
@click.option(
    "--model-uri", help="The ID of the model you want to predict with"
)
def predict(text: str, model_uri: str):
    container = Container()
    config = Config()
    container.config.from_pydantic(config)
    inference_service = container.inference_service()
    result = asyncio.run(
        inference_service.predict_batch(model_uri=model_uri, sentences=[text])
    )[0]
    print(result)


@cli.command()
@click.option("--text", help="The text to classify")
@click.option(
    "--model-uri",
    help="The ID of the model you want to predict with",
    default=None,
)
def predict(text: str, model_uri: Optional[str]):
    print("Warming up...")
    container = Container()
    config = Config()
    container.config.from_pydantic(config)
    inference_service = container.inference_service()
    print("Predicting...")
    result = asyncio.run(
        inference_service.predict_batch(model_uri=model_uri, sentences=[text])
    )[0]
    print(f"Prediction: {result}")


if __name__ == "__main__":
    cli()
