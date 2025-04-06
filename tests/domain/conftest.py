import pytest

from books.application.config import Config
from books.application.container import Container
from tests import TEST_DATA_ROOT


@pytest.fixture(scope="session")
def config():
    return Config(
        reviews_jsonl_filepath=TEST_DATA_ROOT.joinpath(
            "Books_sample.jsonl"
        ).as_posix()
    )


@pytest.fixture(scope="session")
def container(config: Config):
    container = Container()
    container.config.from_pydantic(config)
    return container
