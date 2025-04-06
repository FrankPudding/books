from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

from books import RESOURCES_ROOT


class Config(BaseSettings):
    reviews_jsonl_filepath: str = Field(
        default_factory=lambda _: RESOURCES_ROOT.joinpath(
            "data/Books_10k.jsonl"
        ).as_posix()
    )
    sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2")
    test_split: Optional[float] = None
