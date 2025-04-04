import json
from pathlib import Path
from typing import AsyncGenerator

import aiofiles

from books.domain.data_repository import Repository
from books.domain.entities.review import Review


class JsonlFileReviewRepository(Repository[Review]):
    def __init__(self, filepath: str):
        if not filepath.endswith(".jsonl"):
            raise ValueError(
                "JsonlFileReviewRepository only compatible with .jsonl files"
            )
        self._filepath = Path(filepath)

    async def get_all_items(self) -> AsyncGenerator[Review, None]:
        async with aiofiles.open(self._filepath, "r") as jsonl_file:
            async for line in jsonl_file:
                json_object = json.loads(line)
                yield Review(**json_object)
