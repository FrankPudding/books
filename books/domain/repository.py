from abc import ABC
from typing import AsyncGenerator


class Repository[T](ABC):
    async def get_all_items() -> AsyncGenerator[T, None]:
        raise NotImplementedError()
