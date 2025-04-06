from abc import ABC
from typing import AsyncGenerator


class Repository[T](ABC):
    async def get_all_items(self) -> AsyncGenerator[T, None]:
        raise NotImplementedError()

    async def save_item(self, item: T) -> None:
        raise NotImplementedError()

    async def load_item(sef, item_id: str) -> T:
        raise NotImplementedError()
