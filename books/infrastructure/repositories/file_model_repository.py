import mlflow
from books.domain.model import Model
from books.domain.repository import Repository


class FileModelRepository(Repository[Model]):
    def __init__(self, base_directory: str):
        mlflow.set_tracking_uri(f"file:///{base_directory}")

    async def save_item(self, item: Model):
        return await super().save_item(item)

    async def load_item(sef, item_id: str) -> Model:
        return await super().load_item(item_id)
