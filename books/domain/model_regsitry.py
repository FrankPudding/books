import abc

from books.domain.model import Model


class ModelRegistry(abc.ABC):
    async def log_model(self, model: Model) -> str:
        """
        Returns model_id
        """
        raise NotImplementedError()

    async def load_model(self, model_id: str) -> Model:
        raise NotImplementedError()
