import abc

from books.domain.model import Model


class ModelBuilder(abc.ABC):
    def build_model(self) -> Model:
        raise NotImplementedError()
