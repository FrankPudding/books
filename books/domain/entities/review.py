from pydantic import BaseModel


class Review(BaseModel):
    rating: int
    text: str
