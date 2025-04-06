from pydantic import BaseModel, Field


class Review(BaseModel):
    rating: int = Field(ge=1, le=5)
    text: str
