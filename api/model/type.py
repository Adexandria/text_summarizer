from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ModelParams(BaseModel):
    name: Annotated[
        Literal["t5", "qwen"],
        Field(
            description="The model to use for prediction. Must be either 't5' or 'qwen'."
        ),
    ] = "t5"