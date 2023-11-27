from pydantic import BaseModel, Field
from enum import Enum

class FilterNaem(str, Enum):
    canny = "canny"
    binary = "binary"

class InputModel(BaseModel):
    image_path: str = Field(
        title="Image Path",
        description="The path to the image to apply filter.",
        json_schema_extra={"from_upstream": "always"}
    )
    filter_name: FilterNaem = Field(
        title="Filter Name",
        description="The name of the filter to apply.",
        default="canny",
    )


class OutputModel(BaseModel):
    image_path: str = Field(
        title="Output image Path",
        description="The path to the image with the filter applied.",
    )
