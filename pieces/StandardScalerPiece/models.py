from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class OutputType(str, Enum):
    file = "file"
    object = "object"

class InputModel(BaseModel):
    train_data_path: str = Field(
        title="Train Data Path",
        description="The path to the train data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )
    test_data_path: str = Field(
        title="Test Data Path",
        description="The path to the test data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )


class OutputModel(BaseModel):
    train_data_path: str = None
    test_data_path: str = None