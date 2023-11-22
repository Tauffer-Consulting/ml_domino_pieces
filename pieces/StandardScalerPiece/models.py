from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class OutputType(str, Enum):
    file = "file"
    object = "object"

class InputModel(BaseModel):
    """
    Input data for TextSummarizerPiece
    """
    train_data: Optional[List[dict]] = Field(
        title="Train Data",
        default=None,
        description="The train data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )
    test_data: Optional[List[dict]] = Field(
        title="Test Data",
        default=None,
        description="The test data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )
    train_data_path: Optional[str] = Field(
        title="Train Data Path",
        default=None,
        description="The path to the train data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )
    test_data_path: Optional[str] = Field(
        title="Test Data Path",
        default=None,
        description="The path to the test data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )

    output_type: OutputType = Field(
        default=OutputType.object,
        description="The output type. Use file for large datasets.",
        title="Output Type",
    )


class OutputModel(BaseModel):
    """
    Output data for TextSummarizerPiece
    """
    train_data: Optional[List[dict]] = None
    test_data: Optional[List[dict]] = None

    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None