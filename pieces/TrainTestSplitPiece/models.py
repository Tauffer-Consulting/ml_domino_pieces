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
    data: Optional[List[dict]] = Field(
        title="Data",
        default=None,
        description="The data to be split.",
        json_schema_extra={"from_upstream": "always"}
    )
    data_path: Optional[str] = Field(
        title="Data Path",
        default=None,
        description="The path to the data to be split.",
        json_schema_extra={"from_upstream": "always"}
    )
    test_data_size: float = Field(
        default=0.8,
        description="The size (%) of the test data.",
        title="Test Data Size",
    )
    random_state: int = Field(
        default=42,
        description="The random state for the split.",
        title="Random State",
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