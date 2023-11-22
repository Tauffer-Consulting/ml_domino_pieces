from pydantic import BaseModel, Field
from typing import List


class InputModel(BaseModel):
    """
    Input data for TextSummarizerPiece
    """
    train_data: List[dict] = Field(
        title="Train Data",
        description="The train data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )
    test_data: List[dict] = Field(
        title="Test Data",
        description="The test data to be scaled.",
        json_schema_extra={"from_upstream": "always"}
    )


class OutputModel(BaseModel):
    """
    Output data for TextSummarizerPiece
    """
    train_data: List[dict]
    test_data: List[dict]