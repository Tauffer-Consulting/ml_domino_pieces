from pydantic import BaseModel, Field


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
    train_data_path: str
    test_data_path: str