from pydantic import BaseModel, Field


class InputModel(BaseModel):
    data_path: str = Field(
        title="Data Path",
        description="The path to the data to be split.",
        json_schema_extra={"from_upstream": "always"}
    )
    test_data_size: float = Field(
        default=0.2,
        description="The size (%) of the test data.",
        title="Test Data Ratio",
    )
    random_state: int = Field(
        default=42,
        description="The random state for the split.",
        title="Random State",
    )


class OutputModel(BaseModel):
    train_data_path: str
    test_data_path: str