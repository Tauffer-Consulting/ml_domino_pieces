from pydantic import BaseModel, Field



class InputModel(BaseModel):
    data_path: str = Field(
        title="Data path",
        description="Data path to inference on.",
        json_schema_extra={"from_upstream": "always"}
    )
    model_path: str = Field(
        title="Model path",
        description="Path to the model to use for inference.",
        json_schema_extra={"from_upstream": "always"}
    )


class OutputModel(BaseModel):
    data_path: str
