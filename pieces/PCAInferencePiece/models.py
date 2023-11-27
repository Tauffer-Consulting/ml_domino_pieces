from pydantic import BaseModel, Field


class InputModel(BaseModel):
    model_path: str = Field(
        title="Model Path",
        description="The path to the PCA model.",
        json_schema_extra={"from_upstream": "always"}
    )
    data_path: str = Field(
        title="Data Path",
        description="The path to the train data to apply PCA.",
        json_schema_extra={"from_upstream": "always"}
    )
    use_class_column: bool = Field(
        default=False,
        description="Whether to use the target column as class to plot.",
        title="Use Class Column",
    )


class OutputModel(BaseModel):
    pca_data_path: str = None
