from pydantic import BaseModel, Field


class InputModel(BaseModel):
    data_path: str = Field(
        title="Train Data Path",
        description="The path to the train data to apply PCA.",
        json_schema_extra={"from_upstream": "always"}
    )
    n_components: int = Field(
        default=2,
        description="The number of components for PCA.",
        title="Number of Components",
    )
    use_class_column: bool = Field(
        default=False,
        description="Whether to use the target column as class to plot.",
        title="Use Class Column",
    )


class OutputModel(BaseModel):
    tsne_data_path: str = None

