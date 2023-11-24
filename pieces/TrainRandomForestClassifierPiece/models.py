from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Criterion(str, Enum):
    gini = "gini"
    entropy = "entropy"
    log_loss = "log_loss"

class InputModel(BaseModel):
    train_data_path: str = Field(
        title="Train Data Path",
        description="The path to the train data to train the data.",
        json_schema_extra={"from_upstream": "always"}
    )

    n_estimators: int = Field(
        title="Number of Estimators",
        description="The number of trees in the forest.",
        default=100,
    )

    criterion: Criterion = Field(
        title="Criterion",
        description="The function to measure the quality of a split.",
        default=Criterion.gini,
    )
    max_depth: Optional[int] = Field(
        title="Max Depth",
        description="The maximum depth of the tree.",
        default=None,
    )

    ## other args?


class OutputModel(BaseModel):
    model_path: str
