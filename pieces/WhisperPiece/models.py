from pydantic import BaseModel, Field, FilePath
from typing import Union
from enum import Enum


class ModelSizeType(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


class OutputTypeType(str, Enum):
    xcom = "xcom"
    file = "file"


class InputModel(BaseModel):
    """
    Input data for WhisperPiece
    """
    file_path: str = Field(
        description='The path to the text file to process.',
        required=True
    )
    model_size: ModelSizeType = Field(
        description='The size of the model to use.',
        default=ModelSizeType.base
    )
    output_type: OutputTypeType = Field(
        description='The type of output fot the result text.',
        default=OutputTypeType.xcom
    )


class OutputModel(BaseModel):
    """
    Output data for WhisperPiece
    """
    message: str = Field(
        default="",
        description="Output message to log."
    )
    transcription_result: str = Field(
        default="",
        description="The result transcription text."
    )
    file_path: Union[FilePath, str] = Field(
        default="",
        description="The path to the results text file."
    )