from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots
import cv2
from pathlib import Path


class OpenCVFilterPiece(BasePiece):

    def piece_function(self, input_data: InputModel):
        image = cv2.imread(input_data.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if input_data.filter_name == "canny":
            edged = cv2.Canny(gray, 50, 100)
        elif input_data.filter_name == "binary":
            _, edged = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unknown filter name: {input_data.filter_name}")

        output_image_path = Path(self.results_path) / "opencv_image.png"
        cv2.imwrite(str(output_image_path), edged)
        self.display_result = {
            "file_type": "png",
            "file_path": str(output_image_path)
        }

        return OutputModel(
            image_path=str(output_image_path)
        )


