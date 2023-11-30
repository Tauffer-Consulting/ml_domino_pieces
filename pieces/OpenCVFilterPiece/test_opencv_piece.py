from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs
import pandas as pd


@skip_envs("github")
def test_pca_inference_piece():
    input_data = dict(
        image_path="./pieces/OpenCVFilterPiece/earth.jpg",
        filter_name="binary",
    )
    output_data = piece_dry_run("OpenCVFilterPiece", input_data)
