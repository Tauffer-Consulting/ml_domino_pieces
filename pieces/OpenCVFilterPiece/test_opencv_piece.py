from domino.testing import piece_dry_run, skip_envs
import pandas as pd


@skip_envs("github")
def test_pca_inference_piece():
    input_data = dict(
        image_path="/home/vinicius/Documents/work/tauffer/ml_domino_pieces/pieces/OpenCVSegmentationPiece/earth.jpg",
        filter_name="canny",
    )
    output_data = piece_dry_run("OpenCVFilterPiece", input_data)