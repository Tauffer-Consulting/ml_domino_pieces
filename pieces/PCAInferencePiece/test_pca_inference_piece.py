from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs
import pandas as pd


@skip_envs("github")
def test_pca_inference_piece():
    input_data = dict(
        data_path="tests_data/breast.csv",
        n_components=2,
        use_class_column=False,
        model_path='tests_data/pca_model.pkl'
    )
    output_data = piece_dry_run("PCAInferencePiece", input_data)
    assert output_data.get('pca_data_path') is not None

    # Open and check n rows
    df_input = pd.read_csv(input_data.get('data_path'))
    df = pd.read_csv(output_data.get('pca_data_path'))
    assert len(df) == len(df_input)