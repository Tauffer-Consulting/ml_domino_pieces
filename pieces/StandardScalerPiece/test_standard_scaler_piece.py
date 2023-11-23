from domino.testing import piece_dry_run, skip_envs
import pandas as pd

@skip_envs("github")
def test_standard_scaler_piece():
    input_data = dict(
        train_data_path="tests_data/breast.csv",
        test_data_path="tests_data/breast.csv",
    )
    output_data = piece_dry_run("StandardScalerPiece", input_data)
    assert output_data.get('train_data_path') is not None

    # Open and check n rows
    df_input = pd.read_csv(input_data.get('train_data_path'))
    df = pd.read_csv(output_data.get('train_data_path'))
    assert len(df) == len(df_input)