from domino.testing import piece_dry_run, skip_envs
import pandas as pd

@skip_envs("github")
def test_train_test_split_piece():
    test_data_size = 0.3
    input_data = dict(
        data_path="tests_data/breast.csv",
        test_data_size=0.3,
        random_state=42
    )
    output_data = piece_dry_run("TrainTestSplitPiece", input_data)
    assert output_data.get('train_data_path') is not None
    assert output_data.get('test_data_path') is not None

    df_input = pd.read_csv(input_data.get('data_path'))
    df_train = pd.read_csv(output_data.get('train_data_path'))
    df_test = pd.read_csv(output_data.get('test_data_path'))

    expected_len_train = int(len(df_input) * (1 - test_data_size))
    assert len(df_train) == expected_len_train
    assert len(df_test) == len(df_input) - expected_len_train