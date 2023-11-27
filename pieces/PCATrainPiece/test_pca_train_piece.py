from domino.testing import piece_dry_run, skip_envs
import pandas as pd

@skip_envs("github")
def test_pca_train_piece():
    input_data = dict(
        data_path="tests_data/breast.csv",
        n_components=2,
        use_class_column=False,
    )
    output_data = piece_dry_run("PCATrainPiece", input_data)
    assert output_data.get('pca_data_path') is not None
    assert output_data.get('pca_model_path') is not None

    # Open and check n rows
    df_input = pd.read_csv(input_data.get('data_path'))
    df = pd.read_csv(output_data.get('pca_data_path'))
    assert len(df) == len(df_input)