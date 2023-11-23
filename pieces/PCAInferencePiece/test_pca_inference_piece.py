from domino.testing import piece_dry_run, skip_envs


@skip_envs("github")
def test_pca_inference_piece():
    print('Running pca inference piece test')