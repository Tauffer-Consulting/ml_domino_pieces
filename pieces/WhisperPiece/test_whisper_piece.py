from domino.testing import piece_dry_run
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def run_piece(file_path: str, model_size: str):

    return piece_dry_run(
        #name of the piece
        piece_name="WhisperPiece",

        #values to the InputModel arguments
        input_data={
            "file_path": file_path,
            "model_size": model_size,
        },
    )

def test_whisper_piece():
    output = run_piece(
        file_path=f"{dir_path}/i-am-an-audio-to-transcribe.mp3",
        model_size="tiny"
    )

    assert "message" in output.keys()
    assert "transcription_result" in output.keys()
    assert "file_path" in output.keys()
    assert output.get("transcription_result") == " I am An audio to transcribe."