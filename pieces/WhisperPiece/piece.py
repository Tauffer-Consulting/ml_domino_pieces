from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import whisper


class WhisperPiece(BasePiece):

    def piece_function(self, input_data: InputModel):

        self.logger.info("Loading model...")
        model = whisper.load_model(input_data.model_size)

        self.logger.info("Transcribing audio file...")
        result = model.transcribe(str(input_data.file_path))["text"]

        if input_data.output_type == "xcom":
            self.logger.info("Transcription complete successfully. Result returned as XCom.")
            msg = f"Transcription complete successfully. Result returned as XCom."
            transcription_result = result
            output_file_path = ""
        else:
            self.logger.info("Transcription complete successfully. Result returned as file.")
            msg = f"Transcription complete successfully. Result returned as file."
            transcription_result = ""
            output_file_path = "transcription_result.txt"
            with open(output_file_path, "w") as f:
                f.write(result)

        return OutputModel(
            message=msg,
            transcription_result=transcription_result,
            file_path=output_file_path
        )