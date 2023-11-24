from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from pathlib import Path
import plotly.express as px
import pickle as pk
from sklearn.ensemble import RandomForestClassifier


class InferenceModelPiece(BasePiece):

    def read_data_from_file(self, path):
        """
        Read data from a file.
        """
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".json"):
            return pd.read_json(path)
        else:
            raise ValueError("File type not supported.")

    def piece_function(self, input_data: InputModel):

        # Read data
        data = self.read_data_from_file(input_data.data_path)

        # Train model
        with open(input_data.model_path, "rb") as f:
            model = pk.load(f)

        # Make predictions
        if 'target' in data:
            predictions = model.predict(data.drop('target', axis=1))
        else:
            predictions = model.predict_proba(data)

        # Save predictions
        predictions_path = str(Path(self.results_path) / "predictions.csv")
        data['predictions'] = predictions
        data.to_csv(predictions_path, index=False)

        # Plot predictions
        self.display_result = {
            'file_type': 'md',
            'file_path': predictions_path
        }

        return OutputModel(data_path=predictions_path)
