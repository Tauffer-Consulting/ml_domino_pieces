from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from pathlib import Path
import plotly.express as px
import pickle as pk
from sklearn.ensemble import RandomForestClassifier


class TrainRandomForestClassifierPiece(BasePiece):

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
        train_data = self.read_data_from_file(input_data.train_data_path)

        # Train model
        model = RandomForestClassifier(
            n_estimators=input_data.n_estimators,
            criterion=input_data.criterion,
            max_depth=input_data.max_depth,
        )
        model.fit(train_data.drop(columns=["target"], axis=1), train_data["target"])

        feature_imp = pd.Series(
            model.feature_importances_,
            index=train_data.drop('target', axis=1).columns
        ).sort_values(ascending=True)

        fig = px.bar(x=feature_imp.values, y=feature_imp.index, orientation='h')
        fig.update_layout(
            xaxis_title='Feature Importance Score',
            yaxis_title='Features',
            title='Feature Importance',
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )

        fig_path = str(Path(self.results_path) / "feature_importance.json")
        fig.write_json(fig_path)

        self.display_result = {
            'file_type': 'plotly_json',
            'file_path': fig_path
        }

        model_path = str(Path(self.results_path) / "random_forest_model.pkl")
        with open(model_path, "wb") as f:
            pk.dump(model, f)

        return OutputModel(random_forest_model_path=str(model_path))








