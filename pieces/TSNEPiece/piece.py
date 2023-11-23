from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import plotly.express as px
import pickle as pk

class TSNEPiece(BasePiece):

    def read_data_from_file(self, path):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".json"):
            return pd.read_json(path)
        else:
            raise ValueError("File type not supported.")

    def piece_function(self, input_data: InputModel):
        df = self.read_data_from_file(input_data.data_path)

        if "target" not in df.columns or "target" not in df.columns:
            raise ValueError("Target column not found in data with name 'target'.")

        tsne = TSNE(n_components=input_data.n_components)
        tsne.fit(df.drop('target', axis=1))
        tsne_df = pd.DataFrame(tsne.embedding_, columns=[f'tsne_{i}' for i in range(input_data.n_components)])
        tsne_df['target'] = df['target']

        if input_data.n_components >= 2:
            if input_data.use_class_column:
                fig = px.scatter(tsne_df, x='tsne_0', y='tsne_1', color='target')
            else:
                fig = px.scatter(tsne_df, x='tsne_0', y='tsne_1')
            fig.update_layout(
                title="t-SNE Visualization of Data",
                xaxis_title="First t-SNE",
                yaxis_title="Second t-SNE",
            )
            fig.update_coloraxes(showscale=False)
            json_path = str(Path(self.results_path) / "tsne_figure.json")
            fig.write_json(json_path)
            self.display_result = {
                'file_type': 'plotly_json',
                'file_path': json_path
            }

        tsne_data_path = str(Path(self.results_path) / "tsne_data.csv")
        tsne_df.to_csv(tsne_data_path, index=False)

        return OutputModel(
            tsne_data_path=tsne_data_path,
        )




