from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from umap import UMAP
import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


class UMAPPiece(BasePiece):

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

        umap_model = UMAP(n_components=input_data.n_components, init='random', random_state=0)
        umap_proj = umap_model.fit_transform(df.drop('target', axis=1))

        # Adding UMAP components to DataFrame
        df['UMAP_Component_1'] = umap_proj[:, 0]
        df['UMAP_Component_2'] = umap_proj[:, 1]

        if input_data.n_components >= 2:
            if input_data.use_class_column:
                fig = px.scatter(
                    df,
                    x='UMAP_Component_1',
                    y='UMAP_Component_2',
                    color='target',
                    title='UMAP Visualization of Data',
                )
                fig.update_coloraxes(showscale=False)
            else:
                fig = px.scatter(
                    df,
                    x='UMAP_Component_1',
                    y='UMAP_Component_2',
                    title='UMAP Visualization of Data',
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



