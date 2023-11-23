from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle as pk

class PCATrainPiece(BasePiece):

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
        df = self.read_data_from_file(input_data.data_path)

        if "target" not in df.columns or "target" not in df.columns:
            raise ValueError("Target column not found in data with name 'target'.")

        pca = PCA(n_components=input_data.n_components)
        pca.fit(df.drop('target', axis=1))

        pca_df = pd.DataFrame(pca.transform(df.drop('target', axis=1)), columns=[f"pca_{i}" for i in range(input_data.n_components)])
        pca_df['target'] = df['target']

        # Create a horizontal bar plot
        barplot_df = pd.DataFrame({
            'Principal Component': [f"PC{i + 1}" for i in range(input_data.n_components)],
            'Explained Variance Ratio': pca.explained_variance_ratio_
        })
        barplot_df.sort_values(by='Explained Variance Ratio', ascending=True, inplace=True)

        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(
            go.Bar(
                x=barplot_df['Explained Variance Ratio'], 
                y=barplot_df['Principal Component'], 
                name='Explained Variance Ratio',
                orientation='h'
            ),
            row=1, col=1
        )

        markers = {}
        if input_data.use_class_column:
            markers = dict(
                color=pca_df['target'],
            )
        # add scatterplot to the first component
        fig.add_trace(
            go.Scatter(
                x=pca_df['pca_0'], 
                y=pca_df['pca_1'], 
                mode='markers',
                name='Scatter Plot of First Two Principal Components',
                marker=markers,
                legendgroup='target'
            ),
            row=2, 
            col=1,
        )

        json_path = str(Path(self.results_path) / "pca_explained_variance_ratio.json")
        fig.write_json(json_path)
        self.display_result = {
            'file_type': 'plotly_json',
            'file_path': json_path
        }

        pca_data_path = str(Path(self.results_path) / "pca_data.csv")
        pca_df.to_csv(pca_data_path, index=False)

        pca_model_path = str(Path(self.results_path) / "pca_model.pkl")
        with open(pca_model_path, "wb") as f:
            pk.dump(pca, f)

        return OutputModel(
            pca_data_path=pca_data_path,
            pca_model_path=pca_model_path
        )




