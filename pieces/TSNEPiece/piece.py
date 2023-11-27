from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


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
            fig = go.Figure()
            color_scale = px.colors.qualitative.Bold
            if input_data.use_class_column:
                unique_targets = tsne_df['target'].unique()
                for idx, target_value in enumerate(unique_targets):
                    color = color_scale[idx % len(color_scale)]
                    filtered_data = tsne_df[tsne_df['target'] == target_value]
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data['tsne_0'],
                            y=filtered_data['tsne_1'],
                            mode='markers',
                            name=f'Target: {target_value}',
                            marker=dict(
                                color=color,
                            ),
                        )
                    )
            else:
                color = color_scale[0]
                fig.add_trace(
                    go.Scatter(
                        x=tsne_df['tsne_0'],
                        y=tsne_df['tsne_1'],
                        mode='markers',
                    )
                )

            # Create a combined figure from all separate traces
            fig.update_layout(
                title="t-SNE Projection - First two dimensions",
                xaxis_title="First Dimension",
                yaxis_title="Second Dimension",
                plot_bgcolor='rgba(255, 255, 255, 1)'
            )
            fig.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
            fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
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




