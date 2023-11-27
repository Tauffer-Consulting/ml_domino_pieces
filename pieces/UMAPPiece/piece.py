from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from umap import UMAP
import pandas as pd
from pathlib import Path
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
        df['First Dimension'] = umap_proj[:, 0]
        df['Second Dimension'] = umap_proj[:, 1]

        if input_data.n_components >= 2:
            fig = go.Figure()
            color_scale = px.colors.qualitative.Bold
            if input_data.use_class_column:
                unique_targets = df['target'].unique()
                for idx, target_value in enumerate(unique_targets):
                    color = color_scale[idx % len(color_scale)]
                    filtered_data = df[df['target'] == target_value]
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data['First Dimension'],
                            y=filtered_data['Second Dimension'],
                            mode='markers',
                            name=f'Target: {target_value}',
                            marker=dict(
                                color=color,
                            ),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df['First Dimension'],
                        y=df['Second Dimension'],
                        mode='markers',
                    )
                )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(
                title="UMAP Projection - First two dimensions",
                xaxis_title="First Dimension",
                yaxis_title="Second Dimension",
                plot_bgcolor='rgba(255, 255, 255, 1)'
            )
            fig.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
            json_path = str(Path(self.results_path) / "umap_figure.json")
            fig.write_json(json_path)
            self.display_result = {
                'file_type': 'plotly_json',
                'file_path': json_path
            }

        umap_data_path = str(Path(self.results_path) / "umap_data.csv")
        df.to_csv(umap_data_path, index=False)

        return OutputModel(
            umap_data_path=umap_data_path,
        )




