from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import plotly.express as px

class TrainTestSplitPiece(BasePiece):

    def read_data_from_file(self, path):
        """
        Read data from a file.
        """
        if path.endswith(".csv"):
            return pd.read_csv(path).to_dict(orient='records')
        elif path.endswith(".json"):
            return pd.read_json(path).to_dict(orient='records')
        else:
            raise ValueError("File type not supported.")

    def piece_function(self, input_data: InputModel):

        if input_data.train_data_path:
            input_data.train_data = self.read_data_from_file(path=input_data.train_data_path)

        if input_data.test_data_path:
            input_data.test_data = self.read_data_from_file(path=input_data.test_data_path)

        df_train = pd.DataFrame(input_data.train_data)
        df_test = pd.DataFrame(input_data.test_data)

        if "target" not in df_train.columns or "target" not in df_test.columns:
            raise ValueError("Target column not found in data with name 'target'.")
        

        pca = PCA(n_components=input_data.n_components)
        pca.fit(df_train.drop('target', axis=1))

        pca_train_df = pd.DataFrame(pca.transform(df_train.drop('target', axis=1)), columns=[f"pca_{i}" for i in range(input_data.n_components)])
        pca_train_df['target'] = df_train['target']
        pca_test_df = pd.DataFrame(pca.transform(df_test.drop('target', axis=1)), columns=[f"pca_{i}" for i in range(input_data.n_components)])
        pca_test_df['target'] = df_test['target']

        # Create a horizontal bar plot
        barplot_df = pd.DataFrame({
            'Principal Component': [f"PC{i + 1}" for i in range(input_data.n_components)],
            'Explained Variance Ratio': pca.explained_variance_ratio_
        })
        barplot_df.sort_values(by='Explained Variance Ratio', ascending=True, inplace=True)

        fig = px.bar(barplot_df, x='Explained Variance Ratio', y='Principal Component', 
                    orientation='h', title='Explained Variance Ratio of Principal Components')
        fig.update_layout(xaxis_title='Explained Variance Ratio', yaxis_title='Principal Component')

        json_path = str(Path(self.results_path) / "pca_explained_variance_ratio.json")
        fig.write_json(json_path)
        self.display_result = {
            'file_type': 'plotly_json',
            'file_path': json_path
        }

        if input_data.output_type != 'file':
            return OutputModel(pca_train_data=pca_train_df.to_dict(orient='records'), pca_test_data=pca_test_df.to_dict(orient='records'))

        pca_train_data_path = str(Path(self.results_path) / "pca_train_data.csv")
        pca_test_data_path = str(Path(self.results_path) / "pca_test_data.csv")
        pca_train_df.to_csv(pca_train_data_path, index=False)
        pca_test_df.to_csv(pca_test_data_path, index=False)

        return OutputModel(pca_train_data_path=pca_train_data_path, pca_test_data_path=pca_test_data_path)




