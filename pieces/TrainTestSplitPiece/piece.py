from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from sklearn.model_selection import train_test_split
import pandas as pd


class TrainTestSplitPiece(BasePiece):

    def piece_function(self, input_data: InputModel):
        """
        Split the data into training and test sets.
        """
        df = pd.DataFrame(input_data.data)
        if "target" not in df.columns:
            raise ValueError("Target column not found in data with name 'target'.")

        X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=input_data.test_data_size, random_state=input_data.random_state)

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        return OutputModel(train_data=df_train.to_dict(orient='records'), test_data=df_test.to_dict(orient='records'))