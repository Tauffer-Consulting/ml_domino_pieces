from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import pandas as pd
from sklearn.preprocessing import StandardScaler


class StandardScalerPiece(BasePiece):

    def piece_function(self, input_data: InputModel):
        df_train = pd.DataFrame(input_data.train_data)
        df_test = pd.DataFrame(input_data.test_data)

        if "target" not in df_train.columns or "target" not in df_test.columns:
            raise ValueError("Target column not found in data with name 'target'.")
    

        scaler = StandardScaler()
        scaler.fit(df_train.drop('target', axis=1))
        X_train = scaler.transform(df_train.drop('target', axis=1))
        X_test = scaler.transform(df_test.drop('target', axis=1))

        df_train_scaled = pd.DataFrame(X_train, columns=df_train.drop('target', axis=1).columns)
        df_train_scaled['target'] = df_train['target']
        df_test_scaled = pd.DataFrame(X_test, columns=df_test.drop('target', axis=1).columns)
        df_test_scaled['target'] = df_test['target']

        return OutputModel(train_data=df_train_scaled.to_dict(orient='records'), test_data=df_test_scaled.to_dict(orient='records'))
        
            