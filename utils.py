import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def data_pipeline(df):

    cat_feats = [col for col in df.columns if df[col].dtype == 'O']
    num_feats = [col for col in df.columns if col not in cat_feats]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OrdinalEncoder(handle_unknown='error'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_feats),
        ('cat', cat_transformer, cat_feats)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor)
    ])

    return pipeline


