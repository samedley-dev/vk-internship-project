import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from utils import filter_nans


def catboost_clf():
    return CatBoostClassifier()


def feature_target_split(df):
    filter_nans(df)
    return map(np.array, (df.drop(columns=['id', 'y']), df.loc[:, 'y']))


def train_save_model(model, df, path):
    X, y = feature_target_split(df)
    model.fit(X, y)

    model.save_model(path)


train = pd.read_parquet('input/train_processed.parquet')
train_save_model(
    catboost_clf(), train, 'models/catboost_clf.cbm'
)
