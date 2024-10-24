import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from utils import filter_nans


def catboost_clf():
    return CatBoostClassifier()


def feature_split(df):
    filter_nans(df)
    return np.array(df.drop(columns=['id']))


def submit(df):
    X = feature_split(df)

    model = catboost_clf()
    model.load_model('models/catboost_clf.cbm')
    yhat_proba = model.predict_proba(X)

    submission_df = pd.DataFrame({'id': df['id'], 'score': yhat_proba[:, 1]})
    submission_df.to_csv(path_or_buf='submission.csv',
                         header=['id', 'score'], index=False)


test = pd.read_parquet('input/test_processed.parquet')
submit(test)
