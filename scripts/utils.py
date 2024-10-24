import pandas as pd


def filter_nans(df):
    nan_info = df.isna().sum()
    for column in nan_info.index:
        if nan_info[column]:
            df[column] = df[column].fillna(df[column].mean())
