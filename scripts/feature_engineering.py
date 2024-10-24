import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import datetime
from typing import List


def calculate_date(date: datetime.date) -> int:
    return (date - datetime.date(1970, 1, 1)).days


def get_start_date(dates: List[datetime.date]) -> int:
    return calculate_date(dates[0])


def get_end_date(dates: List[datetime.date]) -> int:
    return calculate_date(dates[-1])


def calculate_duration(dates: List[datetime.date]) -> int:
    return get_end_date(dates) - get_start_date(dates)


def calculate_mean_value(values: List[float]) -> float:
    return sum(values) / len(values) if len(values) > 0 else 0


def calculate_min_value(values: List[float]) -> float:
    return min(values)


def calculate_max_value(values: List[float]) -> float:
    return max(values)


def calculate_std_dev(values: List[float]) -> float:
    return np.std(values) if len(values) > 0 else 0


def calculate_median_value(values: List[float]) -> float:
    return np.median(values) if len(values) > 0 else 0


def count_values(values: List[float]) -> int:
    return len(values)


def calculate_cumulative_change(values: List[float]) -> float:
    return sum(np.diff(values)) if len(values) > 1 else 0


def calculate_iqr(values: List[float]) -> float:
    return np.percentile(values, 75) - np.percentile(values, 25)


def calculate_skewness(values: List[float]) -> float:
    return skew(np.array(values)) if len(values) > 0 else 0


def calculate_kurtosis(values: List[float]) -> float:
    return kurtosis(values) if len(values) > 0 else 0


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = pd.DataFrame({
        'id': df['id'],

        'start_date': df['dates'].apply(get_start_date),
        'end_date': df['dates'].apply(get_end_date),
        'duration': df['dates'].apply(calculate_duration),

        'mean_value': df['values'].apply(calculate_mean_value),
        'min_value': df['values'].apply(calculate_min_value),
        'max_value': df['values'].apply(calculate_max_value),
        'count_values': df['values'].apply(count_values),
        'median_value': df['values'].apply(calculate_median_value),

        'cumulative_change': df['values'].apply(calculate_cumulative_change),
        'std_dev': df['values'].apply(calculate_std_dev),
        'iqr': df['values'].apply(calculate_iqr),
        'skewness': df['values'].apply(calculate_skewness),
        'kurtosis': df['values'].apply(calculate_kurtosis),
    })

    if 'label' in df.columns:
        processed_df['y'] = df['label']

    return processed_df


train = pd.read_parquet('input/train.parquet')
test = pd.read_parquet('input/test.parquet')

processed_train = generate_features(train)
processed_test = generate_features(test)

processed_train.to_parquet('input/train_processed.parquet')
processed_test.to_parquet('input/test_processed.parquet')
