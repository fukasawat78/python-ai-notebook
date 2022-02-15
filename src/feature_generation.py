from typing import List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from feature_engine.datetime import DatetimeFeatures

def create_math_transforms(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    df["A_B_ratio"] = df["A"]/df["B"]
    
    return df

def create_count_sum(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    df["sum_of_ABCD"] = df[["A", "B", "C", "D"]].gt(0.0).sum(axis=1)
    
    return df

def break_down_category(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    df["A_prefix"] = df["A"].str.split("_", n=1, expand=True)[0]
    
    return df

def create_grouped_tranforms(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    df["A_group_median"] = df.groupby("Group")["A"].transform("median")
    
    return df

def create_clustering_features(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    
    X = df.drop(columns=target)
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    
    kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
    df["cluster_number"] = kmeans.fit_predict(X_scaled)
    
    return df


def create_datetime_features(
    df: pd.DataFrame,
    date_col_names: List
) -> pd.DataFrame:
    
    dtfs = DatetimeFeatures(
        variables = date_col_names,
        features_to_extract=["month", "month_end", "day_of_year"],
        drop_original=True
    )
    
    df = dtfs.fit_transform(df)
    
    return df