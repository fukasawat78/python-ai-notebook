from typing import List
import numpy as np
import pandas as pd

from feature_engine.selection import (
    DropFeatures, 
    DropConstantFeatures, 
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
    DropHighPSIFeatures
)

# --------------------------------------
# 特徴量の削除
# --------------------------------------
def drop_features(
    df: pd.DataFrame,
    drop_cols: List
) -> pd.DataFrame:
    
    transformer = DropFeatures(
        features_to_drop=drop_cols
    )
    
    df = transformer.fit_transform(df)
    
    return df

# --------------------------------------
# 変化の乏しい特徴量の削除
# --------------------------------------
def drop_constant_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    transformer = DropConstantFeatures(
        tol=0.8, 
        missing_values='ignore'
    )
    
    transformer.fit(df)
    
    df = transformer.transform(df)
    
    return df

# --------------------------------------
# 不安定な特徴量の削除
# --------------------------------------
def drop_high_psi_features(
    df: pd.DataFrame,
    drop_cols: List
) -> pd.DataFrame:
    
    transformer = DropHighPSIFeatures(
        split_frac=0.6
    )
    
    df = transformer.fit_transform(df)
    
    return df

# --------------------------------------
# 同一特徴量の削除
# --------------------------------------
def drop_duplicate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    transformer = DropDuplicateFeatures()
    
    df = transformer.fit_transform(df)
    
    return df

# --------------------------------------
# 相関している特徴量の削除
# --------------------------------------
def drop_smartcorr_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="variable",
        estimator=None
    )
    
    df = transformer.fit_transform(df)
    
    return df