from typing import List
import numpy as np
import pandas as pd

from feature_engine.imputation import CategoricalImputer, DropMissingData

# --------------------------------------
# カテゴリー値の欠損を文字列"missing"に変換
# --------------------------------------
def categorical_imputer(
    df: pd.DataFrame,
    cat_col_names: List
) -> pd.DataFrame:
    
    imputer = CategoricalImputer(
        variables=cat_col_names
    )
    
    df = imputer.fit_transform(df)
    
    return df

# --------------------------------------
# 欠損のある行を全て削除
# --------------------------------------
def drop_missing_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    df_cols = df.columns.tolist()
    
    missing_data_imputer = DropMissingData(
        variables=df_cols
    )
    
    df = missing_data_imputer.fit_transform(df)
    
    return df
