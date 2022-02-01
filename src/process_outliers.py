from typing import List
import numpy as np
import pandas as pd

from feature_engine.outliers import Winsorizer, OutlierTrimmer

# --------------------------------------
# ±3σを最大値/最小値として外れた値を修正
# --------------------------------------
def censor_outliers(
    df: pd.DataFrame,
    num_col_names: List
) -> None:
    
    capper = Winsorizer(
        capping_method='gaussian',
        tail='right', 
        fold=3, 
        variables=num_col_names
    )
    
    df = capper.fit_transform(df)
    
    return df

# --------------------------------------
# 四分位を基準として外れた値を除去
# --------------------------------------
def remove_outliers(
    df: pd.DataFrame,
    num_col_names: List
) -> None:
    
    capper = OutlierTrimmer(
        capping_method='iqr', 
        tail='right', fold=1.5, 
        variables=num_col_names
    )
    
    df = capper.fit_transform(df)
    
    return df