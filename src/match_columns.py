from typing import Tuple
import numpy as np
import pandas as pd

from feature_engine.preprocessing import MatchVariables

# --------------------------------------
# 学習データとテストデータのカラムを一致させる
# --------------------------------------
def match_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    match_cols = MatchVariables(
        missing_values="ignore"
    )
    
    match_cols.fit(train)
    
    test = match_cols.transform(test)
    
    return train, test