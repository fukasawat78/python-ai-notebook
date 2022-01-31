import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# --------------------------------------
# データをtrain, val, testに分割
# --------------------------------------
def data_splitting(
    df: pd.DataFrame,
    target: str,
    n_splits: str,
    shuffle: bool=True,
    random_state: str=1234
) -> pd.DataFrame:
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx_train_val, idx_test = next(cv.split(df, df[target]))
    df_train_val = df.iloc[idx_train_val]
    df_test = df.iloc[idx_test]
    
    idx_train, idx_val = next(cv.split(df_train_val, df_train_val[target]))
    df_train = df_train_val.iloc[idx_train]
    df_val = df_train_val.iloc[idx_val]
    
    return df_train, df_val, df_test