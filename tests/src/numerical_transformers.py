from typing import List
import numpy as np
import pandas as pd

from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.transformation import LogTransformer, PowerTransformer, BoxCoxTransformer, YeoJohnsonTransformer

# --------------------------------------
# 連続値を等分割したカテゴリーに変換
# --------------------------------------
def equal_freq_discretiser(
    df: pd.DataFrame,
    num_col_names: List
) -> pd.DataFrame:
    
    disc = EqualFrequencyDiscretiser(
        q=10, 
        variables=num_col_names
    )
    
    df_num_disc = disc.fit_transform(df[num_col_names])
    df_num_disc = df_num_disc.add_suffix("_disc")
    
    df_ = pd.concat([df, df_num_disc], axis=1)
    
    return df_

# --------------------------------------
# 連続値を正規変換
# --------------------------------------
def variable_transformer(
    df: pd.DataFrame,
    num_col_names: List,
    variable_type: str
) -> pd.DataFrame:
    
    if variable_type == "log_transformer":
        trans = Logtransfomer(
            variables = num_col_names
        )
        
    elif variable_type == "power_transformer":
        trans = PowerTransformer(
            variables = num_col_names
        )
    elif variable_type == "boxcox_transformer":
        trans = BoxCoxTransformer(
            variables = num_col_names
        )
    elif variable_type == "yeojohnson_transformer":
        trans = YeoJohnsonTransformer(
            variables = num_col_names
        )
    else:
        raise ValueError('variable_type must be either "log_transformer", "power_transformer", "boxcox_transformer", "yeojohnson_transfomer"')

    df_trans = trans.fit_transform(df[num_col_names])
    df_trans = df_trans.add_suffix(f"_{variable_type}")
    
    df_ = pd.concat([df, df_trans], axis=1)
    
    return df_
    