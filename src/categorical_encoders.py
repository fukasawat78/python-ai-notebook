from typing import List
import numpy as np
import pandas as pd
from feature_engine.encoding import (
    OrdinalEncoder, 
    OneHotEncoder, 
    CountFrequencyEncoder, 
    RareLabelEncoder
)

# --------------------------------------
# カテゴリー値に変換
# --------------------------------------
def ordinal_encoder(
    df: pd.DataFrame,
    cat_col_names: List,
) -> pd.DataFrame:
    
    oe = OrdinalEncoder(
        encoding_method='arbitrary',
        variables=cat_col_names,
    )
    
    df = oe.fit_transform(df)
        
    return df

# --------------------------------------
# (0, 1)に変換
# --------------------------------------
def onehot_encoder(
    df: pd.DataFrame,
    cat_col_names: List,
) -> pd.DataFrame:
    
    ohe = OneHotEncoder(
        top_categories=10,
        drop_last=True,
        variables=cat_col_names
    )
    
    df_ohe = ohe.fit_transform(df[cat_col_names])
    
    df_ = pd.concat([df, df_ohe], axis=1)
                          
    return df_

# --------------------------------------
# ラベルの出現頻度に変換
# --------------------------------------
def count_freq_encoder(
    df: pd.DataFrame,
    cat_col_names: List,
) -> pd.DataFrame:
    
    ce = CountFrequencyEncoder(
        encoding_method='frequency',
        variables=cat_col_names
    )
   
    df_ce = ce.fit_transform(df[cat_col_names])
    df_ce = df_ce.add_suffix("_count_freq")
    
    df_ = pd.concat([df, df_ce], axis=1)
    
    return df_

# --------------------------------------
# 出現頻度の低いラベルはまとめたカテゴリーに変換
# --------------------------------------
def rarelabel_encoder(
    df: pd.DataFrame,
    cat_col_names: List,
) -> pd.DataFrame:
    
    re = RareLabelEncoder(
        tol=0.10, 
        n_categories=10,
        variables=cat_col_names
    )
    
    df = re.fit_transform(df)
    
    return df
    
# --------------------------------------
# 出現頻度のランクに変換
# --------------------------------------
def count_rank_encoder(
    df: pd.DataFrame,
    cat_col_names: List,
) -> pd.DataFrame:
    
    for col in cat_col_names:
        count_rank = df.groupby(col)[col].count().rank(ascending=False)
        df[f"{col}_count"] = df[col].map(count_rank)
        
    return df



                          
                