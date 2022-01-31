from typing import List
import numpy as np
import pandas as pd
from src.cat_encoder import *

def target_transformer(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    
    if df[target].dtypes[0] == "category":
        
        df = ordinal_encoder(df, target)
        
    elif df[target].dtypes[0] == "float":
        
        pass
        
    else:
        raise ValueError('target_type must be either "category" or "float"')
        
    return df
    
    