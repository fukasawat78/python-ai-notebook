from typing import List
import numpy as np
import pandas as pd
from src.cat_encoder import *

def target_transformer(
    df: pd.DataFrame,
    target: str,
    target_type: str,
) -> pd.DataFrame:
    
    if target_type == "cat":
        
        df = ordinal_encoder(df, target)
        
    elif target_type == "num":
        
        pass
        
    else:
        raise ValueError('target_type must be either "cat" or "num"')
        
    return df
    
    