from typing import Dict
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

from feature_engine.encoding import (
    OrdinalEncoder, 
    OneHotEncoder, 
    CountFrequencyEncoder, 
    RareLabelEncoder
)

from sklearn.pipeline import Pipeline
import lightgbm as lgbm

# --------------------------------------
# 全体処理のパイプライン
# --------------------------------------
def create_pipeline(
    config: Dict,
    df: pd.DataFrame
) -> pd.DataFrame:
    
    mypipeline = Pipeline([
        
        # Inputation
        ('categorical_imputation'. CategoricalImputer(
            imputation_method='missing', 
            varibales=config["cat_col_names"])
        ),
        
        # add missing indicator to numerical variables
        ('missing_indicator', AddMissingIndicator(
            variables=config["cat_col_names"])
        ),
        
       
        # categorical encoder
        ('rare_label_encoder', RareLabelEncoder(
            tol=0.05, 
            variables=config["cat_col_names"])
        ),
        
        ('categorical_encoder', OneHotEncoder(
            top_categories=10,
            drop_last=True,
            variables=config["cat_col_names"])
        ),
        
    
        
        ('classifier', lgbm.LGBMClassifier()),
        
    ])
    
    return mypipeline
    