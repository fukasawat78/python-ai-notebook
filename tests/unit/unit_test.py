import pytest
import pandas as pd
from src.categorical_encoders import *

"""
@pytest.fixture()
def category_data_examples():
    return pd.DataFrame({
        "num_1": [15.2, 29.2, 8.5],
        "cat_1": ["male", "female", "others"],
        "cat_2": ["car", "bike", "bycycle"],
        "num_2": [0, 2, 1],
        "cat_3": ["10-20", "20-30", "30-40"],
    })

@pytest.fixture()
def cat_names_list_examples():
    return ["cat_1", "cat_2", "cat_3"]
"""

category_data_examples = pd.DataFrame({
        "num_1": [15.2, 29.2, 8.5],
        "cat_1": ["male", "female", "others"],
        "cat_2": ["car", "bike", "bycycle"],
        "num_2": [0, 2, 1],
        "cat_3": ["10-20", "20-30", "30-40"],
    })

cat_names_list_examples = ["cat_1", "cat_2", "cat_3"]

class TestCategoricalEncoders:

    def test_ordinal_encoder(self):
        print(ordinal_encoder(category_data_examples, cat_names_list_examples))
        assert isinstance(ordinal_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
    def test_onehot_encoder(self):
        assert isinstance(onehot_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
    def test_rarelabel_encoder(self):
        assert isinstance(rarelabel_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
    
    

        