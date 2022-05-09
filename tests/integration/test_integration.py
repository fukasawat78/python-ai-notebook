import pytest
import pandas as pd
from src.categorical_encoders import *

def test_ordinal_encoder(self):
    print(ordinal_encoder(category_data_examples, cat_names_list_examples))
    assert isinstance(ordinal_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
def test_onehot_encoder(self):
    assert isinstance(onehot_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
def test_rarelabel_encoder(self):
    assert isinstance(rarelabel_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)


class TestEncoders:
    
    @pytest.fixture()
    def category_data_examples(self):
        return pd.DataFrame({
            "num_1": [15.2, 29.2, 8.5],
            "cat_1": ["male", "female", "others"],
            "cat_2": ["car", "bike", "bycycle"],
            "num_2": [0, 2, 1],
            "cat_3": ["10-20", "20-30", "30-40"],
        })

    @pytest.fixture()
    def cat_names_list_examples(self):
        return ["cat_1", "cat_2", "cat_3"]
        
    def test_categorical_encoders(self):
    
        assert test_ordinal_encoder()
        assert test_onehot_encoder()
        assert test_rarelabel_encoder()
    
    

        