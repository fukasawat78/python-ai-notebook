import pytest
import pandas as pd
from src.categorical_encoders import *
from tests.unit.unit_test import *

class TestEncoders:
    
    @pytest.fixture()
    def _init(self):
        
        self.test_categorical_encoders = TestCategoricalEncoders()
    
        
    def test_categorical_encoders(self, _init):
        
        category_data_examples = pd.DataFrame({
            "num_1": [15.2, 29.2, 8.5],
            "cat_1": ["male", "female", "others"],
            "cat_2": ["car", "bike", "bycycle"],
            "num_2": [0, 2, 1],
            "cat_3": ["10-20", "20-30", "30-40"],
        })

        cat_names_list_examples = ["cat_1", "cat_2", "cat_3"]
    
        print(self.test_categorical_encoders.test_ordinal_encoder())
        print(self.test_categorical_encoders.test_onehot_encoder())
        print(self.test_categorical_encoders.test_rarelabel_encoder())
    
    

        