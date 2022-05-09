import pytest
import pandas as pd
from src.categorical_encoders import *

@pytest.fixture(scope="function", autouse=True)
def pre_function():
    scope = "Function"
    print(f"\n=== SETUP {scope} ===")
    yield
    print(f"\n=== TEARDOWN {scope} ===\n")


@pytest.fixture(scope="class", autouse=True)
def pre_class():
    scope = "Class"
    print(f"\n*** SETUP {scope} ***")
    yield
    print(f"\n*** TEARDOWN {scope} ***\n")


@pytest.fixture(scope="module", autouse=True)
def pre_module():
    scope = "Module"
    print(f"\n--- SETUP {scope} ---")
    yield
    print(f"\n--- TEARDOWN {scope} ---\n")


@pytest.fixture(scope="package", autouse=True)
def pre_package():
    scope = "Package"
    print(f"\n<<< SETUP {scope} >>>")
    yield
    print(f"\n<<< TEARDOWN {scope} >>>\n")


@pytest.fixture(scope="session", autouse=True)
def pre_session():
    scope = "Session"
    print(f"\n@@@ SETUP {scope} @@@")
    yield
    print(f"\n@@@ TEARDOWN {scope} @@@\n")



@pytest.fixture()
def category_data_examples():
    return pd.DataFrame({
        "num_1": [15.2, 29.2, 8.5],
        "cat_1": ["male", "female", "others"],
        "cat_2": ["car", "bike", "bycycle"],
        "num_2": [0, 2, 1],
        "cat_3": ["10-20", "20-30", "30-40"],
    })

class TestCategoricalEncoders:
    
    def __init__(self):
        pass

    def test_ordinal_encoder(self):
        
        category_data_examples = pd.DataFrame({
            "num_1": [15.2, 29.2, 8.5],
            "cat_1": ["male", "female", "others"],
            "cat_2": ["car", "bike", "bycycle"],
            "num_2": [0, 2, 1],
            "cat_3": ["10-20", "20-30", "30-40"],
        })

        cat_names_list_examples = ["cat_1", "cat_2", "cat_3"]

        assert isinstance(ordinal_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
    def test_onehot_encoder(self):
        
        category_data_examples = pd.DataFrame({
            "num_1": [15.2, 29.2, 8.5],
            "cat_1": ["male", "female", "others"],
            "cat_2": ["car", "bike", "bycycle"],
            "num_2": [0, 2, 1],
            "cat_3": ["10-20", "20-30", "30-40"],
        })

        cat_names_list_examples = ["cat_1", "cat_2", "cat_3"]
        
        assert isinstance(onehot_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
    def test_rarelabel_encoder(self):
        
        category_data_examples = pd.DataFrame({
            "num_1": [15.2, 29.2, 8.5],
            "cat_1": ["male", "female", "others"],
            "cat_2": ["car", "bike", "bycycle"],
            "num_2": [0, 2, 1],
            "cat_3": ["10-20", "20-30", "30-40"],
        })

        cat_names_list_examples = ["cat_1", "cat_2", "cat_3"]
        assert isinstance(rarelabel_encoder(category_data_examples, cat_names_list_examples), pd.DataFrame)
    
    
    

        