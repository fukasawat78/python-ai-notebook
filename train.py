import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src import *
from config import *
dataset_path = DATA_DIR + "/01_mobile_price_classification"

cfg = {
    "num_col_names": [
        "battery_power", 
        "clock_speed", 
        "int_memory", 
        "m_dep",
        "mobile_wt", 
        "px_height",
        "px_width",
        "ram",
        "sc_h",
        "sc_w",
        "talk_time",
    ],
    "cat_col_names": [
        "blue", 
        "dual_sim", 
        "fc", 
        "four_g", 
        "n_cores", 
        "pc", 
        "three_g",
        "touch_screen",
        "wifi",
    ],
    "target_col_name": ["price_range"],
    "n_splits": 5,
    "shuffle": True,
    "SEED": 1234
}

def read_data(dataset_path):
    """
    load CSV data 
    """
    train = pd.read_csv(dataset_path + '/train.csv')
    test = pd.read_csv(dataset_path + '/test.csv')

    train["type"] = "train"
    test["type"] = "test"
    df = pd.concat([train, test], axis=0)
    df = df.drop(columns="id")

    return df

    
if __name__ == "__main__":

    ###################
    # Read Data 
    ###################
    df = read_data(dataset_path)

    # set configuration
    df[cfg["num_col_names"]] = df[cfg["num_col_names"]].astype("float")
    df[cfg["cat_col_names"]] = df[cfg["cat_col_names"]].astype("category")
    df[cfg["target_col_name"]] = df[cfg["target_col_name"]].astype("category")

    ###################
    # Prerocessing
    ###################
    df = categorical_imputer(
        df=df, 
        cat_col_names=cfg["cat_col_names"]
    )

    #df = rarelabel_encoder(
    #    df=df, 
    #    cat_col_names=cfg["cat_col_names"]
    #)
    
    df = ordinal_encoder(
        df=df, 
        cat_col_names=cfg["cat_col_names"]
    )

    df = equal_freq_discretiser(
        df=df, 
        num_col_names=cfg["num_col_names"]
    )
    df = variable_transformer(
        df=df, 
        num_col_names=cfg["num_col_names"],
        variable_type="power_transformer"
    )

    df = censor_outliers(
        df=df, 
        num_col_names=cfg["num_col_names"]
    )

    df = drop_constant_features(df)

    ###################
    # Train Test Split
    ###################
    train, test = df[df["type"]=="train"].drop(columns="type"), df[df["type"]=="test"].drop(columns="type")

    train, val, test = data_splitting(
        df=target_transformer(df=train, target=cfg["target_col_name"]),
        target=cfg["target_col_name"],
        n_splits=cfg["n_splits"],
        shuffle=cfg["shuffle"],
        random_state=cfg["SEED"]
    )

    ###################
    # Train and Evaluate
    ###################
    trainer = Trainer(
        model=get_model(),
        target=cfg["target_col_name"],
        model_path="./model",
        logs_path="./logs",
        random_state=cfg["SEED"]
    )