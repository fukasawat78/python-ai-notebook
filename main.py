import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src import *
from config.globals import *

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

def set_config(df, config):
    """
    Load config file
    """

    df_config = df.copy()

    data_settings = config["data_setting"]
    df_config[data_settings["num_col_names"]] = df_config[data_settings["num_col_names"]].astype("float")
    df_config[data_settings["cat_col_names"]] = df_config[data_settings["cat_col_names"]].astype("category")
    df_config[config["model_setting"]["target"]] = df_config[config["model_setting"]["target"]].astype("category")

    return df_config

    
if __name__ == "__main__":

    ###################
    # Read Data 
    ###################
    dataset_path = DATA_DIR + "/mobile_price_classification"
    df = read_data(dataset_path)

    # set configuration
    config_path = CONF_DIR + "/config.yml"
    config = read_config(config_path)

    df = set_config(df, config)

    ###################
    # Prerocessing
    ###################
    df = categorical_imputer(
        df=df, 
        cat_col_names=config["data_setting"]["cat_col_names"]
    )

    df = rarelabel_encoder(
        df=df, 
        cat_col_names=config["data_setting"]["cat_col_names"]
    )
    df = ordinal_encoder(
        df=df, 
        cat_col_names=config["data_setting"]["cat_col_names"]
    )

    df = equal_freq_discretiser(
        df=df, 
        num_col_names=config["data_setting"]["num_col_names"]
    )
    df = variable_transformer(
        df=df, 
        num_col_names=config["data_setting"]["num_col_names"],
        variable_type="power_transformer"
    )

    df = censor_outliers(
        df=df, 
        num_col_names=config["data_setting"]["num_col_names"]
    )

    df = drop_constant_features(df)

    ###################
    # Train Test Split
    ###################
    train, test = df[df["type"]=="train"].drop(columns="type"), df[df["type"]=="test"].drop(columns="type")

    train, val, test = data_splitting(
        df=target_transformer(df=train, target=config["model_setting"]["target"]),
        target=config["model_setting"]["target"],
        n_splits=config["model_setting"]["n_splits"],
        shuffle=config["model_setting"]["shuffle"],
        random_state=config["model_setting"]["SEED"]
    )

    ###################
    # Train and Evaluate
    ###################
    trainer = Trainer(
        model=get_model(),
        target=config["model_setting"]["target"],
        model_path="./model",
        logs_path="./logs",
        random_state=config["model_setting"]["SEED"]
    )