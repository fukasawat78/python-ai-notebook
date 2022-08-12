import dagshub
import joblib
from typing import List
import numpy as np
import pandas as pd
from src.base_trainer import BaseTrainer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------
# Trainer
# --------------------------------------
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    
    def __init__(self, model, target, model_path, logs_path, random_state):
        super().__init__()
        
        self.model = model
        self.target = target
        self.model_path=model_path + "/model.joblib"
        self.metrics_path=logs_path + "/metrics.csv"
        self.params_path=logs_path + "/params.yml"
        self.random_state = random_state
        
    def fit(self, train, val, test):

        with dagshub.dagshub_logger(metrics_path=self.metrics_path, hparams_path=self.params_path) as logger:
            print("Training model...")

            X_train, X_val, X_test = train.drop(columns=self.target), val.drop(columns=self.target), test.drop(columns=self.target)
            y_train, y_val, y_test = train[self.target], val[self.target], test[self.target]
        
            self.model.fit(
                X_train, y_train,
                eval_set = [(X_val, y_val)],
                early_stopping_rounds=100,
            )

            joblib.dump(self.model, self.model_path)
            logger.log_hyperparams(model_class=type(self.model).__name__)
            logger.log_hyperparams({"model": self.model.get_params()})
            print("Evaluating model...")
            train_metrics = self.evaluate(X_train, y_train)
            print("Train metrics:")
            print(train_metrics)
            logger.log_metrics({f"train__{k}": v for k, v in train_metrics.items()})
            test_metrics = self.evaluate(X_test, y_test)
            print("Test metrics:")
            print(test_metrics)
            logger.log_metrics({f"test__{k}": v for k, v in test_metrics.items()})
        
            logger.save() 
            logger.close()

    def evaluate(self, X, y):

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
    
        print(y.nunique())
        if y.nunique()[0] <= 2:
            print("confusion_matrix:\n {}".format(confusion_matrix(y, y_pred)))
            print("classification report:\n {}".format(classification_report(y, y_pred)))
            print("AUC: {}".format(roc_auc_score(y, y_pred_proba)))
            return {
                "roc_auc": roc_auc_score(y, y_pred_proba),
                "average_precision": average_precision_score(y, y_pred_proba),
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1": f1_score(y, y_pred),
            }
        else:
            print("confusion_matrix:\n {}".format(confusion_matrix(y, y_pred)))
            print("classification report:\n {}".format(classification_report(y, y_pred)))
            print("AUC: {}".format(roc_auc_score(y, y_pred_proba, multi_class='ovr')))
            return {
                "roc_auc": roc_auc_score(y, y_pred_proba, multi_class='ovr'),
                #"average_precision": average_precision_score(y, y_pred_proba),
                "accuracy": accuracy_score(y, y_pred),
                #"precision": precision_score(y, y_pred),
                #"recall": recall_score(y, y_pred),
                #"f1": f1_score(y, y_pred),
            }
        
        
        
        
