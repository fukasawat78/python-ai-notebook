from typing import List
import numpy as np
import pandas as pd
from src.base_trainer import BaseTrainer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# --------------------------------------
# Trainer
# --------------------------------------
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    
    def __init__(self, model, target, random_state):
        super().__init__()
        
        self.model = model
        self.target = target
        self.random_state = random_state
        
    def fit(self, train, val):
        
        X_train, X_val = train.drop(columns=self.target), val.drop(columns=self.target)
        y_train, y_val = train[self.target], val[self.target]
        
        self.model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            early_stopping_rounds=100,
        )
        
    def evaluate(self, test):
    
        X_test = test.drop(columns=self.target)
        y_test = test[self.target]
    
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
    
        if y_test.nunique()[0] <= 2:
            print("confusion_matrix:\n {}".format(confusion_matrix(y_test, y_pred)))
            print("classification report:\n {}".format(classification_report(y_test, y_pred)))
            print("AUC: {}".format(roc_auc_score(y_test, y_pred_proba)))
        else:
            print("confusion_matrix:\n {}".format(confusion_matrix(y_test, y_pred)))
            print("classification report:\n {}".format(classification_report(y_test, y_pred)))
            print("AUC: {}".format(roc_auc_score(y_test, y_pred_proba, multi_class='ovr')))
    
    def predict(self, X_test):
        
        y_pred = self.model.predict(X_test)
        
        return y_pred
        
        
        
        
