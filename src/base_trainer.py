import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

class BaseTrainer:
    """
    Base class for all trainers
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self):
        raise NotImplementedError
