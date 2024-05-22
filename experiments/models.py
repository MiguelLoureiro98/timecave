import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from experiment_utils import split_series, shape_series

def predict_tree(ts_train: pd.Series | np.ndarray, ts_val: pd.Series | np.ndarray, model: DecisionTreeRegressor) -> dict:
    
    """
    Train and test a decision tree model.
    """

    pass

def predict_ARMA(ts_train: pd.Series | np.ndarray, ts_val: pd.Series | np.ndarray, model: ARIMA) -> dict:

    """
    Train and test an ARMA model.
    """

    pass