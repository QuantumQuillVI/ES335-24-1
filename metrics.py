from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    # Assert checks to ensure the function handles edge cases
    assert y_hat.size == y.size, "Predicted and actual labels must have the same length."
    assert y_hat.ndim == 1 and y.ndim == 1, "Predicted and actual labels must be 1-dimensional."
    
    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    
    return correct_predictions / total_predictions


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    # Assert checks
    assert y_hat.size == y.size, "Predicted and actual labels must have the same length."
    assert y_hat.ndim == 1 and y.ndim == 1, "Predicted and actual labels must be 1-dimensional."
    
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_positives = ((y_hat == cls) & (y != cls)).sum()
    
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positives)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    # Assert checks
    assert y_hat.size == y.size, "Predicted and actual labels must have the same length."
    assert y_hat.ndim == 1 and y.ndim == 1, "Predicted and actual labels must be 1-dimensional."
    
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_negatives = ((y_hat != cls) & (y == cls)).sum()
    
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_negatives)




def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    # Assert checks
    assert y_hat.size == y.size, "Predicted and actual labels must have the same length."
    assert y_hat.ndim == 1 and y.ndim == 1, "Predicted and actual labels must be 1-dimensional."
    
    mse = np.mean((y_hat - y) ** 2)
    return np.sqrt(mse)


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    # Assert checks
    assert y_hat.size == y.size, "Predicted and actual labels must have the same length."
    assert y_hat.ndim == 1 and y.ndim == 1, "Predicted and actual labels must be 1-dimensional."
    
    return np.mean(np.abs(y_hat - y))
