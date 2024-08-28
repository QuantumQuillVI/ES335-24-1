import pandas as pd
import numpy as np

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    assert isinstance(y, pd.Series), "Input must be a pandas Series."
    return y.dtype in [float, int]


def entropy(y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    unique, counts = np.unique(y, return_counts=True)
    total = y.size
    result = 0
    for i in range(len(unique)):
        proportion = counts[i] / total
        result -= proportion * np.log(proportion)
    return result


def gini_index(y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    unique, counts = np.unique(y, return_counts=True)
    total = y.size
    result = 0
    for i in range(len(unique)):
        proportion = counts[i] / total
        result += proportion * (1 - proportion)
    return result


def gini_discrete_input(y: pd.Series, attr: pd.Series) -> float:
    unique_values = np.unique(attr)
    result = 0
    for i in unique_values:
        y_i = y[attr == i]
        result += (y_i.size / y.size) * gini_index(y_i)
    return result


def gini_real_input(y: pd.Series, attr: pd.Series, value) -> float:
    y_left = y[attr <= value]
    y_right = y[attr > value]
    gini_left = gini_index(y_left) * (y_left.size / y.size)
    gini_right = gini_index(y_right) * (y_right.size / y.size)
    return gini_left + gini_right


def information_gain_discrete_input(y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain for discrete input
    """
    assert y.size == attr.size
    initial_entropy = entropy(y)
    final_entropy = 0
    unique = np.unique(attr)
    for i in unique:
        y_i = y[attr == i]
        final_entropy += (y_i.size / y.size) * entropy(y_i)
    info_gain = initial_entropy - final_entropy
    return info_gain


def information_gain_real_input(y: pd.Series, attr: pd.Series, value) -> float:
    """
    Function to calculate the information gain for real input
    """
    assert y.size == attr.size
    initial_entropy = entropy(y)
    y_left = y[attr <= value]
    y_right = y[attr > value]
    final_entropy = (y_left.size / y.size) * entropy(y_left) + (y_right.size / y.size) * entropy(y_right)
    info_gain = initial_entropy - final_entropy
    return info_gain


def mse(y: pd.Series) -> float:
    """
    Function to calculate the mean squared error (MSE)
    """
    mean_value = y.mean()
    result = ((y - mean_value) ** 2).mean()
    return result


def reduction_mse_discrete_input(y: pd.Series, attr: pd.Series) -> float:
    unique_values = np.unique(attr)
    initial_mse = mse(y)
    result = 0
    for i in unique_values:
        y_i = y[attr == i]
        result += (y_i.size / y.size) * mse(y_i)
    return initial_mse - result


def reduction_mse_real_input(y: pd.Series, attr: pd.Series, value) -> float:
    initial_mse = mse(y)
    y_left = y[attr <= value]
    y_right = y[attr > value]
    mse_left = mse(y_left) * (y_left.size / y.size)
    mse_right = mse(y_right) * (y_right.size / y.size)
    return initial_mse - (mse_left + mse_right)


def opt_split_reduction_mse(X: pd.DataFrame, y: pd.Series, features: pd.Series):
    max_reduction = float('-inf')
    best_feature = None
    best_value = None
    for feature in features:
        if check_ifreal(X[feature]):
            unique_values = np.unique(X[feature])
            sorted_values = (unique_values[:-1] + unique_values[1:]) / 2
            for value in sorted_values:
                current_reduction = reduction_mse_real_input(y, X[feature], value)
                if current_reduction > max_reduction:
                    max_reduction = current_reduction
                    best_feature = feature
                    best_value = value
        else:
            current_reduction = reduction_mse_discrete_input(y, X[feature])
            if current_reduction > max_reduction:
                max_reduction = current_reduction
                best_feature = feature
                best_value = None
    return best_feature, best_value


def opt_split_information_gain(X: pd.DataFrame, y: pd.Series, features: pd.Series):
    max_info_gain = float('-inf')
    best_feature = None
    best_value = None
    for feature in features:
        if check_ifreal(X[feature]):
            unique_values = np.unique(X[feature])
            sorted_values = (unique_values[:-1] + unique_values[1:]) / 2
            for value in sorted_values:
                current_info_gain = information_gain_real_input(y, X[feature], value)
                if current_info_gain > max_info_gain:
                    max_info_gain = current_info_gain
                    best_feature = feature
                    best_value = value
        else:
            current_info_gain = information_gain_discrete_input(y, X[feature])
            if current_info_gain > max_info_gain:
                max_info_gain = current_info_gain
                best_feature = feature
                best_value = None
    return best_feature, best_value


def opt_split_gini_index(X: pd.DataFrame, y: pd.Series, features: pd.Series):
    min_gini = float('inf')
    best_feature = None
    best_value = None
    for feature in features:
        if check_ifreal(X[feature]):
            unique_values = np.unique(X[feature])
            sorted_values = (unique_values[:-1] + unique_values[1:]) / 2
            for value in sorted_values:
                current_gini = gini_real_input(y, X[feature], value)
                if current_gini < min_gini:
                    min_gini = current_gini
                    best_feature = feature
                    best_value = value
        else:
            current_gini = gini_discrete_input(y, X[feature])
            if current_gini < min_gini:
                min_gini = current_gini
                best_feature = feature
                best_value = None
    return best_feature, best_value


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    """
    if check_ifreal(y):
        return opt_split_reduction_mse(X, y, features)
    elif criterion == "information_gain":
        return opt_split_information_gain(X, y, features)
    else:
        return opt_split_gini_index(X, y, features)


def split_data_real(X: pd.DataFrame, y: pd.Series, attribute, value):
    x_left = X[X[attribute] <= value]
    x_right = X[X[attribute] > value]
    y_left = y[X[attribute] <= value]
    y_right = y[X[attribute] > value]
    return x_left, x_right, y_left, y_right


def split_data_discrete(X: pd.DataFrame, y: pd.Series, split_attribute):
    unique_values = np.unique(X[split_attribute])
    dic_x = {}
    dic_y = {}
    for value in unique_values:
        dic_x[value] = X[X[split_attribute] == value]
        dic_y[value] = y[X[split_attribute] == value]
    return dic_x, dic_y
