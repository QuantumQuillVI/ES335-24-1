"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class TreeNodeDiscrete: 
    def __init__(self, feature, attr_val):
        self.feature = feature
        self.attr_val = attr_val
        self.children = {}

class TreeNodeReal:
    def __init__(self, feature, seperation_value, attr_val):
        self.feature = feature
        self.attr_val = attr_val
        self.seperation_value = seperation_value
        self.left = 0
        self.right = 0

class LeafNode:
    def __init__(self, pred, attr_val):
        self.attr_val = attr_val
        self.pred = pred
        self.is_real = True

@dataclass
class DecisionTree:
    criterion: Union[str, None] 
    max_depth: int 

    def __init__(self, criterion=None, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        

    def helper(self, X: pd.DataFrame, y: pd.Series, features: pd.Series, depth, attr_val):
        unique, count = np.unique(y, return_counts=True)
        if depth == self.max_depth or unique.size == 1 or features.size == 0:
            pred = y.mean() if check_ifreal(y) else unique[np.argmax(count)]
            leaf = LeafNode(pred, attr_val)
            leaf.is_real = check_ifreal(y)
            return leaf

        best_feature, seperation_value = opt_split_attribute(X, y, self.criterion, features)
        if check_ifreal(X[best_feature]):
            node = TreeNodeReal(best_feature, seperation_value, attr_val)
            x_left, x_right, y_left, y_right = split_data_real(X, y, best_feature, seperation_value)
            node.left = self.helper(x_left, y_left, features, depth+1, "Y")
            node.right = self.helper(x_right, y_right, features, depth+1, "N")
        else:
            features = features[features != best_feature]
            node = TreeNodeDiscrete(best_feature, attr_val)
            dic_x, dic_y = split_data_discrete(X, y, best_feature)
            for key in self.feature_values[best_feature]:
                if key in dic_x:
                    node.children[key] = self.helper(dic_x[key], dic_y[key], features, depth+1, key)
                else:
                    pred = y.mean() if check_ifreal(y) else unique[np.argmax(count)]
                    node.children[key] = LeafNode(pred, key)
                    node.children[key].is_real = check_ifreal(y)
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        features = pd.Series(X.columns)
        self.feature_values = {feature: np.unique(X[feature]) for feature in features}
        self.root = self.helper(X, y, features, 0, "root")

    def predict_single_point(self, X: pd.Series, node):
        if isinstance(node, LeafNode):
            return node.pred
        if isinstance(node, TreeNodeDiscrete):
            value = X[node.feature]
            if value in node.children:
                return self.predict_single_point(X, node.children[value])
            else:
                print("Error: Unexpected attribute value!")
        else:
            value = X[node.feature]
            if value <= node.seperation_value:
                return self.predict_single_point(X, node.left)
            else:
                return self.predict_single_point(X, node.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda instance: self.predict_single_point(instance, self.root), axis=1)

    def print_leaf(self, node, tabs):
        pred = f"Predicted Value is {node.pred}" if node.is_real else f"Class {node.pred}"
        print("\t" * tabs, f"Value -> {node.attr_val}: {pred}")

    def print_discrete(self, node, tabs):
        print("\t" * tabs, f"Value -> {node.attr_val}? (Feature -> {node.feature})")

    def print_real(self, node, tabs):
        print("\t" * tabs, f"Value -> {node.attr_val}? (Feature -> {node.feature} <= {node.seperation_value})")

    def plot_helper(self, node, tabs):
        if isinstance(node, LeafNode):
            self.print_leaf(node, tabs)
        elif isinstance(node, TreeNodeDiscrete):
            self.print_discrete(node, tabs)
            for child_node in node.children.values():
                self.plot_helper(child_node, tabs + 1)
        else:
            self.print_real(node, tabs)
            self.plot_helper(node.left, tabs + 1)
            self.plot_helper(node.right, tabs + 1)

    def plot(self) -> None:
        self.plot_helper(self.root, 0)
