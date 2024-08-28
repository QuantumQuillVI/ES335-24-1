import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from itertools import product

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Scatter plot of Generated Dataset")
plt.show()

# Split data into training and test sets (70%-30%)
split_index = int(0.7 * len(X))
X_train, y_train = pd.DataFrame(X[:split_index], columns=['index 1', 'index 2']), pd.Series(y[:split_index], dtype="category")
X_test, y_test = pd.DataFrame(X[split_index:], columns=['index 1', 'index 2']), pd.Series(y[split_index:], dtype="category")

# Part a: Train and evaluate the decision tree on the test dataset
print("Q2(a)")

for criteria in ["information_gain", "gini_index"]:
    Dtree = DecisionTree(criterion=criteria)
    print(f"\nCriteria: {criteria}") 
    Dtree.fit(X_train, y_train)
    y_hat = Dtree.predict(X_test)
    
    # Show results
    print("Accuracy: ", accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print(f"Precision for class {cls}: {precision(y_hat, y_test, cls)}")
        print(f"Recall for class {cls}: {recall(y_hat, y_test, cls)}")

# Part b: 5-fold cross-validation and nested cross-validation for optimal depth
print("\nQ2(b)")

def k_fold_split(X, y, k, i):
    fold_size = len(X) // k
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    X_test, y_test = X[test_start:test_end], y[test_start:test_end]
    X_train = np.concatenate((X[:test_start], X[test_end:]), axis=0)
    y_train = np.concatenate((y[:test_start], y[test_end:]), axis=0)
    return pd.DataFrame(X_train, columns=['index 1', 'index 2']), pd.DataFrame(X_test, columns=['index 1', 'index 2']), pd.Series(y_train, dtype="category"), pd.Series(y_test, dtype="category")

# 5-fold cross-validation
k = 5
accuracies = []

for i in range(k):
    training_set, test_set, training_labels, test_labels = k_fold_split(X, y, k, i)
    dt_classifier = DecisionTree(criterion='information_gain')
    dt_classifier.fit(training_set, training_labels)
    fold_predictions = dt_classifier.predict(test_set)
    fold_accuracy = accuracy(fold_predictions, test_labels)
    accuracies.append(fold_accuracy)
    print(f"Fold {i+1}: Accuracy: {fold_accuracy:.4f}")

# Nested cross-validation for hyperparameter tuning
hyperparameters = {'max_depth': list(range(1, 11)), 'criterion': ['gini_index', 'information_gain']}
num_outer_folds = 5
num_inner_folds = 5
results = []

for outer_count in range(num_outer_folds):
    X_outer_train, X_outer_test, y_outer_train, y_outer_test = k_fold_split(X, y, num_outer_folds, outer_count)
    
    for inner_count in range(num_inner_folds):
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = k_fold_split(X_outer_train, y_outer_train, num_inner_folds, inner_count)
        
        for max_depth, criterion in product(hyperparameters['max_depth'], hyperparameters['criterion']):
            dt_classifier = DecisionTree(max_depth=max_depth, criterion=criterion)
            dt_classifier.fit(X_inner_train, y_inner_train)
            y_hat = dt_classifier.predict(X_inner_test)
            val_accuracy = accuracy(y_hat, y_inner_test)
            results.append({
                'outer_fold': outer_count, 
                'inner_fold': inner_count, 
                'max_depth': max_depth, 
                'criterion': criterion, 
                'val_accuracy': val_accuracy
            })

# Organize and display the results
results_df = pd.DataFrame(results)
for outer_fold in range(num_outer_folds):
    outer_fold_df = results_df[results_df['outer_fold'] == outer_fold]
    best_params = outer_fold_df.groupby(['max_depth', 'criterion']).mean()['val_accuracy'].idxmax()
    best_score = outer_fold_df.groupby(['max_depth', 'criterion']).mean()['val_accuracy'].max()
    print(f"\nBest params for outer fold {outer_fold+1}: Depth {best_params[0]}, Criterion {best_params[1]}, Accuracy: {best_score:.4f}")





