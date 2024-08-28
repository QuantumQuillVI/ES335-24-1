import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Load the Auto MPG dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])

# Data cleaning and preprocessing
y = data["mpg"]
X = data[["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]]

# Remove rows with missing values and convert types
X = X[X["horsepower"] != '?']
y = y[X.index]  # Keep only corresponding rows in y
X["horsepower"] = X["horsepower"].astype(float)
X["model year"] = X["model year"].astype(int)
X["cylinders"] = X["cylinders"].astype(int)

# Feature scaling for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled arrays back to DataFrame for compatibility with custom DecisionTree
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Reset indices to avoid misalignment issues
X_scaled_df = X_scaled_df.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X_scaled_df, y, random_state=0)

# Part a: Train and evaluate using the custom DecisionTree
print("Custom DecisionTree Performance:")

custom_tree = DecisionTree(criterion="gini_index", max_depth=10)
custom_tree.fit(train_x, train_y)
y_hat_custom = custom_tree.predict(test_x)
custom_tree.plot()

# Calculate performance metrics
print("RMSE (Custom Tree):", rmse(y_hat_custom, test_y))
print("MAE (Custom Tree):", mae(y_hat_custom, test_y))

# Part b: Train and evaluate using scikit-learn DecisionTreeRegressor
print("\nScikit-Learn DecisionTreeRegressor Performance:")

sklearn_tree = DecisionTreeRegressor(random_state=0)
sklearn_tree.fit(train_x, train_y)
y_hat_sklearn = sklearn_tree.predict(test_x)

# Calculate performance metrics
print("RMSE (Sklearn Tree):", rmse(y_hat_sklearn, test_y))
print("MAE (Sklearn Tree):", mae(y_hat_sklearn, test_y))

# Optional: Visualize predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(test_y, y_hat_custom, color='blue', label='Custom Tree Predictions')
plt.scatter(test_y, y_hat_sklearn, color='red', label='Sklearn Tree Predictions', alpha=0.6)
plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], '--k')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values for Custom and Sklearn Decision Trees')
plt.legend()
plt.show()
