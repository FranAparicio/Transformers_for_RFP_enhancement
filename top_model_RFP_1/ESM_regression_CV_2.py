# Loading the dataset and other initializations
import pandas as pd
import numpy as np
import ast
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk

# Parameters to establish
data_df = pd.read_csv("/home/atuin/b114cb/b114cb13/ESM_RFP_1/RFPs3max_ESM_2.csv")
column_of_interest = "states.0.em_max"
alphas = [0.1, 1.0, 10.0]
mask = "YS"
model = linear_model.RidgeCV(alphas=alphas)

# Dropping rows with missing values in the column of interest
print("Shape before dropping parameters:")
print(data_df.shape)
data_df.dropna(subset=[column_of_interest], inplace=True)
print("Shape after dropping parameters:")
print(data_df.shape)

# Dropping rows with empty embeddings
data_df.dropna(subset=["embedding"], inplace=True)
print("Shape after dropping embedding:")
print(data_df.shape)

# Defining X and Y
X = data_df["embedding"].values
new_X = []
for x in X:
    arr = ast.literal_eval(x)
    new_X.append(arr)
X = np.array(new_X)

Y = data_df[column_of_interest].values

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)
print("Shape of the train - test, X - Y values:")
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Perform cross-validation on the training set
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
mse_cv = -np.mean(cv_scores)

# Train the model on the full training set
model.fit(X_train, Y_train)

# Predict the test set results
Y_pred = model.predict(X_test)

# Filter outliers
if mask == "YES":
    mask = np.logical_and(Y_pred >= min_mask, Y_pred <= max_mask)  # This is for QY, change as appropriate
    Y_test_filtered = Y_test[mask]
    Y_pred_filtered = Y_pred[mask]
    Y_test = Y_test_filtered
    Y_pred = Y_pred_filtered
    filtered_indices = np.where(mask)[0]

# Evaluate the model on the test set
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print('Mean squared error (MSE) - Cross-validation: %.2f' % mse_cv)
print('Mean squared error (MSE) - Test set: %.2f' % mse)
print('Coefficient of determination (R^2) - Test set: %.2f' % r2)

# Predicted values
if mask == "YES":
    filtered_uuids = data_df['uuid'].values[filtered_indices]  # Get the corresponding uuid values
    pred_Y_df = pd.DataFrame({'ID': filtered_uuids, 'Actual Value': Y_test, 'Predicted Value': Y_pred,
                              'Difference': Y_test - Y_pred})
    print(pred_Y_df[["ID", "Actual Value", "Predicted Value", "Difference"]])
else:
    pred_Y_df = pd.DataFrame({'Actual Value': Y_test, 'Predicted Value': Y_pred, 'Difference': Y_test - Y_pred})
    print(pred_Y_df)

# Scatter plot of predicted vs actual values
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.text(0.1, 0.9, f"MSE: {mse:.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"R^2: {r2:.2f}", transform=plt.gca().transAxes)
plt.show()

