# Loading the dataset
import pandas as pd
import numpy as np
import ast
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from sklearn.model_selection import cross_val_score

# Parameters to establish
data_df = pd.read_csv("/home/atuin/b114cb/b114cb13/ESM_RFP_1/RFPs3max_ESM_2.csv")
column_of_interest = "states.0.qy"
alphas = [0.1, 1.0, 10.0]
mask = "YS"
model = linear_model.RidgeCV(alphas=alphas)
""" Options
linear_model.LinearRegression() # Ordinary least squares
linear_model.LinearRegression(positive=True) # Non-negative least squares (condition extrapolable to other models)
linear_model.Ridge(alpha=.5) # Ridge regression
linear_model.RidgeCV(alphas=alphas)
linear_model.LassoLarsCV()
etc.
"""

# Dropping those rows which have a missing value for the column of interest
print("Shape before dropping:")
print(data_df.shape)
data_df.dropna(subset=[column_of_interest], inplace=True)
print("Shape after dropping parameters:")
print(data_df.shape)

# Dropping also the empty embeddings (erase once we solve the problems with the feature extractor)
data_df.dropna(subset=["embedding"], inplace=True)
print("Shape after dropping embedding:")
print(data_df.shape)

# Defining X and Y
X = data_df["embedding"].values
new_X = []
for x in X:
	arr = ast.literal_eval (x)
	new_X.append(arr)
X = np.array(new_X)

Y = data_df[column_of_interest].values

# Cross-validation
scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores  # Convert negative MSE scores to positive

# Evaluate the model
mse_mean = mse_scores.mean()
r2_mean = scores.mean()

print('Mean squared error (MSE): %.2f' % mse_mean)
print('Mean R^2 score: %.2f' % r2_mean)

# Train the linear regression model
model.fit(X, Y)

# Predict the test set results
Y_pred = model.predict(X)
print("Y predictions:")
print(Y_pred)

"""
We could also use:
model.predict([[14.96,14.76,...,199.43,17.43]])
To predict a specific example
"""

# Evaluate the model
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print('Coefficients:', model.coef_)
print('Intercept:',model.intercept_)
print('Mean squared error (MSE): %.2f' % mse)
print('Coefficient of determination (R^2): %.2f' % r2)

# Predicted values
if mask == "YES":
	filtered_uuids = data_df['uuid'].values[filtered_indices]  # Get the corresponding uuid values
	pred_Y_df = pd.DataFrame({'ID': filtered_uuids, 'Actual Value': Y, 'Predicted Value': Y_pred, 'Difference': Y - Y_pred})
	print(pred_Y_df[["ID", "Actual Value", "Predicted Value", "Difference"]].head(20))
else:
	uuids = data_df['uuid'].values[:len(Y)]  # Get the corresponding uuid values for all samples
	pred_Y_df = pd.DataFrame({'ID': uuids, 'Actual Value': Y, 'Predicted Value': Y_pred, 'Difference': Y - Y_pred})
	print(pred_Y_df[["ID", "Actual Value", "Predicted Value", "Difference"]])


# Scatter plots
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window
plt.scatter(Y, Y_pred, alpha=0.5)
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.text(0.1, 0.9, f"MSE: {mse:.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"R^2: {r2:.2f}", transform=plt.gca().transAxes)
plt.show()
