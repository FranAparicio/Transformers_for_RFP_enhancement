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

# Parameters to establish
data_df = pd.read_csv("/home/atuin/b114cb/b114cb13/ESM_RFP_1/RFPs3max_ESM_2.csv")
column_of_interest = "states.0.ex_max"
seed = 13
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
print("Shape after dropping:")
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


# np.save("X_values.npy", X) # To store value for faster processing uncomment
# np.load("X_values.npy") # To use stored value uncomment

Y = data_df[column_of_interest].values

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
print("shape of the train - test, X - Y values:")
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Train the linear regression model
model.fit(X_train, Y_train)

# Predict the test set results
Y_pred = model.predict(X_test)
print("Y predictions:")
print(Y_pred)

"""
We could also use:
model.predict([[14.96,14.76,...,199.43,17.43]])
To predict a specific example
"""

# Filter outliers
if mask == "YES":
	mask = np.logical_and(Y_pred >= min_mask, Y_pred <= max_mask) # This is for QY, change as appropriate
	Y_test_filtered = Y_test[mask]
	Y_pred_filtered = Y_pred[mask]
	Y_test = Y_test_filtered
	Y_pred = Y_pred_filtered
	filtered_indices = np.where(mask)[0]

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print('Coefficients:', model.coef_)
print('Intercept:',model.intercept_)
print('Mean squared error (MSE): %.2f' % mse)
print('Coefficient of determination (R^2): %.2f' % r2)

# Predicted values
if mask == "YES":
	filtered_uuids = data_df['uuid'].values[filtered_indices]  # Get the corresponding uuid values
	print("Length of filtered_uuids:", len(filtered_uuids))
	print("Length of Y_test:", len(Y_test))
	print("Length of Y_pred:", len(Y_pred))
	pred_Y_df = pd.DataFrame({'ID': filtered_uuids, 'Actual Value': Y_test, 'Predicted Value': Y_pred, 'Difference': Y_test - Y_pred})
	print(pred_Y_df[["ID", "Actual Value", "Predicted Value", "Difference"]].head(20))
else:
	uuids = data_df['uuid'].values[:len(Y_test)]  # Get the corresponding uuid values for all samples
	print("Length of Y_test:", len(Y_test))
	print("Length of Y_pred:", len(Y_pred))
	pred_Y_df = pd.DataFrame({'ID': uuids, 'Actual Value': Y_test, 'Predicted Value': Y_pred, 'Difference': Y_test - Y_pred})
	print(pred_Y_df[["ID", "Actual Value", "Predicted Value", "Difference"]])


# Scatter plots
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.text(0.1, 0.9, f"MSE: {mse:.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"R^2: {r2:.2f}", transform=plt.gca().transAxes)
plt.show()
