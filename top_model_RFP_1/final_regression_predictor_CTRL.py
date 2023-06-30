# Loading the dataset
import pandas as pd
import numpy as np
import ast
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Input sets and parameters to establish
data_df = pd.read_csv("/home/atuin/b114cb/b114cb13/ESM_RFP_1/RFPs3max_ESM_2small.csv")
column_of_interest_1 = "states.0.qy"
column_of_interest_2 = "states.0.em_max"
prediction_df = pd.read_csv("/home/atuin/b114cb/b114cb13/ESM_RFP_1/CTRL_library_500_ESM_small.csv")
output_csv = "predictions_CTRL_library_500_ESM_small_max.csv"

""" Options
linear_model.LinearRegression() # Ordinary least squares
linear_model.LinearRegression(positive=True) # Non-negative least squares (condition extrapolable to other models)
linear_model.Ridge(alpha=.5) # Ridge regression
linear_model.RidgeCV(alphas=alphas)
linear_model.LassoLarsCV()
etc.
"""

# Dropping those rows which have a missing value for the columns of interest
print("Shape before dropping:")
print(data_df.shape)
data_df.dropna(subset=[column_of_interest_1, column_of_interest_2, "embedding"], inplace=True)
print("Shape after dropping:")
print(data_df.shape)

# Defining X and Y for QY
X_qy = np.array([ast.literal_eval(x) for x in data_df["embedding"].values])
Y_qy = data_df[column_of_interest_1].values
print("Shape of X_qy and Y_qy")
print(X_qy.shape)
print(Y_qy.shape)

# Defining X and Y for EM_MAX
X_em_max = np.array([ast.literal_eval(x) for x in data_df["embedding"].values])
Y_em_max = data_df[column_of_interest_2].values
print("Shape of X_em_max and Y_em_max")
print(X_qy.shape)
print(Y_qy.shape)

# Training the linear regression model for QY
model_qy = linear_model.RidgeCV()
model_qy.fit(X_qy, Y_qy)

# Training the linear regression model for EM_MAX
model_em_max = linear_model.RidgeCV()
model_em_max.fit(X_em_max, Y_em_max)

# Predict the training set results (for the values the model has been trained on)
Y_qy_pred = model_qy.predict(X_qy)
Y_em_pred = model_em_max.predict(X_em_max)

# Evaluate the model trained upon itself (too see if it's working)
mse_qy = mean_squared_error(Y_qy, Y_qy_pred)
mse_em_max = mean_squared_error(Y_em_max, Y_em_pred)
r2_qy = r2_score(Y_qy, Y_qy_pred)
r2_em_max = r2_score(Y_em_max, Y_em_pred)
print('----- EVALUATION OF THE MODEL UPON ITS TRAINING SET -----')
print('Coefficients qy:', model_qy.coef_)
print('Coefficients em_max:', model_em_max.coef_)
print('Intercept qy:',model_qy.intercept_)
print('Intercept em_max:',model_em_max.intercept_)
print('Mean squared error (MSE) for qy: %.2f' % mse_qy)
print('Mean squared error (MSE) for em_max: %.2f' % mse_em_max)
print('Coefficient of determination (R^2) for qy: %.2f' % r2_qy)
print('Coefficient of determination (R^2) for em_max: %.2f' % r2_em_max)

# Dropping those rows which have a missing embeddings in the
print("Shape before dropping:")
print(prediction_df.shape)
prediction_df.dropna(subset=["embedding"])
print("Shape after dropping:")
print(prediction_df.shape)

# Copying the prediction dataset into the predicted values dataset
predicted_values_df = prediction_df.copy()

# Making predictions for QY
X_pred_qy = np.array([ast.literal_eval(x) for x in prediction_df["embedding"].values])
predicted_values_df["pred.qy"] = model_qy.predict(X_pred_qy)

# Making predictions for EM_MAX
X_pred_em_max = np.array([ast.literal_eval(x) for x in prediction_df["embedding"].values])
predicted_values_df["pred.em_max"] = model_em_max.predict(X_pred_em_max)

# Saving the predicted values dataset
predicted_values_df.to_csv(output_csv, index=False)

# Printing the number of sequences predicted
num_sequences_predicted = len(predicted_values_df)
print("Number of Sequences Predicted:", num_sequences_predicted)

# Plotting the predicted QY vs predicted EM_MAX
plt.scatter(predicted_values_df["pred.qy"], predicted_values_df["pred.em_max"], color='blue', alpha=0.5, label='Predicted proteins')
plt.scatter(data_df[column_of_interest_1], data_df[column_of_interest_2], color='red', alpha=0.5, label='Characterized proteins')
plt.xlabel("Quantum yield")
plt.ylabel("Maximum emission wavelength")
plt.legend()
plt.show()


