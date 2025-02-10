# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:24:36 2024

@author: youss
"""

# Numpy to create data as array
import numpy as np
# Pandas to Get data and develop it
import pandas as pd

# Matplotlib to vis data  
import matplotlib.pyplot as plt
# Split data into train and test
from sklearn.model_selection import train_test_split
# Linear regression model
from sklearn.linear_model import LinearRegression
# Evaluate Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score


# Read dataset 
df1 = pd.read_csv('XAU_USD.csv')
df = df1.replace('[^\d.]', '', regex=True).astype(float)
# Print First 5 rows
df.head(10)

# Create copy of dataset to visualize it 
viz = df.copy()


# Check if there null cell or not
df.isnull().sum()

# Print shape of dataset (Number of rows, columns) 
df.shape

# Prnit data info like column and number of rows of each column and if it null or not and data type of it
df.info()

# Print data describtion
df.describe().T

# split data into train and test set with 20% to test and 80% of train
train, test = train_test_split(df, test_size = 0.2)
test_pred = test.copy()
train.head(10)

test.head(10)

# x Train and test with 4 column (Open, High, Low, Volum), Drop (Date, Adj Close, Close(Prediction))
# x Train and test with 4 column (Open, High, Low, Volum), Drop (Date, Adj Close, Close(Prediction))
x_train = train[['Open', 'High', 'Low', 'OpenY', 'HighY' ,'LowY' ,'OpenYY' ,'HighYY' ,'LowYY']].values
x_test = test[['Open', 'High', 'Low', 'OpenY', 'HighY' ,'LowY' ,'OpenYY' ,'HighYY' ,'LowYY']].values
# Create y train and test with close columns
y_train = train['Close'].values
y_test = test['Close'].values


# Set Linear Regression model with name (model_lnr)
model_lnr = LinearRegression()
# Fit Training data
model_lnr.fit(x_train, y_train)
LinearRegression()
# Predict Data with x_test
y_pred = model_lnr.predict(x_test)
# Test Model
# result = model_lnr.predict([[262.000000, 267.899994, 250.029999, 11896100]])
# print(result)
# Get accuracy of model
print("MSE",round(mean_squared_error(y_test,y_pred), 3))
print("RMSE",round(np.sqrt(mean_squared_error(y_test,y_pred)), 3))
print("MAE",round(mean_absolute_error(y_test,y_pred), 3))
print("MAPE",round(mean_absolute_percentage_error(y_test,y_pred), 3))
print("R2 Score : ", round(r2_score(y_test,y_pred), 3) * 100)

# def style():
#     plt.figure(facecolor='black', figsize=(15,10))
#     ax = plt.axes()

#     ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to white
#     ax.tick_params(axis='y', colors='white')    #setting up Y-axis tick color to white

#     ax.spines['left'].set_color('white')        #setting up Y-axis spine color to white
#     #ax.spines['right'].set_color('white')
#     #ax.spines['top'].set_color('white')
#     ax.spines['bottom'].set_color('white')      #setting up X-axis spine color to white

#     ax.set_facecolor("black")                   # Setting the background color of the plot using set_facecolor() method
# viz['Date'] = pd.to_datetime(viz['Date'], format='%m/%d/%Y')
# data = pd.DataFrame(viz[['Date','Close']])
# data=data.reset_index()
# data=data.drop('index',axis=1)
# data.set_index('Date', inplace=True)
# data = data.asfreq('D')
# data
# style()

# plt.title('Closing Stock Price', color="white")
# plt.plot(viz.Date, viz.Close, color="#94F008")
# plt.legend(["Close"], loc ="lower right", facecolor='black', labelcolor='white')

# style()

# plt.scatter(y_pred, y_test, color='red', marker='o')
# plt.scatter(y_test, y_test, color='blue')
# plt.plot(y_test, y_test, color='lime')
# test_pred['Close_Prediction'] = y_pred
# test_pred
# 1.
# test_pred[['Close', 'Close_Prediction']].describe().T

# test_pred['Date'] = pd.to_datetime(test_pred['Date'],format='%m/%d/%Y')
# output = pd.DataFrame(test_pred[['Date', 'Close', 'Close_Prediction']])
# output = output.reset_index()
# output = output.drop('index',axis=1)
# output.set_index('Date', inplace=True)
# output =  output.asfreq('D')
# output

 # output.to_csv('Close_Prediction.csv', index=True)
 # print("CSV successfully saved!")

def predict_stock_price(open_price, high_price, low_price, openy_price, highy_price, lowy_price, openyy_price, highyy_price, lowyy_price):
    input_data = np.array([open_price, high_price, low_price, openy_price, highy_price, lowy_price, openyy_price, highyy_price, lowyy_price]).reshape(1, -1)
    predicted_price = model_lnr.predict(input_data)
    return predicted_price[0]
# User input
user_open = float(input("Enter the Open price: "))
user_high = float(input("Enter the High price: "))
user_low = float(input("Enter the Low price: "))
user_openy = float(input("Enter the Open price Y: "))
user_highy = float(input("Enter the High price Y: "))
user_lowy = float(input("Enter the Low price Y: "))
user_openyy = float(input("Enter the Open price YY: "))
user_highyy = float(input("Enter the High price YY: "))
user_lowyy = float(input("Enter the Low price YY: "))
#user_volume = float(input("Enter the Volume: "))

# Predict using user input
predicted_stock_price = predict_stock_price(user_open, user_high, user_low, user_openy, user_highy, user_lowy, user_openyy, user_highyy, user_lowyy)
print(f"Predicted Close Price: {predicted_stock_price}")


