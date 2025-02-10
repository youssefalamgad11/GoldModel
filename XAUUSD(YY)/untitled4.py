import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression

# Load and clean the data
def clean_data(file_path):
    data = pd.read_csv(file_path)
    # Remove commas and convert columns to numeric
    numeric_cols = ['Price', 'Open', 'High', 'Low']
    for col in numeric_cols:
        data[col] = data[col].str.replace(',', '').astype(float)

    # Clean and convert "Change %" column
    data['Change %'] = data['Change %'].str.replace('%', '').astype(float) / 100.0

    # Drop the "Vol." column as it has too many missing values
    data.drop(columns=['Vol.'], inplace=True)

    # Convert "Date" to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

    # Extract date features
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    return data

# Load the dataset
file_path = 'XAU_USD2.csv'  # Update with your file path
data = clean_data(file_path)

# Features and target for the models
X = data[['Open', 'High', 'Low', 'Day', 'Month', 'Year']]
y_price = data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

# Model 1: Random Forest Regressor for predicting Price
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the first model
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Model 1 (Random Forest) MAE: {mae_rf}")

# Model 2: Neural Network for predicting Price
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
    
])

# Compile the model
nn_model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the second model
y_pred_nn = nn_model.predict(X_test)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
print(f"Model 2 (Neural Network) MAE: {mae_nn}")

# Model 3: Linear Regression for predicting Price
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate the third model
y_pred_lr = lr_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f"Model 3 (Linear Regression) MAE: {mae_lr}")

# Use Pipeline and GridSearchCV for XGBoost model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', xgb.XGBRegressor(objective='reg:squarederror'))
])

# Define hyperparameters for tuning
params = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
print(f"Best Model (XGBoost) MAE: {mae_best}")

# Interactive prediction function
def predict():
    print("Enter the following inputs:")
    date_input = input("Date (YYYY-MM-DD): ")
    date = pd.to_datetime(date_input)
    open_price = float(input("Open price: "))
    high_price = float(input("High price: "))
    low_price = float(input("Low price: "))

    # Extract date features
    day = date.day
    month = date.month
    year = date.year
    
    input_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Day': [day],
        'Month': [month],
        'Year': [year]
    })

    # Predict closing price using the Random Forest model
    predicted_price_rf = rf_model.predict(input_data)[0]
    print(f"Predicted Closing Price (Random Forest): {predicted_price_rf}")

    # Predict closing price using the Neural Network model
    predicted_price_nn = nn_model.predict(input_data)[0][0]
    print(f"Predicted Closing Price (Neural Network): {predicted_price_nn}")

    # Predict closing price using the Linear Regression model
    predicted_price_lr = lr_model.predict(input_data)[0]
    print(f"Predicted Closing Price (Linear Regression): {predicted_price_lr}")

    # Predict closing price using the XGBoost model
    predicted_price_best = best_model.predict(input_data)[0]
    print(f"Predicted Closing Price (Best Model - XGBoost): {predicted_price_best}")

#Uncomment the line below to run the prediction function
predict()