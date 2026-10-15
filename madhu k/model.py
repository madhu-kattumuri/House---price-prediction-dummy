import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "mumbai_cleaned.csv")

# Load data
df = pd.read_csv(CSV_PATH)

# Auto-detect columns - nee CSV ki taggattu
# Expected: area, bedroom, bathroom, furnish, price
# Price last column anukundam
if 'price' in df.columns:
    y = df['price']
    X = df.drop('price', axis=1)
else:
    # price last column ayithe
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

# Keep only 4 columns if more exist
if X.shape[1] > 4:
    X = X.iloc[:, :4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def linear_regression(X_train, X_test, y_train, y_test, data):
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(data)
    mse = mean_squared_error(y_test, model.predict(X_test))
    return mse, pred[0]

def decision_tree(X_train, X_test, y_train, y_test, data):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(data)
    mse = mean_squared_error(y_test, model.predict(X_test))
    return mse, pred[0]

def random_forest(X_train, X_test, y_train, y_test, data):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(data)
    mse = mean_squared_error(y_test, model.predict(X_test))
    return mse, pred[0]

def support_vector_regressor(X_train, X_test, y_train, y_test, data):
    model = SVR()
    model.fit(X_train, y_train)
    pred = model.predict(data)
    mse = mean_squared_error(y_test, model.predict(X_test))
    return mse, pred[0]