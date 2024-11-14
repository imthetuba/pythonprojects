import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# This file runs a simple random forest algo (from scilearn) to calculate whether to sell or buy
# on a specific WEEK regarding the SSAB stock using data from yfinance. 
# it then plots the historical data against the prediction model to see if it indeed was 
# a good idea (probably not since the model only looks at three things)

# Step 1: Download data 
stock_data = yf.download("SSAB-A.ST", start="2018-01-01", end="2023-01-01")
stock_data['Return'] = stock_data['Adj Close'].pct_change()
stock_data['SMA_10'] = stock_data['Adj Close'].rolling(window=10).mean()
stock_data['SMA_50'] = stock_data['Adj Close'].rolling(window=50).mean()
stock_data['Volatility_10'] = stock_data['Return'].rolling(window=10).std()
stock_data.dropna(inplace=True)

# Step 2: Define target variable for weekly outlook
stock_data['Target'] = np.where(stock_data['Adj Close'].shift(-5) > stock_data['Adj Close'], 1, 0)
stock_data.dropna(inplace=True)

# Step 3: Feature selection and train-test split 
features = ['Return', 'SMA_10', 'SMA_50', 'Volatility_10']
X = stock_data[features]
y = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 7: Generate predictions for the entire dataset for weekly buy/sell signals
stock_data['Prediction'] = rf_model.predict(scaler.transform(X))

# Plotting weekly buy/sell signals
buy_signals = stock_data[stock_data['Prediction'] == 1]
sell_signals = stock_data[stock_data['Prediction'] == 0]

plt.figure(figsize=(14, 8))
plt.plot(stock_data.index, stock_data['Adj Close'], label='SSAB Adjusted Close Price', color='blue')
plt.scatter(buy_signals.index, buy_signals['Adj Close'], label='Weekly Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(sell_signals.index, sell_signals['Adj Close'], label='Weekly Sell Signal', marker='v', color='red', alpha=1)
plt.plot(stock_data.index, stock_data['SMA_10'], label='10-day SMA', color='orange', linestyle='--')
plt.plot(stock_data.index, stock_data['SMA_50'], label='50-day SMA', color='purple', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (SEK)')
plt.title('SSAB Stock Price with Weekly Model-Based Buy/Sell Signals')
plt.legend()
plt.grid(True)
plt.show()
