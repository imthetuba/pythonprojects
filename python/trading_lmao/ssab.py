# Filter for buy and sell signals
buy_signals = stock_data[stock_data['Prediction'] == 1]
sell_signals = stock_data[stock_data['Prediction'] == 0]

# Plot the historical adjusted close price
plt.figure(figsize=(14, 8))
plt.plot(stock_data.index, stock_data['Adj Close'], label='SSAB Adjusted Close Price', color='blue')

# Plot buy signals
plt.scatter(buy_signals.index, buy_signals['Adj Close'], label='Buy Signal', marker='^', color='green', alpha=1)

# Plot sell signals
plt.scatter(sell_signals.index, sell_signals['Adj Close'], label='Sell Signal', marker='v', color='red', alpha=1)

# Adding moving averages for context
plt.plot(stock_data.index, stock_data['SMA_10'], label='10-day SMA', color='orange', linestyle='--')
plt.plot(stock_data.index, stock_data['SMA_50'], label='50-day SMA', color='purple', linestyle='--')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (SEK)')
plt.title('SSAB Stock Price with Model-Based Buy/Sell Signals')
plt.legend()
plt.grid(True)
plt.show()
