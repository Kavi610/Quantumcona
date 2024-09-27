import pandas as pd
import numpy as np
import os

# Step 1: Data Handling Plan
def load_data(derivative, expiry_date):
    path = f"./data/{derivative}/"
    filename = f"expiry_{expiry_date}.feather"
    if filename in os.listdir(path):
        data = pd.read_feather(os.path.join(path, filename))
        return data
    else:
        raise FileNotFoundError("Data for this expiry does not exist.")

# Step 2: User Input Handling
def get_user_input():
    derivative = input("Choose derivative (Nifty, BankNifty, FinNifty, BankEx): ")
    expiry_date = input("Enter expiry date (YYYY-MM-DD): ")
    timeframes = input("Choose timeframes (1m, 5m, 1h, 1d), separated by commas: ").split(',')
    strategy = input("Choose strategy (MA, RSI): ")
    
    return derivative, expiry_date, timeframes, strategy

# Step 3: Strategy Implementation
def moving_average_crossover(data, short_window=5, long_window=20):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    
    return signals

def rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 4: Backtesting Across Timeframes
def backtest(signals):
    initial_capital = 10000
    positions = pd.DataFrame(index=signals.index).fillna(0)
    positions['Nifty'] = 100 * signals['positions']  # assuming 100 shares per trade

    portfolio = positions.multiply(signals['price'], axis=0)
    pos_diff = positions.diff()
    
    portfolio['holdings'] = (positions.multiply(signals['price'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['price'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    
    return portfolio

def evaluate_performance(portfolio):
    total_trades = portfolio['holdings'].count()
    total_profit = portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]
    win_rate = (portfolio['total'] > portfolio['total'].shift(1)).mean() * 100
    
    daily_returns = portfolio['total'].pct_change()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    
    return total_trades, total_profit, win_rate, sharpe_ratio

# Main Workflow
if __name__ == "__main__":
    derivative, expiry_date, timeframes, strategy = get_user_input()
    data = load_data(derivative, expiry_date)

    if strategy == "MA":
        signals = moving_average_crossover(data)
    elif strategy == "RSI":
        data['RSI'] = rsi(data)
        signals = data[data['RSI'] < 30]  # Example condition for RSI

    portfolio = backtest(signals)
    
    performance_metrics = evaluate_performance(portfolio)
    print("Performance Metrics:")
    print(f"Total Trades: {performance_metrics[0]}")
    print(f"Total Profit/Loss: {performance_metrics[1]}")
    print(f"Win Rate: {performance_metrics[2]:.2f}%")
    print(f"Sharpe Ratio: {performance_metrics[3]:.2f}")
