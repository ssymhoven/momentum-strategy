import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create images folder if not exists
if not os.path.exists('images'):
    os.makedirs('images')


def calculate_signals(data, short_window=50, long_window=200):
    """
    Calculate buy and sell signals based on moving averages.
    """
    # Calculate short-term and long-term moving averages
    data['Short_MA'] = data['Adj Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Adj Close'].rolling(window=long_window).mean()

    # Initialize signal column: 1 for buy (long), -1 for sell (short), 0 for no position
    data['Signal'] = 0

    # Use iloc for positional slicing since we want to slice by index position
    data.iloc[short_window:, data.columns.get_loc('Signal')] = np.where(
        data['Short_MA'].iloc[short_window:] > data['Long_MA'].iloc[short_window:], 1, 0)
    data.iloc[short_window:, data.columns.get_loc('Signal')] = np.where(
        data['Short_MA'].iloc[short_window:] < data['Long_MA'].iloc[short_window:], -1,
        data['Signal'].iloc[short_window:])

    # Shift signal to avoid lookahead bias (signal should be on the next day)
    data['Position'] = data['Signal'].shift()

    return data


def calculate_performance(data):
    """
    Calculate the performance of the strategy by applying the buy/sell signals to the data.
    Track each trade with entry price, exit price, and performance.
    """
    # Calculate daily returns
    data['Returns'] = data['Adj Close'].pct_change()

    # Initialize performance-related columns
    data['Strategy Returns'] = 0.0  # Will hold cumulative returns for the strategy
    data['Position'] = 0  # Track if we have an open position (1 for long, -1 for short, 0 for no position)

    # Create a list to track trades and build a DataFrame later
    trades_list = []

    # Track when positions are opened and closed
    position_opened = False
    position_type = 0  # 1 for long, -1 for short
    entry_price = 0
    entry_date = None
    position_opening_points = []
    position_closing_points = []

    for i in range(1, len(data)):
        signal = data['Signal'].iloc[i]

        # Check if we should open or close a position
        if signal == 1 and not position_opened:  # Buy signal and no open position
            position_opened = True
            position_type = 1  # Long position
            entry_price = data['Adj Close'].iloc[i]
            entry_date = data.index[i]
            data.loc[data.index[i], 'Position'] = 1  # Open long position
            position_opening_points.append((data.index[i], entry_price))  # Track the buy point

        elif signal == -1 and not position_opened:  # Sell signal and no open position
            position_opened = True
            position_type = -1  # Short position
            entry_price = data['Adj Close'].iloc[i]
            entry_date = data.index[i]
            data.loc[data.index[i], 'Position'] = -1  # Open short position
            position_opening_points.append((data.index[i], entry_price))  # Track the sell point

        elif signal != position_type and position_opened:  # Close the position
            exit_price = data['Adj Close'].iloc[i]
            exit_date = data.index[i]

            # Calculate returns from the open position
            if position_type == 1:  # Close long position
                position_return = (exit_price / entry_price) - 1
                trade_type = 'Long'
            else:  # Close short position
                position_return = (entry_price / exit_price) - 1
                trade_type = 'Short'

            # Track the strategy returns
            data.loc[data.index[i], 'Strategy Returns'] = position_return

            # Track the close position point
            position_closing_points.append((data.index[i], exit_price))

            # Record the trade in the trades list
            trades_list.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Type': trade_type,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Performance': position_return * 100  # Convert to percentage
            })

            # Close position
            position_opened = False
            entry_price = 0

    # Calculate cumulative strategy returns (ignore days with no position)
    data['Cumulative Strategy Returns'] = (1 + data['Strategy Returns']).cumprod() - 1
    data['Cumulative Returns'] = (1 + data['Returns']).cumprod() - 1

    # Create a DataFrame for the trades
    trades = pd.DataFrame(trades_list)

    return data, position_opening_points, position_closing_points, trades


def plot_signals(data, name, opening_points, closing_points):
    """
    Plot the original timeseries along with buy and sell signals.
    Only plot the actual buy/sell execution points (opening and closing positions).
    """
    plt.figure(figsize=(14, 8))

    # Plot adjusted close price
    plt.plot(data.index, data['Adj Close'], label='Price', color='black')

    # Plot Buy and Sell signals based on actual positions being opened and closed
    buy_signals = [point for point in opening_points if data['Signal'].loc[point[0]] == 1]
    sell_signals = [point for point in opening_points if data['Signal'].loc[point[0]] == -1]

    # Plot opening signals
    for point in buy_signals:
        plt.plot(point[0], point[1], '^', markersize=10, color='green', label='Open Long Position')

    for point in sell_signals:
        plt.plot(point[0], point[1], 'v', markersize=10, color='red', label='Open Short Position')

    # Plot closing signals
    for point in closing_points:
        plt.plot(point[0], point[1], 'o', markersize=10, color='blue', label='Close Position')

    plt.title(f'{name}: Buy/Sell Signals with Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f'images/{name}_signals.png')
    plt.close()


def analyze_strategy(ticker, name, short_window=50, long_window=200, use_seasonality=False):
    # Fetch historical data from yfinance
    data = yf.download(ticker, start="2000-01-01", end="2024-01-01")

    # Calculate buy/sell signals
    data = calculate_signals(data, short_window=short_window, long_window=long_window)

    if use_seasonality:
        # Restrict buy signals to winter (Nov-Apr) and sell signals to summer (May-Oct)
        data['Month'] = data.index.month
        winter_months = [11, 12, 1, 2, 3, 4]
        summer_months = [5, 6, 7, 8, 9, 10]

        data.loc[~data['Month'].isin(winter_months), 'Signal'] = 0  # No buy signals in summer
        data.loc[~data['Month'].isin(summer_months), 'Signal'] = 0  # No sell signals in winter

    # Calculate strategy performance and collect opening and closing points and trades
    data, opening_points, closing_points, trades = calculate_performance(data)

    # Plot buy and sell signals
    plot_signals(data, name, opening_points, closing_points)

    # Display performance metrics
    total_return = data['Cumulative Returns'].iloc[-1]
    strategy_return = data['Cumulative Strategy Returns'].iloc[-1]
    print(f"Total return of {name} (Buy/Hold): {total_return * 100:.2f}%")
    print(f"Total return of {name} Strategy: {strategy_return * 100:.2f}%")

    # Save trades to Excel
    trades.to_excel(f'{name}_trades.xlsx', index=False)
    print(f"Trades saved to {name}_trades.xlsx")

    return data, trades


# Run the analysis for S&P 500 and Stoxx 600 with and without seasonality
print("Running analysis without seasonality...")
analyze_strategy('^GSPC', 'S&P 500', use_seasonality=False)
print("Running analysis with seasonality (Winter/Summer strategy)...")
#analyze_strategy('^GSPC', 'S&P 500', use_seasonality=True)

print("Buy/sell signals, performance analysis, and trades export completed. Charts saved in the images folder.")
