import pandas as pd
import numpy as np
import plotly.express as px

# Example DataFrame loading (replace with your data source)
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='Date')

class TechnicalAnalyzer:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self):
        # Example pattern detection (replace with your actual logic)
        patterns = {
            'hammer': pd.Series([False]*len(self.df), index=self.df.index),
            'bullish_engulfing': pd.Series([False]*len(self.df), index=self.df.index),
            'morning_star': pd.Series([False]*len(self.df), index=self.df.index),
            'piercing_line': pd.Series([False]*len(self.df), index=self.df.index),
            'three_white_soldiers': pd.Series([False]*len(self.df), index=self.df.index),
            'shooting_star': pd.Series([False]*len(self.df), index=self.df.index),
            'bearish_engulfing': pd.Series([False]*len(self.df), index=self.df.index),
            'evening_star': pd.Series([False]*len(self.df), index=self.df.index),
            'dark_cloud_cover': pd.Series([False]*len(self.df), index=self.df.index),
            'three_black_crows': pd.Series([False]*len(self.df), index=self.df.index),
        }

        # Debug: Check the calculated patterns
        print("Detected Patterns:")
        for pattern_name, detected in patterns.items():
            if detected.any():
                print(f"{pattern_name}: {detected.sum()} instance(s)")

        # Example indicators (replace with your actual calculations)
        self.df['RSI14'] = self.rsi(self.df['close'], window=14)
        self.df['SMA20'] = self.df['close'].rolling(window=20).mean()
        self.df['momentum'] = self.df['close'].diff()

        # Initialize signal column
        self.df['signal'] = 0

        # Refined Buy Signal: Bullish pattern + RSI < 30 + price > SMA20 + Momentum positive
        bullish = (
            patterns['hammer'] |
            patterns['bullish_engulfing'] |
            patterns['morning_star'] |
            patterns['piercing_line'] |
            patterns['three_white_soldiers']
        )
        buy_condition = bullish & (self.df['RSI14'] < 30) & (self.df['close'] > self.df['SMA20']) & (self.df['momentum'] > 0)

        # Refined Sell Signal: Bearish pattern + RSI > 70 + price < SMA20 + Momentum negative
        bearish = (
            patterns['shooting_star'] |
            patterns['bearish_engulfing'] |
            patterns['evening_star'] |
            patterns['dark_cloud_cover'] |
            patterns['three_black_crows']
        )
        sell_condition = bearish & (self.df['RSI14'] > 70) & (self.df['close'] < self.df['SMA20']) & (self.df['momentum'] < 0)

        self.df.loc[buy_condition, 'signal'] = 1
        self.df.loc[sell_condition, 'signal'] = -1

        return self.df[['signal']]

    @staticmethod
    def obv(df):
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

    @staticmethod
    def rvi(df, period=10):
        num = ((df['close'] - df['open']).rolling(window=period).mean())
        denom = ((df['high'] - df['low']).rolling(window=period).mean())
        return num / denom

    @staticmethod
    def cmo(df, period=14, column='close'):
        diff = df[column].diff()
        up = diff.where(diff > 0, 0).abs().rolling(window=period).sum()
        down = diff.where(diff < 0, 0).abs().rolling(window=period).sum()
        return 100 * (up - down) / (up + down)

# Example DataFrame for demonstration (replace with your data)
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='Date')
# For now, let's create a dummy DataFrame:
df = pd.DataFrame({
    'open': np.random.rand(100) * 100,
    'high': np.random.rand(100) * 100,
    'low': np.random.rand(100) * 100,
    'close': np.random.rand(100) * 100,
    'volume': np.random.randint(100, 1000, 100)
}, index=pd.date_range('2024-01-01', periods=100))

analyzer = TechnicalAnalyzer(df)
signals = analyzer.generate_signals()
print(signals)

# Plotting signals on the closing price chart
fig = px.line(analyzer.df, x=analyzer.df.index, y='close', title='Trading Signals',
              labels={'x': 'Date', 'close': 'Close Price'})
fig.add_scatter(x=signals.index[signals['signal'] == 1], 
                y=analyzer.df['close'][signals['signal'] == 1],
                mode='markers', marker=dict(color='green', size=10), name='Buy Signal')
fig.add_scatter(x=signals.index[signals['signal'] == -1], 
                y=analyzer.df['close'][signals['signal'] == -1],
                mode='markers', marker=dict(color='red', size=10), name='Sell Signal')
fig.show()

def backtest_signals(df, initial_cash=10000):
    cash = initial_cash
    position = 0
    equity_curve = []
    for i in range(1, len(df)):
        if df['signal'].iloc[i-1] == 1 and position == 0:
            # Buy at open of this bar
            position = cash / df['open'].iloc[i]
            cash = 0
        elif df['signal'].iloc[i-1] == -1 and position > 0:
            # Sell at open of this bar
            cash = position * df['open'].iloc[i]
            position = 0
        equity = cash + position * df['close'].iloc[i]
        equity_curve.append(equity)
    return equity_curve
