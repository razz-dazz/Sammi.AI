import pandas as pd
import numpy as np

def sma(df, period=14, column='close'):
    """Simple Moving Average"""
    return df[column].rolling(window=period).mean()

def ema(df, period=14, column='close'):
    """Exponential Moving Average"""
    return df[column].ewm(span=period, adjust=False).mean()

def rsi(df, period=14, column='close'):
    """Relative Strength Index"""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(df, fast=12, slow=26, signal=9, column='close'):
    """Moving Average Convergence Divergence"""
    ema_fast = ema(df, fast, column)
    ema_slow = ema(df, slow, column)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(df, period=20, column='close', num_std=2):
    """Bollinger Bands"""
    sma_ = sma(df, period, column)
    std = df[column].rolling(window=period).std()
    upper = sma_ + num_std * std
    lower = sma_ - num_std * std
    return upper, sma_, lower

def stochastic_oscillator(df, k_period=14, d_period=3):
    """Stochastic Oscillator %K and %D"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d

def atr(df, period=14):
    """Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def parabolic_sar(df, af=0.02, max_af=0.2):
    """Parabolic SAR (simplified version)"""
    sar = df['low'].copy()
    long = True
    ep = df['high'][0]
    af_val = af
    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        if long:
            sar.iloc[i] = prev_sar + af_val * (ep - prev_sar)
            if df['low'].iloc[i] < sar.iloc[i]:
                long = False
                sar.iloc[i] = ep
                ep = df['low'].iloc[i]
                af_val = af
            else:
                if df['high'].iloc[i] > ep:
                    ep = df['high'].iloc[i]
                    af_val = min(af_val + af, max_af)
        else:
            sar.iloc[i] = prev_sar + af_val * (ep - prev_sar)
            if df['high'].iloc[i] > sar.iloc[i]:
                long = True
                sar.iloc[i] = ep
                ep = df['high'].iloc[i]
                af_val = af
            else:
                if df['low'].iloc[i] < ep:
                    ep = df['low'].iloc[i]
                    af_val = min(af_val + af, max_af)
    return sar

def fibonacci_retracement(df, levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
    """Fibonacci Retracement Levels (using last swing high/low)"""
    max_price = df['high'].max()
    min_price = df['low'].min()
    diff = max_price - min_price
    retracements = {f"{int(l*100)}%": max_price - l * diff for l in levels}
    return retracements

class TechnicalAnalyzer:
    def __init__(self, df):
        self.df = df

    def calculate_indicators(self):
        self.df['SMA20'] = sma(self.df, 20)
        self.df['RSI14'] = rsi(self.df, 14)
        self.df['CCI'] = commodity_channel_index(self.df, 20)
        self.df['Williams_%R'] = williams_r(self.df, 14)
        self.df['ROC'] = roc(self.df, 12)
        self.df['Momentum'] = self.df['close'].diff(4)
        self.df['OBV'] = obv(self.df)
        self.df['RVI'] = rvi(self.df, 10)
        self.df['CMO'] = cmo(self.df, 14)
        self.df['Williams_%R'] = williams_r(self.df, 14)
        self.df['ROC'] = roc(self.df, 12)
        self.df['Momentum'] = self.df['close'].diff(4)
        self.df['OBV'] = obv(self.df)
        self.df['RVI'] = rvi(self.df, 10)
        self.df['CMO'] = cmo(self.df, 14)
        self.df['Williams_%R'] = williams_r(self.df, 14)
        self.df['ROC'] = roc(self.df, 12)
        self.df['Momentum'] = self.df['close'].diff(4)
        self.df['OBV'] = obv(self.df)
        self.df['RVI'] = rvi(self.df, 10)
        self.df['CMO'] = cmo(self.df, 14)
        self.df['Williams_%R'] = williams_r(self.df, 14)
        self.df['ROC'] = roc(self.df, 12)
        self.df['Momentum'] = self.df['close'].diff(4)
        self.df['OBV'] = obv(self.df)
        self.df['RVI'] = rvi(self.df, 10)
        self.df['CMO'] = cmo(self.df, 14)
        self.df['Williams_%R'] = williams_r(self.df, 14)
        self.df['ROC'] = roc(self.df, 12)
        self.df['Momentum'] = self.df['close'].diff(4)
        self.df['OBV'] = obv(self.df)
        self.df['RVI'] = rvi(self.df, 10)
        self.df['CMO'] = cmo(self.df, 14)
        self.df['Williams_%R'] = williams_r(self.df, 14)
        self.df['ROC'] = roc(self.df, 12)
        self.df['Momentum'] = self.df['close'].diff(4)
        self.df['OBV'] = obv(self.df)
        self.df['RVI'] = rvi(self.df, 10)
        self.df['CMO'] = cmo(self.df, 14)
        self.df['Williams_%R'] = williams_r(self.df, 14)
        self.df['ROC'] = roc(self.df, 12)
        self.df['Momentum'] = self.df['close'].diff(4)
        self.df['OBV'] = obv(self.df)
        self.df['RVI'] = rvi(self.df, 10)
        self.df['CMO'] = cmo(self.df, 14)
    
    # --- Candlestick Pattern Detection Methods ---

    def detect_hammer(self):
        body = abs(self.df['close'] - self.df['open'])
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        return (lower_shadow > 2 * body) & (upper_shadow < body)

    def detect_inverted_hammer(self):
        body = abs(self.df['close'] - self.df['open'])
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        return (upper_shadow > 2 * body) & (lower_shadow < body)

    def detect_hanging_man(self):
        body = abs(self.df['close'] - self.df['open'])
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        return (lower_shadow > 2 * body) & (upper_shadow < body)

    def detect_shooting_star(self):
        body = abs(self.df['close'] - self.df['open'])
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        return (upper_shadow > 2 * body) & (lower_shadow < body)

    def detect_doji(self):
        body = abs(self.df['close'] - self.df['open'])
        return body < (0.1 * (self.df['high'] - self.df['low']))

    def detect_bullish_engulfing(self):
        prev = self.df.shift(1)
        return (
            (prev['close'] < prev['open']) &
            (self.df['close'] > self.df['open']) &
            (self.df['close'] > prev['open']) &
            (self.df['open'] < prev['close'])
        )

    def detect_bearish_engulfing(self):
        prev = self.df.shift(1)
        return (
            (prev['close'] > prev['open']) &
            (self.df['close'] < self.df['open']) &
            (self.df['open'] > prev['close']) &
            (self.df['close'] < prev['open'])
        )

    def detect_morning_star(self):
        prev1 = self.df.shift(1)
        prev2 = self.df.shift(2)
        return (
            (prev2['close'] < prev2['open']) &
            (abs(prev1['close'] - prev1['open']) < 0.3 * (prev1['high'] - prev1['low'])) &
            (self.df['close'] > self.df['open']) &
            (self.df['close'] > ((prev2['open'] + prev2['close']) / 2))
        )

    def detect_evening_star(self):
        prev1 = self.df.shift(1)
        prev2 = self.df.shift(2)
        return (
            (prev2['close'] > prev2['open']) &
            (abs(prev1['close'] - prev1['open']) < 0.3 * (prev1['high'] - prev1['low'])) &
            (self.df['close'] < self.df['open']) &
            (self.df['close'] < ((prev2['open'] + prev2['close']) / 2))
        )

    def detect_spinning_top(self):
        body = abs(self.df['close'] - self.df['open'])
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        return (
            (body < 0.3 * (self.df['high'] - self.df['low'])) &
            (upper_shadow > body) &
            (lower_shadow > body)
        )

    def detect_piercing_line(self):
        prev = self.df.shift(1)
        return (
            (prev['close'] < prev['open']) &
            (self.df['open'] < prev['close']) &
            (self.df['close'] > self.df['open']) &
            (self.df['close'] > ((prev['open'] + prev['close']) / 2))
        )

    def detect_dark_cloud_cover(self):
        prev = self.df.shift(1)
        return (
            (prev['close'] > prev['open']) &
            (self.df['open'] > prev['close']) &
            (self.df['close'] < self.df['open']) &
            (self.df['close'] < ((prev['open'] + prev['close']) / 2))
        )

    def detect_three_white_soldiers(self):
        return (
            (self.df['close'] > self.df['open']) &
            (self.df['close'] > self.df['close'].shift(1)) &
            (self.df['close'].shift(1) > self.df['close'].shift(2)) &
            (self.df['close'].shift(2) > self.df['open'].shift(2))
        )

    def detect_three_black_crows(self):
        return (
            (self.df['close'] < self.df['open']) &
            (self.df['close'] < self.df['close'].shift(1)) &
            (self.df['close'].shift(1) < self.df['close'].shift(2)) &
            (self.df['close'].shift(2) < self.df['open'].shift(2))
        )

    def detect_patterns(self):
        patterns = {
            'hammer': self.detect_hammer(),
            'inverted_hammer': self.detect_inverted_hammer(),
            'hanging_man': self.detect_hanging_man(),
            'shooting_star': self.detect_shooting_star(),
            'doji': self.detect_doji(),
            'bullish_engulfing': self.detect_bullish_engulfing(),
            'bearish_engulfing': self.detect_bearish_engulfing(),
            'morning_star': self.detect_morning_star(),
            'evening_star': self.detect_evening_star(),
            'spinning_top': self.detect_spinning_top(),
            'piercing_line': self.detect_piercing_line(),
            'dark_cloud_cover': self.detect_dark_cloud_cover(),
            'three_white_soldiers': self.detect_three_white_soldiers(),
            'three_black_crows': self.detect_three_black_crows()
        }
        return patterns

    def generate_signals(self):
        # Ensure indicators and patterns are calculated
        self.calculate_indicators()
        patterns = self.detect_patterns()

        # Initialize signal column
        self.df['signal'] = 0

        # Example Buy Signal: Bullish pattern + RSI < 30 + price > SMA20
        bullish = (
            patterns['hammer'] |
            patterns['bullish_engulfing'] |
            patterns['morning_star'] |
            patterns['piercing_line'] |
            patterns['three_white_soldiers']
        )
        buy_condition = bullish & (self.df['RSI14'] < 30) & (self.df['close'] > self.df['SMA20'])

        # Example Sell Signal: Bearish pattern + RSI > 70 + price < SMA20
        bearish = (
            patterns['shooting_star'] |
            patterns['bearish_engulfing'] |
            patterns['evening_star'] |
            patterns['dark_cloud_cover'] |
            patterns['three_black_crows']
        )
        sell_condition = bearish & (self.df['RSI14'] > 70) & (self.df['close'] < self.df['SMA20'])

        self.df.loc[buy_condition, 'signal'] = 1
        self.df.loc[sell_condition, 'signal'] = -1

        return self.df[['signal']]

    # --- Additional Analysis Methods ---

    def support_resistance_levels(df, window=14):
        """Support and Resistance Levels (pivot points)"""
        pivot = df['close'].rolling(window=window).mean()
        support = pivot - (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()) / 2
        resistance = pivot + (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()) / 2
        return support, resistance

    def add_support_resistance(self):
        self.df['support'], self.df['resistance'] = support_resistance_levels(self.df)

    def momentum_indicator(df, column='close'):
        """Momentum Indicator (Rate of Change)"""
        return df[column].diff(4)  # Compare close price with that of 4 periods ago

    def add_momentum(self):
        self.df['momentum'] = momentum_indicator(self.df)

    def volume_weighted_average_price(df, window=14):
        """Volume Weighted Average Price (VWAP)"""
        vwap = (df['close'] * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        return vwap

    def add_vwap(self):
        self.df['VWAP'] = volume_weighted_average_price(self.df)

    def analyze_trade_volume(df):
        """Analyze trade volume and detect unusual spikes"""
        volume_avg = df['volume'].rolling(window=20).mean()
        volume_std = df['volume'].rolling(window=20).std()
        threshold = volume_avg + 2 * volume_std
        return df['volume'] > threshold

    def add_volume_spike_indicator(self):
        self.df['volume_spike'] = analyze_trade_volume(self.df)

    def elder_ray_index(df):
        """Elder Ray Index (Bull and Bear Power)"""
        hl2 = (df['high'] + df['low']) / 2
        ema200 = ema(df, 200)
        bull_power = df['high'] - ema200
        bear_power = df['low'] - ema200
        return bull_power, bear_power

    def add_elder_ray(self):
        self.df['bull_power'], self.df['bear_power'] = elder_ray_index(self.df)

    def donchian_channels(df, window=20):
        """Donchian Channels (Trend Following Indicator)"""
        upper = df['high'].rolling(window=window).max()
        lower = df['low'].rolling(window=window).min()
        middle = (upper + lower) / 2
        return upper, middle, lower

    def add_donchian_channels(self):
        self.df['donchian_upper'], self.df['donchian_middle'], self.df['donchian_lower'] = donchian_channels(self.df)

    def average_true_range(df, window=14):
        """Average True Range (Volatility Measure)"""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    def add_average_true_range(self):
        self.df['ATR'] = average_true_range(self.df)

    def commodity_channel_index(df, window=20):
        """Commodity Channel Index (CCI)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_typical = sma(df, window, column='close')
        mean_deviation = (typical_price - sma_typical).abs().rolling(window=window).mean()
        cci = (typical_price - sma_typical) / (0.015 * mean_deviation)
        return cci

    def add_commodity_channel_index(self):
        self.df['CCI'] = commodity_channel_index(self.df)

    def signals_with_added_indicators(self):
        # Recalculate signals with added indicators like VWAP, Momentum, etc.
        self.calculate_indicators()
        patterns = self.detect_patterns()

        # Debug: Check the calculated patterns
        print("Detected Patterns:")
        for pattern_name, detected in patterns.items():
            if detected.any():
                print(f"{pattern_name}: {detected.sum()} instance(s)")

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

    def rvi(df, period=10):
        num = ((df['close'] - df['open']).rolling(window=period).mean())
        denom = ((df['high'] - df['low']).rolling(window=period).mean())
        return num / denom

    def cmo(df, period=14, column='close'):
        diff = df[column].diff()
        up = diff.where(diff > 0, 0).abs().rolling(window=period).sum()
        down = diff.where(diff < 0, 0).abs().rolling(window=period).sum()
        return 100 * (up - down) / (up + down)

analyzer = TechnicalAnalyzer(df)
signals = analyzer.generate_signals()
print(signals)

# Plotting signals on the closing price chart
fig = px.line(analyzer.df, x=analyzer.df.index, y='close', title='Trading Signals',
              labels={'x': 'Date', 'close': 'Close Price    
fig.add_scatter(x=signals.index[signals['signal'] == 1], y=signals['signal'][signals['signal'] == 1],
                mode='markers', marker=dict(color='green', size=10), name='Buy Signal')
fig.add_scatter(x=signals.index[signals['signal'] == -1], y=signals['signal'][signals['signal'] == -1],
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
