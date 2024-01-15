import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from db_util import DataBase

class FeatureExtractor:
    NECESSARY_COLUMNS = ['open', 'high', 'low', 'close', 'adj_close', 'volume']

    def __init__(self, df):
        self.validate_dataframe(df)
        self.df = df.copy()
        self.DF_VOLUME_DIVISOR = self.df['volume'].mean()

    @staticmethod
    def validate_dataframe(df_):
        for column in FeatureExtractor.NECESSARY_COLUMNS:
            if column not in df_.columns:
                raise ValueError(f"DataFrame must include {column} columns")

    def accumulation_distribution_line(self, period=14):
        close_minus_low = self.df['close'] - self.df['low']
        high_minus_close = self.df['high'] - self.df['close']
        high_minus_low = self.df['high'] - self.df['low']
        money_flow = (close_minus_low - high_minus_close) / high_minus_low
        money_flow[high_minus_low < 0.01] = 0
        self.df['money_flow_volume'] = money_flow * (self.df['volume'] / self.DF_VOLUME_DIVISOR)
        ADL = self.df['money_flow_volume'].rolling(period).sum()
        return ADL

    def aroon(self, period=25):
        aroon_high = self.df['high'].rolling(period+1).apply(lambda x: x.argmax()) / period * 100
        aroon_low = self.df['low'].rolling(period+1).apply(lambda x: x.argmin()) / period * 100
        aroon_oscillator = aroon_high - aroon_low
        return aroon_high, aroon_low, aroon_oscillator

    def average_true_range(self, period=14):
        df_temp = pd.DataFrame({
            't1': self.df['high'] - self.df['low'],
            't2': np.abs(self.df['high'] - self.df['close'].shift(1)),
            't3': np.abs(self.df['low'] - self.df['close'].shift(1))
        })
        self.df['true_range'] = df_temp.max(axis=1)
        return self.df['true_range'].ewm(alpha=1/period, adjust=False).mean()

    def average_true_range_percentage(self, period=14):
        if 'true_range' not in self.df.columns:
            self.average_true_range(period=period)
        tr_percentage = self.df['true_range'] / self.df['close'] * 100
        return tr_percentage.ewm(alpha=1/period, adjust=False).mean()

    def average_directional_index(self, period=14):
        if 'true_range' not in self.df.columns:
            self.average_true_range(period=period)
        alpha = 1 / period

        df_temp = pd.DataFrame({
            '+DM': self.df['high'] - self.df['high'].shift(1),
            '-DM': self.df['low'].shift(1) - self.df['low']
        })
        df_temp['+DM'] = np.where((df_temp['+DM'] > df_temp['-DM']) & (df_temp['+DM'] > 0), df_temp['+DM'], 0.0)
        df_temp['-DM'] = np.where((df_temp['-DM'] > df_temp['+DM']) & (df_temp['-DM'] > 0), df_temp['-DM'], 0.0)

        df_temp['TR'] = self.df['true_range'].ewm(alpha=alpha, adjust=False).mean()
        df_temp['+DI'] = df_temp['+DM'].ewm(alpha=alpha, adjust=False).mean() / df_temp['TR'] * 100
        df_temp['-DI'] = df_temp['-DM'].ewm(alpha=alpha, adjust=False).mean() / df_temp['TR'] * 100

        df_temp['DX'] = np.abs(df_temp['+DI'] - df_temp['-DI']) / (df_temp['+DI'] + df_temp['-DI']) * 100
        df_temp['ADX'] = df_temp['DX'].ewm(alpha=alpha, adjust=False).mean()

        return df_temp['+DI'], df_temp['-DI'], df_temp['ADX']

    def balance_of_power(self, period=14):
        bop = (self.df['close'] - self.df['open']) / (self.df['high'] - self.df['low'])
        return bop.rolling(period).mean()

    def bollinger_band(self, period=20):
        self.df['bollinger_middle'] = self.df['close'].rolling(period).mean()
        std = self.df['close'].rolling(period).std()
        self.df['bollinger_upper'] = self.df['bollinger_middle'] + 2.0 * std
        self.df['bollinger_lower'] = self.df['bollinger_middle'] - 2.0 * std
        return self.df['bollinger_upper'], self.df['bollinger_middle'], self.df['bollinger_lower']

    def bollinger_bandwidth(self, period=20):
        if 'bollinger_middle' not in self.df.columns:
            self.bollinger_band(period=period)
        return (self.df['bollinger_upper'] - self.df['bollinger_lower']) / self.df['bollinger_middle'] * 100

    def bollinger_percentage(self, period=20):
        if 'bollinger_middle' not in self.df.columns:
            self.bollinger_band(period=period)
        bandwidth = self.df['bollinger_upper'] - self.df['bollinger_lower']
        return (self.df['close'] - self.df['bollinger_lower']) / bandwidth

    def chainkin_money_flow(self, period=20):
        if 'money_flow_volume' not in self.df.columns:
            self.accumulation_distribution_line()
        volume_sum = self.df['volume'].rolling(period).sum() / self.DF_VOLUME_DIVISOR
        return self.df['money_flow_volume'].rolling(period).sum() - volume_sum

    def chainkin_oscillator(self, periods=(20, 3, 10)):
        if 'money_flow_volume' not in self.df.columns:
            self.accumulation_distribution_line()
        adl = self.df['money_flow_volume'].rolling(periods[0]).sum()
        return adl.ewm(span=periods[1], adjust=False).mean() - adl.ewm(span=periods[2], adjust=False).mean()

    def change_percentage(self, col='close', lookback=1):
        return (self.df[col] - self.df[col].shift(lookback)) / self.df[col].shift(lookback) * 100

    def change_direction(self, col='close', lookback=1):
        return np.where(self.df[col] > self.df[col].shift(lookback), 1.0, 0.0)

    def force_index(self, period=14):
        fi = (self.df['close'] - self.df['close'].shift(1)) * (self.df['volume'] / self.DF_VOLUME_DIVISOR)
        return fi.ewm(span=period, adjust=False).mean()

    def macd(self, periods=(12, 26, 9)):
        macd_line = (
            self.df['close'].ewm(span=periods[0], adjust=False).mean() -
            self.df['close'].ewm(span=periods[1], adjust=False).mean()
        )

        signal_line = macd_line.ewm(span=periods[2], adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram

    def stochastic_oscillator(self, period=14):
        lowest_low = self.df['low'].rolling(period).min()
        highest_high = self.df['high'].rolling(period).max()

        k_percent = (self.df['close'] - lowest_low) / (highest_high - lowest_low) * 100
        d_percent = k_percent.rolling(3).mean()
        return k_percent, d_percent


if __name__ == "__main__":
    matplotlib.use('Qt5Agg')
    db = DataBase(db_path='../data/project1.db')
    df = db.get_data(ticker='^GSPC', period='2y', interval='1d')
    print(df.shape)

    feature_extractor = FeatureExtractor(df)
    t1 = feature_extractor.change_percentage()

    ax = df['close'].plot(color='g')
    t1.plot(secondary_y=True, ax=ax, color='b')

    plt.show()