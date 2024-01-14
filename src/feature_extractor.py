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

    def accumulation_distribution_line(self, window=14):
        close_minus_low = self.df['close'] - self.df['low']
        high_minus_close = self.df['high'] - self.df['close']
        high_minus_low = self.df['high'] - self.df['low']
        money_flow = (close_minus_low - high_minus_close) / high_minus_low
        money_flow[high_minus_low < 0.01] = 0
        self.df['money_flow_volume'] = money_flow * (self.df['volume'] / self.DF_VOLUME_DIVISOR)
        ADL = self.df['money_flow_volume'].rolling(window=window).sum()
        return ADL

    def aroon(self, window=25):
        aroon_high = self.df['high'].rolling(window+1).apply(lambda x: x.argmax()) / window * 100
        aroon_low = self.df['low'].rolling(window+1).apply(lambda x: x.argmin()) / window * 100
        aroon_oscillator = aroon_high - aroon_low
        return aroon_high, aroon_low, aroon_oscillator

    def average_true_range(self, window=14):
        df_temp = pd.DataFrame({
            't1': self.df['high'] - self.df['low'],
            't2': np.abs(self.df['high'] - self.df['close'].shift(1)),
            't3': np.abs(self.df['low'] - self.df['close'].shift(1))
        })
        self.df['true_range'] = df_temp.max(axis=1)
        return self.df['true_range'].ewm(alpha=1/window, adjust=False).mean()

    def average_true_range_percentage(self, window=14):
        if 'true_range' not in self.df.columns:
            self.average_true_range(window=window)
        tr_percentage = self.df['true_range'] / self.df['close'] * 100
        return tr_percentage.ewm(alpha=1/window, adjust=False).mean()

    def average_directional_index(self, window=14):
        if 'true_range' not in self.df.columns:
            self.average_true_range(window=window)
        alpha = 1 / window

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

    def balance_of_power(self, window=14):
        bop = (self.df['close'] - self.df['open']) / (self.df['high'] - self.df['low'])
        return bop.rolling(window=window).mean()

    def bollinger_band(self, window=20):
        self.df['bollinger_middle'] = self.df['close'].rolling(window=window).mean()
        std = self.df['close'].rolling(window=window).std()
        self.df['bollinger_upper'] = self.df['bollinger_middle'] + 2.0 * std
        self.df['bollinger_lower'] = self.df['bollinger_middle'] - 2.0 * std
        return self.df['bollinger_upper'], self.df['bollinger_middle'], self.df['bollinger_lower']

    def bollinger_bandwidth(self, window=20):
        if 'bollinger_middle' not in self.df.columns:
            self.bollinger_band(window=window)
        return (self.df['bollinger_upper'] - self.df['bollinger_lower']) / self.df['bollinger_middle'] * 100

    def bollinger_percentage(self, window=20):
        if 'bollinger_middle' not in self.df.columns:
            self.bollinger_band(window=window)
        bandwidth = self.df['bollinger_upper'] - self.df['bollinger_lower']
        return (self.df['close'] - self.df['bollinger_lower']) / bandwidth

    def chainkin_money_flow(self, window=20):
        if 'money_flow_volume' not in self.df.columns:
            self.accumulation_distribution_line()
        volume_sum = self.df['volume'].rolling(window=window).sum() / self.DF_VOLUME_DIVISOR
        return self.df['money_flow_volume'].rolling(window=window).sum() - volume_sum

    def chainkin_oscillator(self, windows=(20, 3, 10)):
        if 'money_flow_volume' not in self.df.columns:
            self.accumulation_distribution_line()
        adl = self.df['money_flow_volume'].rolling(window=windows[0]).sum()
        return adl.ewm(span=windows[1], adjust=False).mean() - adl.ewm(span=windows[2], adjust=False).mean()

    def force_index(self, window=14):
        fi = (self.df['close'] - self.df['close'].shift(1)) * (self.df['volume'] / self.DF_VOLUME_DIVISOR)
        return fi.ewm(span=window, adjust=False).mean()

    def macd(self, windows=(12, 26, 9)):
        macd_line = (
            self.df['close'].ewm(span=windows[0], adjust=False).mean() -
            self.df['close'].ewm(span=windows[1],adjust=False).mean()
        )

        signal_line = macd_line.ewm(span=windows[2], adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram

    def stochastic_oscillator(self, window=14):
        lowest_low = self.df['low'].rolling(window=window).min()
        highest_high = self.df['high'].rolling(window=window).max()

        k_percent = (self.df['close'] - lowest_low) / (highest_high - lowest_low) * 100
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent


if __name__ == "__main__":
    matplotlib.use('Qt5Agg')
    db = DataBase(db_path='../data/project1.db')
    df = db.get_data(ticker='^GSPC', period='2y', interval='1d')
    print(df.shape)

    feature_extractor = FeatureExtractor(df)
    t1 = feature_extractor.chainkin_oscillator()

    ax = df['close'].plot(color='g')
    t1.plot(secondary_y=True, ax=ax, color='b')

    plt.show()