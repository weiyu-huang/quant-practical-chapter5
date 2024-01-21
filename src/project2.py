import yfinance as yf
import pandas as pd


class StockSelector:
    def __init__(self, universe="SP500", period="1mo", custom_filter=None):
        self.universe = self.get_universe(universe)
        self.period = period
        self.custom_filter = custom_filter
        self.df = pd.DataFrame(columns=['Symbol', 'AverageVolume', 'AverageClose'])

    def get_universe(self, universe):
        if isinstance(universe, list):
            return universe
        elif universe == "SP500":
            return self.get_sp500_list()
        else:
            raise NotImplementedError(f"Not implemented for string input {universe}")

    @staticmethod
    def get_sp500_list():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)
        sp500_symbols = table[0]['Symbol'].tolist()
        return sp500_symbols

    def collect_stock_data(self):
        for symbol in self.universe:
            stock = yf.Ticker(symbol)
            stock_info = stock.history(period=self.period)
            avg_volume = stock_info['Volume'].mean()
            avg_close = stock_info['Close'].mean()

            self.df.loc[len(self.df)] = {
                'Symbol': symbol,
                'AverageVolume': avg_volume,
                'AverageClose': avg_close
            }

    def select_stocks(self):
        self.collect_stock_data()
        if self.custom_filter is not None:
            self.df = self.df[self.custom_filter(self.df)]
            self.df.reset_index(drop=True, inplace=True)
        return self.df


def main():
    def my_filter(df):
        price_condition = (
            (df['AverageClose'] >= df['AverageClose'].quantile(0.33)) &
            (df['AverageClose'] <= df['AverageClose'].quantile(0.66))
        )
        volume_condition = (
            (df['AverageVolume'] >= df['AverageVolume'].quantile(0.33)) &
            (df['AverageVolume'] <= df['AverageVolume'].quantile(0.66))
        )
        return price_condition & volume_condition

    selector = StockSelector(custom_filter=my_filter)
    df_stocks = selector.select_stocks()
    df_stocks.to_csv("../data/selected_stocks.csv", index=False)


if __name__ == "__main__":
    main()
