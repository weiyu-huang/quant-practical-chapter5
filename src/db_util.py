import sqlite3
import yfinance as yf
import pandas as pd

CREATE_TABLE_SQL = '''
    CREATE TABLE {}(
        ticker TEXT,
        date DATE,
        open DECIMAL(18, 7),
        high DECIMAL(18, 7),
        low DECIMAL(18, 7),
        close DECIMAL(18, 7),
        adj_close DECIMAL(18, 7),
        volume INTEGER
    )
'''


class DataBase:
    def __init__(self, db_path):
        self.db_path = db_path

    def create_db(self, conn_, table_name_):
        cursor = conn_.cursor()
        cursor.execute(CREATE_TABLE_SQL.format(table_name_))
        conn_.commit()

    def get_data(self, ticker, interval='1d', period='1y'):
        table_name = ticker if ticker[0] != '^' else ticker[1:]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if cursor.fetchone()[0] == 0:
                self.create_db(conn_=conn, table_name_=table_name)

                data = yf.download(ticker, group_by='Ticker', interval=interval, period=period)
                data['ticker'] = ticker

                data.reset_index(inplace=True)
                data.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']
                data = data[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

                data.to_sql(table_name, conn, if_exists='replace', index=False)

            query = f"SELECT * FROM {table_name}"
            df_ = pd.read_sql_query(query, conn)
            return df_


if __name__ == '__main__':
    db = DataBase(db_path='../data/sample.db')
    df = db.get_data(ticker='^GSPC', period='1mo')
    print(df.shape)

    df2 = db.get_data(ticker='AAPL', period='1mo')
    print(df2.shape)
