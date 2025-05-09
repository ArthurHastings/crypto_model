from tvDatafeed import TvDatafeed, Interval
import pandas as pd

tv = TvDatafeed(username='SOLOMON_ROCKS', password='zazacox1234567!')

df = tv.get_hist(symbol='AAPL', exchange='NASDAQ', interval=Interval.in_daily, n_bars=5)

df = df.reset_index()

df = df.rename(columns={
    'datetime': 'Date',
    'close': 'Close',
    'high': 'High',
    'low': 'Low',
    'open': 'Open',
    'volume': 'Volume'
})

df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

print(df)
