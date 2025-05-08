from alpha_vantage.timeseries import TimeSeries
import pandas as pd

API_KEY = '47FA2C8JICIFQCWG'

ts = TimeSeries(key=API_KEY, output_format='pandas')

data, _ = ts.get_daily(symbol="PLTR", outputsize='compact')

data = data.rename(columns={
    '1. open': 'Open',
    '2. high': 'High',
    '3. low': 'Low',
    '4. close': 'Close',
    '5. volume': 'Volume'
})

data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date', ascending=False)
data = data[data['date'] >= pd.Timestamp.today() - pd.Timedelta(days=30)]
data = data.sort_values('date')

print(data)
