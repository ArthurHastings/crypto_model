import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv("filled_headline_sentiments.csv")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df = df.dropna(subset=['Date'])

end_date = datetime.now().date()

date_list = [end_date - timedelta(days=i) for i in range(360)]

for date in date_list:
    count = df[df['Date'].dt.date == date].shape[0]
    print(f"Date: {date}, Number of Articles: {count}")
