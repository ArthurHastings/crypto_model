import pandas_market_calendars as mcal
from datetime import datetime

nyse = mcal.get_calendar('NYSE')
today = datetime.now().date()

schedule = nyse.schedule(start_date=str(today), end_date=str(today))

if schedule.empty:
    print("Market is closed today. Exiting workflow.")
    exit(1)
else:
    print("Market is open today.")
