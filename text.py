import yfinance as yf
import pandas as pd

df_price = yf.download("PLTR", period=f"30d", interval="1d").reset_index()

print(df_price)