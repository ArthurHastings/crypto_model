"""
Romania timeframe when market is open:
4:30 PM to 11:00 PM
5:30 PM to 12:00 AM
"""

import re
import os
import math
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv(override=True)

PREDICTIONS_FILE = "predictions.txt"
CONFIDENCE_THRESHOLD = 0.65

APCA_API_KEY_ID = os.getenv("APCA_API_KEY")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaTradingWorkflow:
    def __init__(self):
        self.api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, BASE_URL, api_version='v2')

    def run(self):
        # Uncomment this if u want the app to work only when the market is open not in pre-market
        try:
            clock = self.api.get_clock()
            if not clock.is_open:
                print("Market is closed or in pre-/after-hours. Skipping trading.")
                return
        except Exception as e:
            print(f"Error checking market clock: {e}")
            return
        # --------------------------------------------------------
        latest_predictions = self.get_latest_predictions()
        if not latest_predictions:
            print("Could not find predictions for a new day.")
            return

        print("\n--- Starting Trading Day ---\n")

        try:
            positions = {position.symbol: position for position in self.api.list_positions()}
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return

        try:
            open_orders = self.api.list_orders(status='open')
            pending_buy_symbols = {order.symbol for order in open_orders if order.side == 'buy'}
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            pending_buy_symbols = set()

        for symbol, position in positions.items():
            pred = latest_predictions.get(symbol)
            if pred and pred[0] == "Down":
                print(f"Selling all shares of {symbol} (predicted Down)")
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                except Exception as e:
                    print(f"Error selling {symbol}: {e}")

        try:
            account = self.api.get_account()
            available_cash = float(account.cash)
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return

        print(f"\nAvailable cash after sales: ${available_cash:.2f}")

        buy_candidates = {
            symbol: conf for symbol, (mov, conf) in latest_predictions.items()
            if mov == "Up" and conf >= CONFIDENCE_THRESHOLD and symbol not in positions
        }

        if not buy_candidates:
            print("No stocks meet the buy criteria.")
            print("\n--- Trading Day Completed ---")
            return

        sorted_candidates = sorted(buy_candidates.items(), key=lambda x: x[1], reverse=True)

        for symbol, conf in sorted_candidates:
            if symbol in pending_buy_symbols:
                print(f"Skipping {symbol} buy, order already pending.")
                continue

            allocation_per_stock = available_cash * 0.20

            try:
                bars = self.api.get_bars(symbol, '1Min', limit=1)
                bars_list = list(bars)
                if bars_list:
                    last_bar = bars_list[-1]
                    last_price = last_bar.c
                else:
                    print(f"No bar data available for {symbol}")
                    continue
            except Exception as e:
                print(f"Error fetching price for {symbol}: {e}")
                continue

            shares = math.floor(allocation_per_stock / last_price)
            if shares < 1:
                print(f"Not enough cash to buy shares of {symbol}")
                continue

            print(f"Buying {shares} shares of {symbol} at ${last_price:.2f} (Allocation: ${allocation_per_stock:.2f})")
            try:
                self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                available_cash -= shares * last_price
            except Exception as e:
                print(f"Error buying {symbol}: {e}")

        print(f"\nAvailable cash at end of trading day: ${available_cash:.2f}")
        print("\n--- Trading Day Completed ---")

    def get_latest_predictions(self):
        """Parses the predictions file to extract the latest predictions.
           Expected line format: 'AAPL: Predicted movement = Up (0.78)'."""
        try:
            with open(PREDICTIONS_FILE, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return None

        blocks = re.findall(r"--- Predictions.*?---([\s\S]*?)(?=--- Predictions|\Z)", content)
        if not blocks:
            return None

        latest_block = blocks[-1]
        lines = latest_block.strip().splitlines()
        predictions = {}
        for line in lines:
            match = re.match(r"(\w+): Predicted movement = (\w+) \(([\d.]+)\)", line.strip())
            if match:
                stock, movement, confidence = match.groups()
                predictions[stock] = (movement, float(confidence))
        return predictions


if __name__ == "__main__":
    workflow = AlpacaTradingWorkflow()
    workflow.run()
