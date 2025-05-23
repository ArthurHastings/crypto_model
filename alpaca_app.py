"""
Romania timeframe when market is open:
4:30 PM to 11:00 PM
5:30 PM to 12:00 AM 
"""

import tkinter as tk
from tkinter import messagebox
import re
import json
import os
import math
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("APCA_API_KEY")
debug = os.getenv("APCA_API_SECRET_KEY")


PREDICTIONS_FILE = "predictions.txt"
CONFIDENCE_THRESHOLD = 0.65

APCA_API_KEY_ID = api_key
APCA_API_SECRET_KEY = debug
BASE_URL = "https://paper-api.alpaca.markets"

class AlpacaTradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alpaca Trading Simulator")

        self.api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, BASE_URL, api_version='v2')
        self.account = self.api.get_account()

        self.setup_ui()
        self.load_portfolio()
        self.load_pending_orders()

    def setup_ui(self):
        tk.Label(self.root, text="Available Cash ($):").grid(row=0, column=0)
        self.cash_label = tk.Label(self.root, text=f"${self.account.cash}")
        self.cash_label.grid(row=0, column=1)

        self.simulate_button = tk.Button(self.root, text="Simulate Trading Day", command=self.simulate_day)
        self.simulate_button.grid(row=1, column=0, columnspan=2)

        self.output_text = tk.Text(self.root, height=20, width=80)
        self.output_text.grid(row=2, column=0, columnspan=2)

    def simulate_day(self):
        """Fetches the latest predictions and sends Alpaca orders accordingly."""
        try:
            open_orders = self.api.list_orders(status='open')
            pending_buy_symbols = {order.symbol for order in open_orders if order.side == 'buy'}
        except Exception as e:
            self.output_text.insert(tk.END, f"Error fetching open orders: {e}\n")
            pending_buy_symbols = set()
        # Uncomment this if u want the app to work only when the market is open not in pre-market
        try:
            clock = self.api.get_clock()
            if not clock.is_open:
                self.output_text.insert(tk.END, "Market is closed or in pre-/after-hours. Skipping trading.\n")
                return
        except Exception as e:
            self.output_text.insert(tk.END, f"Error checking market clock: {e}\n")
            return
        # --------------------------------------------------------
        latest_predictions = self.get_latest_predictions()
        if not latest_predictions:
            messagebox.showerror("No Predictions", "Could not find predictions for a new day.")
            return

        self.output_text.insert(tk.END, "\n--- Starting Trading Day ---\n")

        try:
            positions = {position.symbol: position for position in self.api.list_positions()}
        except Exception as e:
            self.output_text.insert(tk.END, f"Error fetching positions: {e}\n")
            return

        for symbol, position in positions.items():
            pred = latest_predictions.get(symbol)
            if pred and pred[0] == "Down":
                self.output_text.insert(tk.END, f"Selling all shares of {symbol} (predicted Down)\n")
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                except Exception as e:
                    self.output_text.insert(tk.END, f"Error selling {symbol}: {e}\n")

        account = self.api.get_account()
        available_cash = float(account.cash)
        self.output_text.insert(tk.END, f"\nAvailable cash after sales: ${available_cash:.2f}\n")

        buy_candidates = {
            symbol: conf for symbol, (mov, conf) in latest_predictions.items()
            if mov == "Up" and conf >= CONFIDENCE_THRESHOLD and symbol not in positions
        }

        if not buy_candidates:
            self.output_text.insert(tk.END, "No stocks meet the buy criteria.\n")
            self.cash_label.config(text=f"${available_cash:.2f}")
            self.output_text.insert(tk.END, "\n--- Trading Day Completed ---\n")
            return

        sorted_candidates = sorted(buy_candidates.items(), key=lambda x: x[1], reverse=True)

        for symbol, conf in sorted_candidates:

            if symbol in pending_buy_symbols:
                self.output_text.insert(tk.END, f"Skipping {symbol} buy, order already pending.\n")
                continue
            allocation_per_stock = available_cash * 0.20

            try:
                bars = self.api.get_bars(symbol, '1Min', limit=1)
                bars_list = list(bars)
                if bars_list:
                    last_bar = bars_list[-1]
                    last_price = last_bar.c
                else:
                    self.output_text.insert(tk.END, f"No bar data available for {symbol}\n")
                    continue
            except Exception as e:
                self.output_text.insert(tk.END, f"Error fetching price for {symbol}: {e}\n")
                continue

            shares = math.floor(allocation_per_stock / last_price)
            if shares < 1:
                self.output_text.insert(tk.END, f"Not enough cash to buy shares of {symbol}\n")
                continue

            self.output_text.insert(tk.END,
                f"Buying {shares} shares of {symbol} at ${last_price:.2f} (Allocation: ${allocation_per_stock:.2f})\n"
            )
            try:
                self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                # Update available cash after purchase
                available_cash -= shares * last_price
            except Exception as e:
                self.output_text.insert(tk.END, f"Error buying {symbol}: {e}\n")

        self.cash_label.config(text=f"${available_cash:.2f}")
        self.output_text.insert(tk.END, "\n--- Trading Day Completed ---\n")


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

    def load_portfolio(self):
        """Loads and displays current positions from your Alpaca paper account."""
        try:
            positions = self.api.list_positions()
            if positions:
                self.output_text.insert(tk.END, "Current Positions:\n")
                for pos in positions:
                    self.output_text.insert(
                        tk.END, f"{pos.symbol}: {pos.qty} shares at an average entry price of ${pos.avg_entry_price}\n"
                    )
            else:
                self.output_text.insert(tk.END, "No positions currently held.\n")
        except Exception as e:
            self.output_text.insert(tk.END, f"Error loading portfolio: {e}\n")

    def load_pending_orders(self):
        """Loads and displays all open/pending orders."""
        try:
            orders = self.api.list_orders(status='open')
            if orders:
                self.output_text.insert(tk.END, "\nPending Orders:\n")
                for order in orders:
                    self.output_text.insert(
                        tk.END,
                        f"{order.symbol}: {order.qty} shares, side: {order.side}, status: {order.status}\n"
                    )
            else:
                self.output_text.insert(tk.END, "\nNo pending orders.\n")
        except Exception as e:
            self.output_text.insert(tk.END, f"Error loading pending orders: {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlpacaTradingApp(root)
    root.mainloop()
