import tkinter as tk
from tkinter import messagebox
import re
import json
import os

PREDICTIONS_FILE = "predictions.txt"
PORTFOLIO_FILE = "portfolio.json"
CONFIDENCE_THRESHOLD = 0.65
SIMULATED_RETURN = 0.01

class PaperTradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paper Trading Simulator")
        self.cash = 0.0
        self.portfolio = {}
        self.total_capital = 0.0  # sum of cash + portfolio invested

        self.setup_ui()
        self.load_portfolio()

    def setup_ui(self):
        tk.Label(self.root, text="Starting Capital ($):").grid(row=0, column=0)
        self.capital_entry = tk.Entry(self.root)
        self.capital_entry.grid(row=0, column=1)

        self.start_button = tk.Button(self.root, text="Start", command=self.start_simulation)
        self.start_button.grid(row=0, column=2)

        self.simulate_button = tk.Button(self.root, text="Simulate Next Day", command=self.simulate_day, state=tk.DISABLED)
        self.simulate_button.grid(row=1, column=0, columnspan=3)

        self.save_button = tk.Button(self.root, text="💾 Save Portfolio", command=self.save_portfolio)
        self.save_button.grid(row=1, column=2)

        self.output_text = tk.Text(self.root, height=20, width=80)
        self.output_text.grid(row=2, column=0, columnspan=3)

    def start_simulation(self):
        try:
            self.total_capital = float(self.capital_entry.get())
            self.cash = self.total_capital
            self.portfolio = {}
            self.output_text.insert(tk.END, f"Simulation started with ${self.cash:.2f} cash available\n")
            self.simulate_button.config(state=tk.NORMAL)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")

    def simulate_day(self):
        latest_predictions = self.get_latest_predictions()
        if not latest_predictions:
            messagebox.showerror("No Predictions", "Could not find predictions for a new day.")
            return

        self.output_text.insert(tk.END, "\n--- Portfolio Review ---\n")

        stocks_to_sell = []
        for stock in list(self.portfolio):
            prediction = latest_predictions.get(stock)
            if prediction and prediction[0] == "Down":
                self.output_text.insert(tk.END, f"\u26a0\ufe0f Sell {stock}: predicted to go Down\n")
                stocks_to_sell.append(stock)

        for stock in stocks_to_sell:
            sold_amount = self.portfolio.pop(stock)
            self.cash += sold_amount

        held_confidences = {}
        for stock in self.portfolio:
            pred = latest_predictions.get(stock)
            held_confidences[stock] = pred[1] if pred and pred[0] == "Up" else 0.0

        candidates = {
            stock: conf for stock, (mov, conf) in latest_predictions.items()
            if mov == "Up" and conf >= CONFIDENCE_THRESHOLD and stock not in self.portfolio
        }

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        sorted_held = sorted(held_confidences.items(), key=lambda x: x[1])

        replaced_any = False
        for c_stock, c_conf in sorted_candidates:
            if sorted_held:
                weakest_stock, weakest_conf = sorted_held[0]
                if c_conf > weakest_conf:
                    sold_amount = self.portfolio.pop(weakest_stock)
                    self.cash += sold_amount
                    self.output_text.insert(tk.END,
                        f"\u26a0\ufe0f Sold {weakest_stock} (Conf: {weakest_conf:.4f}) to buy {c_stock} (Conf: {c_conf:.4f})\n")

                    self.portfolio[c_stock] = self.cash
                    self.output_text.insert(tk.END,
                        f"✅ Bought {c_stock}: Invested ${self.cash:.2f}\n")
                    self.cash = 0.0
                    replaced_any = True

                    sorted_held.pop(0)
                    sorted_held.append((c_stock, c_conf))
                    sorted_held.sort(key=lambda x: x[1])
                else:
                    break
            else:
                break

        if self.cash > 0 and candidates:
            self.output_text.insert(tk.END, f"\n--- New Investments with remaining cash ---\n")
            new_candidates = {stock: conf for stock, conf in candidates.items() if stock not in self.portfolio}
            if new_candidates:
                total_conf = sum(new_candidates.values())
                day_gain = 0
                total_investment = 0
                for stock, conf in new_candidates.items():
                    weight = conf / total_conf
                    investment = self.cash * weight
                    self.portfolio[stock] = investment
                    simulated_result = 1 + SIMULATED_RETURN
                    gain = investment * (simulated_result - 1)
                    day_gain += gain
                    total_investment += investment
                    self.output_text.insert(tk.END,
                        f"✅ Bought {stock}: Invested ${investment:.2f} (Conf: {conf:.4f})\n"
                    )
                self.cash -= total_investment

        gain_sum = 0
        for stock, amount in self.portfolio.items():
            simulated_result = 1 + SIMULATED_RETURN
            gain = amount * (simulated_result - 1)
            self.portfolio[stock] += gain
            gain_sum += gain
        if gain_sum > 0:
            self.output_text.insert(tk.END, f"\n📈 Portfolio gains today: ${gain_sum:.2f}\n")

        total_portfolio_value = sum(self.portfolio.values())
        self.total_capital = self.cash + total_portfolio_value

        self.output_text.insert(tk.END, f"\n--- Current Portfolio ---\n")
        for stock, amount in self.portfolio.items():
            self.output_text.insert(tk.END, f"{stock}: ${amount:.2f}\n")

        self.output_text.insert(tk.END, f"💰 Cash available: ${self.cash:.2f}\n")
        self.output_text.insert(tk.END, f"💼 Portfolio value: ${total_portfolio_value:.2f}\n")
        self.output_text.insert(tk.END, f"💰 Total account value: ${self.total_capital:.2f}\n")

    def get_latest_predictions(self):
        with open(PREDICTIONS_FILE, "r") as f:
            content = f.read()

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

    def save_portfolio(self):
        data = {
            "cash": self.cash,
            "portfolio": self.portfolio,
            "total_capital": self.total_capital
        }
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(data, f, indent=2)
        self.output_text.insert(tk.END, "\n✅ Portfolio saved successfully.\n")

    def load_portfolio(self):
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                self.cash = data.get("cash", 0.0)
                self.portfolio = data.get("portfolio", {})
                self.total_capital = data.get("total_capital", self.cash + sum(self.portfolio.values()))
                self.capital_entry.insert(0, str(self.total_capital))
                self.output_text.insert(tk.END, f"📂 Loaded existing portfolio. Cash available: ${self.cash:.2f}\n")
                self.simulate_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = PaperTradingApp(root)
    root.mainloop()
