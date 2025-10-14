import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Parameters and fees
# ---------------------------------------------------------
initial_capital = 10_000
cad_fee = 30            # flat CAD fee on CAD side
usd_fee_rate = 0.0003    # fee rate on USD side (converted to CAD)

# ------------------------------
# 1) Company / Exchange info
# ------------------------------
domestic_ticker = "SHOP.TO"   # TSX 🇨🇦
foreign_ticker  = "SHOP"      # NYSE 🇺🇸

domestic_info = yf.Ticker(domestic_ticker).info
foreign_info  = yf.Ticker(foreign_ticker).info

name = domestic_info.get("longName", "N/A")
industry = domestic_info.get("industry", "N/A")
sector = domestic_info.get("sector", "N/A")
location = f"{domestic_info.get('city', '')}, {domestic_info.get('country', '')}"
domestic_exchange = domestic_info.get("exchange", "N/A")
foreign_exchange = foreign_info.get("exchange", "N/A")

# Time frame: 1 year ago → yesterday
end_date = datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=365)

company_df = pd.DataFrame([{
    "name": name,
    "domestic_ticker": domestic_ticker,
    "industry": industry,
    "sector": sector,
    "location": location,
    "domestic_exchange": domestic_exchange,
    "foreign_exchange": foreign_exchange,
    "foreign_ticker": foreign_ticker
}])

print("Company / Exchange Info:")
print(company_df, "\n")

# --------------------------------
# 2) FX history getter (CADUSD=X)
# --------------------------------
def get_rate(ex_ticker, start_date, end_date):
    ex_rate = yf.Ticker(ex_ticker)
    ex_hist = ex_rate.history(start=start_date, end=end_date)
    ex_df = pd.DataFrame({"Exchange Rate": ex_hist["Close"]})
    ex_df.index = pd.to_datetime(ex_df.index.date)  # clean date index
    return ex_df

exchange_df = get_rate('CADUSD=X', start_date, end_date)  # USD per 1 CAD

# -----------------------------------------
# 3) Download stock prices and merge with FX
# -----------------------------------------
prices = yf.download([domestic_ticker, foreign_ticker], start=start_date, end=end_date, auto_adjust=True)["Close"]
prices.index = pd.to_datetime(prices.index.date)

prices = prices.rename(columns={
    domestic_ticker: "Close_domestic",   # CAD
    foreign_ticker:  "Close_foreign"     # USD
})

# Merging FX rates with stock prices
merged = prices.join(exchange_df, how="inner")
merged["Exchange Rate"] = merged["Exchange Rate"].ffill()

# Implied CAD price of US side
merged["CAD_implied_close_price"] = merged["Close_foreign"] / merged["Exchange Rate"]

merged_portfolio = merged[[
    "Close_domestic",
    "CAD_implied_close_price",
    "Close_foreign",
    "Exchange Rate"
]].dropna().copy() # drop rows with any NaNs

print("Merged portfolio:")
print(merged_portfolio.head(), "\n")

# ---------------------------------------------------------
# 4) Compounding arbitrage strategy 
# ---------------------------------------------------------

min_rel_spread = 0.001   # 10 bps min relative spread to avoid noise

def buy_sell_compounding(df, initial_capital, cad_fee, usd_fee_rate, min_rel_spread=0.0):
    capital = float(initial_capital)
    dates, capitals, trades = [], [], []

    df = df.dropna(subset=["Close_domestic", "CAD_implied_close_price", "Close_foreign", "Exchange Rate"])

    for d in df.index:
        actual = df.loc[d, 'Close_domestic']
        implied = df.loc[d, 'CAD_implied_close_price']
        implied_in_usd = df.loc[d, 'Close_foreign']
        cadusd = df.loc[d, 'Exchange Rate']
        executed = 0

        if cadusd > 0:
            cad_per_usd = 1.0 / cadusd
            spread = abs(actual - implied)
            lower = min(actual, implied)

            if lower > 0 and (spread / lower) >= min_rel_spread:
                num_shares = int(np.floor(capital / lower))
                if num_shares > 0:
                    usd_notional = num_shares * implied_in_usd
                    usd_fee_cad = usd_notional * usd_fee_rate * cad_per_usd
                    fees_cad = cad_fee + usd_fee_cad

                    gross_edge_cad = spread * num_shares
                    if gross_edge_cad > fees_cad:
                        capital += gross_edge_cad - fees_cad
                        executed = 1

        dates.append(d)
        capitals.append(capital)
        trades.append(executed)

    equity_df = pd.DataFrame( {"Equity": capitals, "TradesExecuted": trades}, index=pd.to_datetime(dates))
    equity_df["Profit"] = equity_df["Equity"] - initial_capital
    return equity_df

equity_df = buy_sell_compounding(
    merged_portfolio, initial_capital, cad_fee, usd_fee_rate, min_rel_spread=min_rel_spread
)

# --- Summary ---
print(f"Backtest window: {equity_df.index.min().date()} → {equity_df.index.max().date()}")
print(f"Final Equity (CAD): ${equity_df['Equity'].iloc[-1]:,.2f}")
print(f"Total Profit (CAD): ${equity_df['Profit'].iloc[-1]:,.2f}")
print(f"Trades executed: {int(equity_df['TradesExecuted'].sum())}")

# ---------------------------------------------------------
# 5) Profit over time plot
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(equity_df.index, equity_df["Profit"], color="green", linewidth=2)
plt.title("Arbitrage Strategy Profit Over Time (CAD)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Profit (CAD)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()