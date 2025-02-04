Below is your complete, fixed code for Moon Dev’s HierarchicalBreakout strategy. All indicators are now calculated only using TA‑Lib (wrapped via self.I) with NO backtesting.lib usage – and plenty of Moon Dev themed debug prints to boot! Enjoy! 🌙✨

#!/usr/bin/env python3
"""
Moon Dev's Backtest AI 🌙
Backtesting implementation for the HierarchicalBreakout strategy.
This strategy uses a multi‐timeframe analysis (simulated on our 15‑minute bars)
to decide on trades based solely on price action and market structure.
It uses TA‑Lib for indicator calculation (NEVER using backtesting.lib built‐ins!)
"""

import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

# ─────────────── Strategy Class ─────────────────────────────────────────────
class HierarchicalBreakout(Strategy):
    # PARAMETERS (and optimization candidates!)
    risk_pct      = 0.01        # Risk per trade as a fraction of account equity.
    risk_reward   = 2.0         # Risk/Reward ratio. (Try optimizing e.g. range(2,5).)
    weekly_period = 672         # 15m bars for one week = 7 * 24*4 = 672 bars.
    daily_period  = 96          # 15m bars for 1 day   = 24*4 = 96 bars.
    cons_period   = 16          # Consolidation period (4‑hr approx.; 16 x 15m bars).
    swing_period  = 4           # Swing lookback for stop loss (1 hr ~ 4 x 15mins).

    def init(self):
        print("🌙 [INIT] Initializing HierarchicalBreakout strategy indicators...")

        # Multi-timeframe trend confirmation indicators:
        self.weekly_sma = self.I(talib.SMA, self.data.Close, timeperiod=self.weekly_period)
        self.daily_sma  = self.I(talib.SMA, self.data.Close, timeperiod=self.daily_period)

        # 4‑hour consolidation zone based on recent highs/lows:
        self.cons_high = self.I(talib.MAX, self.data.High, timeperiod=self.cons_period)
        self.cons_low  = self.I(talib.MIN, self.data.Low, timeperiod=self.cons_period)

        # For placing stop loss levels based on recent swing lows/highs on lower timeframe:
        self.swing_low  = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)

        print("✨ [INIT] weekly_sma, daily_sma, cons_high/low, and swing_high/low calculated. Ready for action! 🚀")

    def next(self):
        current_price = self.data.Close[-1]
        print(f"🌙 [NEXT] Bar Date: {self.data.index[-1]} | Close: {current_price:.2f}")

        # Determine trend from our “higher” timeframes.
        weekly_trend_bull = current_price > self.weekly_sma[-1]
        daily_trend_bull  = current_price > self.daily_sma[-1]
        weekly_trend_bear = current_price < self.weekly_sma[-1]
        daily_trend_bear  = current_price < self.daily_sma[-1]

        # If there is no open position, evaluate entry signals.
        if not self.position:
            # LONG ENTRY: Bullish overall bias.
            if weekly_trend_bull and daily_trend_bull:
                # Lower timeframe breakout: price must break above recent 4‑hr consolidation high.
                if current_price > self.cons_high[-1]:
                    stop_loss = self.swing_low[-1]
                    if current_price - stop_loss <= 0:
                        print("🚀 [LONG] Aborting long entry: calculated stop loss >= entry price!")
                        return

                    risk_amount = self.equity * self.risk_pct
                    risk_per_unit = current_price - stop_loss
                    pos_size = risk_amount / risk_per_unit
                    take_profit = current_price + self.risk_reward * risk_per_unit

                    print(f"🌙 [LONG ENTRY SIGNAL] Price {current_price:.2f} > cons_high {self.cons_high[-1]:.2f}!")
                    print(f"✨ [LONG ENTRY] Risk: ${risk_amount:.2f} at risk/unit: {risk_per_unit:.2f} => Size: {pos_size:.4f} shares.")
                    print(f"🚀 [LONG ORDER] Entry: {current_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

                    self.buy(size=pos_size, sl=stop_loss, tp=take_profit)

            # SHORT ENTRY: Bearish overall bias.
            elif weekly_trend_bear and daily_trend_bear:
                if current_price < self.cons_low[-1]:
                    stop_loss = self.swing_high[-1]
                    if stop_loss - current_price <= 0:
                        print("🚀 [SHORT] Aborting short entry: calculated stop loss <= entry price!")
                        return

                    risk_amount = self.equity * self.risk_pct
                    risk_per_unit = stop_loss - current_price
                    pos_size = risk_amount / risk_per_unit
                    take_profit = current_price - self.risk_reward * risk_per_unit

                    print(f"🌙 [SHORT ENTRY SIGNAL] Price {current_price:.2f} < cons_low {self.cons_low[-1]:.2f}!")
                    print(f"✨ [SHORT ENTRY] Risk: ${risk_amount:.2f} at risk/unit: {risk_per_unit:.2f} => Size: {pos_size:.4f} shares.")
                    print(f"🚀 [SHORT ORDER] Entry: {current_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

                    self.sell(size=pos_size, sl=stop_loss, tp=take_profit)
        
        # If already in a trade, check for break‐even cues.
        else:
            if self.position.is_long:
                entry_price = self.position.entry_price
                target_profit = (self.position.tp - entry_price) if hasattr(self.position, "tp") else 0
                if target_profit and current_price > entry_price + 0.5 * target_profit:
                    print("🚀 [LONG POSITION] Break-even threshold reached! Consider moving SL to entry price to protect gains! 🌙")
            elif self.position.is_short:
                entry_price = self.position.entry_price
                target_profit = (entry_price - self.position.tp) if hasattr(self.position, "tp") else 0
                if target_profit and current_price < entry_price - 0.5 * target_profit:
                    print("🚀 [SHORT POSITION] Break-even threshold reached! Consider moving SL to entry price to protect gains! 🌙")


# ─────────────── MAIN EXECUTION ─────────────────────────────────────────────
if __name__ == '__main__':
    # DATA HANDLING:
    data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
    print("🌙 [DATA] Loading data from:", data_path)
    data = pd.read_csv(data_path, parse_dates=['datetime'])

    # Clean up column names and remove any unnamed columns.
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    print("✨ [DATA] Column names cleaned:", list(data.columns))

    # Map columns as required by backtesting.py.
    rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    data.rename(columns=rename_map, inplace=True)
    if 'datetime' in data.columns:
        data.set_index('datetime', inplace=True)
    data.sort_index(inplace=True)
    print("🚀 [DATA] Final columns mapped as:", data.columns.tolist())

    # Create the Backtest instance with our strategy.
    bt = Backtest(data, HierarchicalBreakout, cash=1_000_000, commission=0.0)
    print("🌙 [BACKTEST] Starting initial backtest with default parameters for HierarchicalBreakout...")

    stats = bt.run()
    print("\n🌙 [STATS] Initial backtest complete! Full stats below:")
    print(stats)
    print("✨ [STRATEGY DETAILS]:", stats._strategy)
    
    strategy_name = "HierarchicalBreakout"
    chart_dir = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/charts"
    chart_file = os.path.join(chart_dir, f"{strategy_name}_chart.html")
    print(f"🚀 [PLOT] Saving initial performance plot to: {chart_file}")
    bt.plot(filename=chart_file, open_browser=False)

    print("\n🌙 [OPTI] Starting optimization! Optimizing risk_reward and risk_pct parameters...")
    optimized_stats = bt.optimize(
        risk_reward = range(2, 5),       # Try risk/reward of 2, 3, 4
        risk_pct    = [0.005, 0.01, 0.015],
        maximize    = 'Equity Final Value',
        constraint  = None  # Add custom constraints if necessary.
    )
    print("\n✨ [OPTI] Optimization complete! Best settings and stats:")
    print(optimized_stats)
    print("🚀 [OPTI STRATEGY DETAILS]:", optimized_stats._strategy)

    opt_chart_file = os.path.join(chart_dir, f"{strategy_name}_optimized_chart.html")
    print(f"🌙 [PLOT] Saving optimized performance plot to: {opt_chart_file}")
    bt.plot(filename=opt_chart_file, open_browser=False)

    print("🚀 [DONE] HierarchicalBreakout backtest and optimization completed successfully! 🌙✨")

Happy backtesting – may the Moon Dev spirit guide your trades!