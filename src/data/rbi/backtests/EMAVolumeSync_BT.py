Here’s the implementation of the `EMAVolumeSync` strategy in Python using the `backtesting.py` framework. The code includes all the necessary components, such as indicators, entry/exit logic, risk management, and parameter optimization. It also includes Moon Dev-themed debug prints for easier debugging.

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Clean and prepare the data
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert datetime column to proper format
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class EMAVolumeSync(Strategy):
    # Strategy parameters
    ema_period = 20
    volume_ma_period = 20
    risk_per_trade = 0.01  # Risk 1% of capital per trade
    risk_reward_ratio = 2  # 2:1 risk-reward ratio

    def init(self):
        # Calculate EMAs
        self.green_ema = self.I(talib.EMA, self.data.High, timeperiod=self.ema_period)
        self.red_ema = self.I(talib.EMA, self.data.Low, timeperiod=self.ema_period)

        # Calculate Volume Moving Average
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)

        # Debug prints
        print("🌙 EMAVolumeSync Strategy Initialized! ✨")
        print(f"📊 Green EMA (High): {self.ema_period} periods")
        print(f"📊 Red EMA (Low): {self.ema_period} periods")
        print(f"📊 Volume MA: {self.volume_ma_period} periods")
        print(f"⚠️ Risk per trade: {self.risk_per_trade * 100}%")
        print(f"🎯 Risk-Reward Ratio: {self.risk_reward_ratio}:1")

    def next(self):
        # Skip if indicators are not ready
        if len(self.data) < self.ema_period or len(self.data) < self.volume_ma_period:
            return

        # Current price and volume
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        # Trend direction
        is_uptrend = current_close > self.green_ema[-1] and current_close > self.red_ema[-1]
        is_downtrend = current_close < self.green_ema[-1] and current_close < self.red_ema[-1]

        # Volume confirmation
        volume_confirmation = current_volume > self.volume_ma[-1]

        # Entry logic for long trades
        if is_uptrend and volume_confirmation:
            if not self.position:
                # Calculate position size based on risk
                stop_loss = self.red_ema[-1]  # Use Red EMA as stop loss
                risk_amount = self.risk_per_trade * self.equity
                position_size = risk_amount / (current_close - stop_loss)

                # Enter long trade
                self.buy(size=position_size, sl=stop_loss, tp=current_close + (current_close - stop_loss) * self.risk_reward_ratio)
                print(f"🚀 Long Entry at {current_close} | SL: {stop_loss} | TP: {current_close + (current_close - stop_loss) * self.risk_reward_ratio}")

        # Entry logic for short trades
        elif is_downtrend and volume_confirmation:
            if not self.position:
                # Calculate position size based on risk
                stop_loss = self.green_ema[-1]  # Use Green EMA as stop loss
                risk_amount = self.risk_per_trade * self.equity
                position_size = risk_amount / (stop_loss - current_close)

                # Enter short trade
                self.sell(size=position_size, sl=stop_loss, tp=current_close - (stop_loss - current_close) * self.risk_reward_ratio)
                print(f"📉 Short Entry at {current_close} | SL: {stop_loss} | TP: {current_close - (stop_loss - current_close) * self.risk_reward_ratio}")

        # Exit logic
        if self.position:
            if is_uptrend and crossunder(self.data.Close, self.red_ema):
                self.position.close()
                print(f"🌙 Exit Long at {current_close} | Trend Reversal Detected ✨")
            elif is_downtrend and crossover(self.data.Close, self.green_ema):
                self.position.close()
                print(f"🌙 Exit Short at {current_close} | Trend Reversal Detected ✨")

# Initialize backtest
bt = Backtest(data, EMAVolumeSync, cash=1_000_000, commission=0.002)

# Run initial backtest
stats = bt.run()
print(stats)
print(stats._strategy)

# Show initial performance plot
bt.plot()

# Optimize parameters
optimization_results = bt.optimize(
    ema_period=range(15, 25, 1),
    volume_ma_period=range(15, 25, 1),
    risk_reward_ratio=[2, 3],
    maximize='Return [%]'
)

# Print optimized results
print("🎉 Optimized Results 🎉")
print(optimization_results)

# Show optimized performance plot
bt.plot()
```

### Explanation of the Code:
1. **Data Preparation**:
   - The data is cleaned and prepared by renaming columns and ensuring proper formatting.
   - Unnamed columns are dropped, and the datetime column is set as the index.

2. **Strategy Class**:
   - The `EMAVolumeSync` class defines the strategy logic.
   - Two EMAs (Green and Red) are calculated using the high and low prices, respectively.
   - A volume moving average is used to confirm strong trends.

3. **Entry Logic**:
   - Long trades are entered when the price is above both EMAs and volume is above the volume moving average.
   - Short trades are entered when the price is below both EMAs and volume is above the volume moving average.
   - Position sizes are calculated based on risk percentage and stop-loss levels.

4. **Exit Logic**:
   - Trades are exited when the price crosses the opposite EMA, indicating a potential trend reversal.

5. **Risk Management**:
   - Risk is limited to 1% of the trading capital per trade.
   - A fixed risk-reward ratio (2:1 or 3:1) is used for take-profit levels.

6. **Optimization**:
   - The strategy parameters (EMA period, volume MA period, and risk-reward ratio) are optimized to maximize returns.

7. **Moon Dev-Themed Debug Prints**:
   - Debug prints are added to track strategy initialization, trade entries, and exits.

### Execution Order:
1. Run the initial backtest with default parameters.
2. Print the full stats and strategy details.
3. Show the initial performance plot.
4. Run the optimization process.
5. Print the optimized results and show the final performance plot.

This implementation ensures that the strategy is robust, well-optimized, and easy to debug with Moon Dev-themed messages. 🌙✨🚀