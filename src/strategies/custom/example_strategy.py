from src.strategies.base_strategy import BaseStrategy
import pandas_ta as ta

class ExampleStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Moon Dev Example Strategy 🌙"
        
    def generate_signals(self, symbol, data):
        """Generate trading signals based on EMA crossover
        
        Args:
            symbol (str): Trading pair symbol
            data (pd.DataFrame): Historical price data
        """
        try:
            # Calculate EMAs
            data['ema_10'] = ta.ema(data['close'], length=10)
            data['ema_20'] = ta.ema(data['close'], length=20)
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            ema_10 = data['ema_10'].iloc[-1]
            ema_20 = data['ema_20'].iloc[-1]
            
            # Generate signal
            if ema_10 > ema_20:
                return {
                    'strength': 0.8,
                    'direction': 'BUY',
                    'metadata': {
                        'price': current_price,
                        'ema_10': ema_10,
                        'ema_20': ema_20
                    }
                }
            elif ema_10 < ema_20:
                return {
                    'strength': 0.8,
                    'direction': 'SELL',
                    'metadata': {
                        'price': current_price,
                        'ema_10': ema_10,
                        'ema_20': ema_20
                    }
                }
            
            return {
                'strength': 0,
                'direction': 'NOTHING',
                'metadata': {}
            }
            
        except Exception as e:
            print(f"❌ Error in example strategy: {str(e)}")
            return None 