"""
🌙 Moon Dev's Base Strategy Class
All custom strategies should inherit from this
"""

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        
    def generate_signals(self, symbol, data):
        """Generate trading signals based on strategy rules
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USD')
            data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Signal dictionary with keys:
                - strength (float): Signal strength 0-1
                - direction (str): 'BUY', 'SELL', or 'NOTHING'
                - metadata (dict): Additional signal information
        """
        raise NotImplementedError("Each strategy must implement generate_signals method") 