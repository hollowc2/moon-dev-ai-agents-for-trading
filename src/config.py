"""
🌙 Billy Bitcoin's Configuration File
Built with love by Billy Bitcoin 🚀
"""

# 💰 Trading Configuration
USDC_ADDRESS = "USDC-USD"  # Coinbase uses trading pair format
SOL_ADDRESS = "SOL-USD"    # Coinbase uses trading pair format

# Create a list of addresses to exclude from trading/closing
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]

# Token List for Trading 📋
MONITORED_TOKENS = [
    'BTC-USD',    # Bitcoin
    'ETH-USD',    # Ethereum
    'SOL-USD',    # Solana
    'MOG-USD',
]

# Billy Bitcoin's Token Trading List 🚀
# Each token is carefully selected by Billy Bitcoin for maximum moon potential! 🌙
tokens_to_trade = MONITORED_TOKENS

# Token and wallet settings
symbol = 'BTC-USD'  # Default trading pair
address = None      # Not needed for Coinbase

# Position sizing 🎯
usd_size = 10  # Size of position to hold
max_usd_order_size = 3  # Max order size
tx_sleep = 1  # Reduced sleep between transactions for CEX
slippage = 100  # 1% slippage (100 = 1%)

# Risk Management Settings 🛡️
CASH_PERCENTAGE = 20  # Minimum % to keep in USDC as safety buffer (0-100)
MAX_POSITION_PERCENTAGE = 30  # Maximum % allocation per position (0-100)
STOPLOSS_PRICE = 1 # NOT USED YET 1/5/25    
BREAKOUT_PRICE = .0001 # NOT USED YET 1/5/25
SLEEP_AFTER_CLOSE = 600  # Prevent overtrading

MAX_LOSS_GAIN_CHECK_HOURS = 12  # How far back to check for max loss/gain limits (in hours)
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒


# Max Loss/Gain Settings FOR RISK AGENT 1/5/25
USE_PERCENTAGE = False  # If True, use percentage-based limits. If False, use USD-based limits

# USD-based limits (used if USE_PERCENTAGE is False)
MAX_LOSS_USD = 25  # Maximum loss in USD before stopping trading
MAX_GAIN_USD = 25 # Maximum gain in USD before stopping trading

# USD MINIMUM BALANCE RISK CONTROL
MINIMUM_BALANCE_USD = 10  # If balance falls below this, risk agent will consider closing all positions
USE_AI_CONFIRMATION = True  # If True, consult AI before closing positions. If False, close immediately on breach

# Percentage-based limits (used if USE_PERCENTAGE is True)
MAX_LOSS_PERCENT = 5  # Maximum loss as percentage (e.g., 20 = 20% loss)
MAX_GAIN_PERCENT = 5  # Maximum gain as percentage (e.g., 50 = 50% gain)

# Transaction settings ⚡
PRIORITY_FEE = None  # Remove as not needed for Coinbase
orders_per_open = 1  # Single order is typically sufficient for CEX

# Market maker settings 📊
buy_under = 0.99  # Buy when price is 1% below target
sell_over = 1.01  # Sell when price is 1% above target

# Data collection settings 📈
DAYSBACK_4_DATA = 3
DATA_TIMEFRAME = '1h'  # Changed to lowercase to match Coinbase format
SAVE_OHLCV_DATA = False  # 🌙 Set to True to save data permanently, False will only use temp data during run

# AI Model Settings 🤖
AI_MODEL = "claude-3-haiku-20240307"  # Model Options:
                                     # - claude-3-haiku-20240307 (Fast, efficient Claude model)
                                     # - claude-3-sonnet-20240229 (Balanced Claude model)
                                     # - claude-3-opus-20240229 (Most powerful Claude model)
AI_MAX_TOKENS = 1024  # Max tokens for response
AI_TEMPERATURE = 0.7  # Creativity vs precision (0-1)

# Trading Strategy Agent Settings - MAY NOT BE USED YET 1/5/25
ENABLE_STRATEGIES = True  # Set this to True to use strategies
STRATEGY_MIN_CONFIDENCE = 0.7  # Minimum confidence to act on strategy signals

# Sleep time between main agent runs
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒

# Future variables (not active yet) 🔮
sell_at_multiple = 3
USDC_SIZE = 1
limit = 49
timeframe = '15m'
stop_loss_perctentage = -.24
EXIT_ALL_POSITIONS = False
DO_NOT_TRADE_LIST = ['777']
CLOSED_POSITIONS_TXT = '777'
minimum_trades_in_last_hour = 777

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,  # Disable existing loggers
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'WARNING',  # Only show warnings and above
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'WARNING',  # Only show warnings and above
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'moondev.log',
            'mode': 'a',
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'WARNING',
            'propagate': False
        },
        'coinbase': {
            'handlers': ['console', 'file'],
            'level': 'WARNING',  # Set Coinbase logging to WARNING only
            'propagate': False
        },
        'urllib3': {
            'level': 'WARNING',
            'propagate': False
        },
        'requests': {
            'level': 'WARNING',
            'propagate': False
        }
    }
}

# Apply logging config
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)

# Trading Agent Settings
REQUIRE_STRATEGY_SIGNALS = False  # If True, only trade with strategy confirmation
