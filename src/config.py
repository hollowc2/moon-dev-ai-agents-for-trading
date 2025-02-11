"""
🌙 Billy Bitcoin's Configuration File
Built with love by Billy Bitcoin 🚀
"""

from datetime import datetime
import nice_funcs_cb as cb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading Agent Configuration
ANTHROPIC_KEY: str = os.getenv('ANTHROPIC_KEY')
AI_MODEL: str = os.getenv('AI_MODEL', "claude-3-sonnet-20240229")
AI_MAX_TOKENS: int = int(os.getenv('AI_MAX_TOKENS', '4096'))
AI_TEMPERATURE: float = float(os.getenv('AI_TEMPERATURE', '0.7'))
SLEEP_BETWEEN_RUNS_MINUTES: int = int(os.getenv('SLEEP_BETWEEN_RUNS_MINUTES', '5'))
MAX_POSITION_PERCENTAGE: float = float(os.getenv('MAX_POSITION_PERCENTAGE', '20.0'))
CASH_PERCENTAGE: float = float(os.getenv('CASH_PERCENTAGE', '20.0'))

# API Keys and Credentials
BIRDEYE_API_KEY: str = os.getenv('BIRDEYE_API_KEY')
RPC_ENDPOINT: str = os.getenv('RPC_ENDPOINT')
MOONDEV_API_KEY: str = os.getenv('MOONDEV_API_KEY')
SOLANA_PRIVATE_KEY: str = os.getenv('SOLANA_PRIVATE_KEY')
OPENAI_KEY: str = os.getenv('OPENAI_KEY')
TWITTER_USERNAME: str = os.getenv('TWITTER_USERNAME')
TWITTER_EMAIL: str = os.getenv('TWITTER_EMAIL')
TWITTER_PASSWORD: str = os.getenv('TWITTER_PASSWORD')
COINBASE_API_KEY: str = os.getenv('COINBASE_API_KEY')
COINBASE_API_SECRET: str = os.getenv('COINBASE_API_SECRET')

# Trading Configuration 💰
SLEEP_BETWEEN_RUNS_MINUTES = int(os.getenv('SLEEP_BETWEEN_RUNS_MINUTES', '5'))
MAX_POSITION_PERCENTAGE = float(os.getenv('MAX_POSITION_PERCENTAGE', '20.0'))
CASH_PERCENTAGE = float(os.getenv('CASH_PERCENTAGE', '20.0'))

# Trading Agent Settings
REQUIRE_STRATEGY_SIGNALS = False  # If True, only trade with strategy confirmation


# 💰 Trading Configuration
USDC_ADDRESS = "USDC-USD"  # Coinbase uses trading pair format
SOL_ADDRESS = "SOL-USD"    # Coinbase uses trading pair format

# Create a list of addresses to exclude from trading/closing
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]

# Token List for Trading 📋
STABLECOIN_IDENTIFIERS = ['USDC', 'USDT', 'DAI', 'BUSD', 'UST', 'TUSD', 'USDP', 'GUSD']
TOP_TOKENS_TO_MONITOR = 10  # Number of top tokens by volume to monitor

# Add caching for trading pairs
_cached_trading_pairs = None
_last_cache_time = None



# Dynamic token list - will be updated periodically
MONITORED_TOKENS = [
    'BTC-USD',    # Bitcoin
    'ETH-USD'    # Ethereum
]


# Token and wallet settings
symbol = 'BTC-USD'  # Default trading pair
address = None      # Not needed for Coinbase

# Position sizing 🎯
usd_size = 10  # Size of position to hold
max_usd_order_size = 3  # Max order size
MIN_TRADE_SIZE_USD = 1.0  # Minimum trade size allowed by Coinbase
tx_sleep = 1  # Reduced sleep between transactions for CEX
slippage = 100  # 1% slippage (100 = 1%)

# Risk Management Settings 🛡️
STOPLOSS_PRICE = 1 # NOT USED YET 1/5/25    
BREAKOUT_PRICE = .0001 # NOT USED YET 1/5/25
SLEEP_AFTER_CLOSE = 600  # Prevent overtrading

MAX_LOSS_GAIN_CHECK_HOURS = 12  # How far back to check for max loss/gain limits (in hours)


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

