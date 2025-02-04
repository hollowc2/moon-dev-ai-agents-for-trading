"""
🌙 Moon Dev's Nice Functions - Coinbase Edition
Built with love by Moon Dev 🚀
"""

import logging

# Disable all Coinbase-related logging
logging.getLogger('coinbase').setLevel(logging.WARNING)
logging.getLogger('coinbase.RESTClient').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

from coinbase.rest import RESTClient
from coinbase.websocket import WSClient
import pandas as pd
import pprint
import os
import time
import json
import numpy as np
import datetime
import pandas_ta as ta
from datetime import datetime, timedelta
from termcolor import colored, cprint
from dotenv import load_dotenv
import shutil
import atexit
import math
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from coinbase import jwt_generator
import requests

load_dotenv(override=True)

def init_client():
    """Initialize REST client with proper error handling"""
    try:
        api_key = os.getenv("COINBASE_API_KEY")
        private_key = os.getenv('COINBASE_API_SECRET').replace('\\n', '\n')
        client = RESTClient(
            api_key=api_key,
            api_secret=private_key,
            verbose=True
        )
        
        # Test the connection
        _ = client.get_accounts()
        return client
        
    except Exception as e:
        print(f"Failed to initialize client: {str(e)}")
        raise

# Initialize the client
try:
    rest_client = init_client()
except Exception as e:
    print(f"❌ Failed to initialize Coinbase client: {str(e)}")
    rest_client = None

# Base URLs - Updated according to documentation
BASE_URL = "https://api.coinbase.com/api/v3/brokerage"  # For authenticated endpoints
MARKET_URL = "https://api.coinbase.com/api/v3/market"   # For public endpoints

# Create temp directory and register cleanup
os.makedirs('temp_data', exist_ok=True)

def cleanup_temp_data():
    if os.path.exists('temp_data'):
        print("🧹 Moon Dev cleaning up temporary data...")
        shutil.rmtree('temp_data')

atexit.register(cleanup_temp_data)

def generate_jwt(method, request_path):
    """
    Generate JWT token for Coinbase Advanced Trade API using jwt_generator
    """
    try:
        api_key = os.getenv("COINBASE_API_KEY")
        api_secret = os.getenv("COINBASE_API_SECRET")
        
        jwt_uri = jwt_generator.format_jwt_uri(method, request_path)
        jwt_token = jwt_generator.build_rest_jwt(jwt_uri, api_key, api_secret)
        
        return jwt_token
        
    except Exception as e:
        print(f"Error generating JWT: {str(e)}")
        raise

def get_coinbase_headers(method, request_path, body=''):
    """
    Generate headers for Coinbase Advanced Trade API
    """
    jwt_token = generate_jwt(method, request_path)
    return {
        'Authorization': f'Bearer {jwt_token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

# Custom function to print JSON in a human-readable format
def print_pretty_json(data):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

def get_product_overview(symbol):
    """
    Fetch product overview for a given symbol (e.g. 'BTC-USD')
    """
    try:
        response = rest_client.get_product(product_id=symbol)
        return response.to_dict() if response else None
    except Exception as e:
        print(f"Error getting product overview: {str(e)}")
        return None

def get_product_stats(symbol):
    """
    Get 24h stats for a product
    """
    request_path = f'/products/{symbol}/stats'
    headers = get_coinbase_headers('GET', request_path)
    
    response = requests.get(f"{BASE_URL}{request_path}", headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve stats for {symbol}: HTTP status code {response.status_code}")
        return None

def get_product_price(symbol):
    """
    Get current price for a symbol
    """
    try:
        response = rest_client.get_market_trades(
            product_id=symbol,
            limit=1
        )
        if response and response.trades:
            return float(response.trades[0].price)
        return None
    except Exception as e:
        print(f"Error getting price: {str(e)}")
        return None

def get_available_products():
    """
    Get list of top 10 USD trading pairs by volume
    Returns: list of product IDs (e.g. ['BTC-USD', 'ETH-USD', ...]) or None on error
    """
    try:
        response = rest_client.get_products(verbose=False)  # Add verbose=False to suppress output
        if response and hasattr(response, 'products'):
            # Filter for USD pairs and convert to list of dictionaries
            products = [
                {
                    'product_id': p.product_id,
                    'volume': float(p.volume_24h) if hasattr(p, 'volume_24h') and p.volume_24h else 0
                }
                for p in response.products
                if p.product_id.endswith('-USD')  # Only include USD pairs
            ]
            
            # Sort by volume and get top 10
            sorted_products = sorted(products, key=lambda x: x['volume'], reverse=True)[:10]
            return [p['product_id'] for p in sorted_products]
        return None
    except Exception as e:
        print(f"Error getting products: {str(e)}")
        return None

def get_time_range(days_back=10):
    """
    Get time range for historical data queries
    Returns timestamps in ISO format required by Coinbase
    """
    now = datetime.now()
    start_date = now - timedelta(days=days_back)
    
    return start_date.isoformat(), now.isoformat()

def get_historical_data(symbol, granularity=3600, days_back=1):
    """
    Get historical candle data for a product
    Respects Coinbase's 350 candle limit
    """
    try:
        cprint(f"\n📊 Fetching historical data for {symbol}...", "white", "on_blue")
        end_time = datetime.now()  # Use UTC time
        start_time = end_time - timedelta(days=days_back)
        
        # Debug timezone info
        local_time = datetime.now()
        cprint(f"\n🕒 Timezone Debug:", "cyan")
        print(f"Local time: {local_time}")
        print(f"UTC time: {end_time}")
        print(f"Time difference: {local_time - end_time}")
        
        # Calculate how many candles we'll get based on granularity and days_back
        candles_requested = (days_back * 24 * 3600) / granularity
        
        # If we're requesting more than 300 candles (leaving buffer), adjust days_back
        if candles_requested > 300:
            # Adjust days_back to stay under limit
            days_back = (300 * granularity) / (24 * 3600)
            start_time = end_time - timedelta(days=days_back)
            cprint(f"⚠️ Adjusted request to {days_back:.1f} days to respect Coinbase's candle limit", "yellow")
        
        # Convert to Unix timestamps (ensure they're in the past)
        start_unix = int(start_time.timestamp())
        end_unix = int(end_time.timestamp())
        
        # Convert numeric granularity to Coinbase's expected format
        granularity_map = {
            60: "ONE_MINUTE",
            300: "FIVE_MINUTE",
            900: "FIFTEEN_MINUTE",
            1800: "THIRTY_MINUTE",
            3600: "ONE_HOUR",
            7200: "TWO_HOUR",
            21600: "SIX_HOUR",
            86400: "ONE_DAY"
        }
        
        cb_granularity = granularity_map.get(granularity, "ONE_HOUR")
        
        # Debug info
        cprint(f"\nRequest details:", "cyan")
        print(f"Symbol: {symbol}")
        print(f"Start time (UTC): {start_time}")
        print(f"End time (UTC): {end_time}")
        print(f"Start timestamp: {start_unix}")
        print(f"End timestamp: {end_unix}")
        print(f"Granularity: {cb_granularity}")
        
        response = rest_client.get_candles(
            product_id=symbol,
            start=start_unix,
            end=end_unix,
            granularity=cb_granularity
        )
        
        if not response:
            cprint(f"❌ No response received for {symbol}", "white", "on_red")
            return pd.DataFrame()
            
        if not hasattr(response, 'candles'):
            cprint(f"❌ No candles data in response for {symbol}", "white", "on_red")
            print("Response content:", response)
            return pd.DataFrame()
            
        if not response.candles:
            cprint(f"❌ Empty candles list for {symbol}", "white", "on_red")
            return pd.DataFrame()
        
        data = []
        for candle in response.candles:
            data.append({
                'start': int(candle.start),
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume)
            })
        
        if not data:
            cprint(f"⚠️ No candle data returned for {symbol}", "yellow")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['start'], unit='s')
        
        # Calculate indicators
        if len(df) > 0:
            cprint(f"✅ Successfully retrieved {len(df)} candles for {symbol}", "green")
            # Moving averages
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA40'] = df['close'].rolling(window=40).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Indicator crossovers and comparisons
            df['Price_above_MA20'] = df['close'] > df['MA20']
            df['Price_above_MA40'] = df['close'] > df['MA40']
            df['MA20_above_MA40'] = df['MA20'] > df['MA40']
        
        return df.sort_values('start', ascending=True)
        
    except Exception as e:
        cprint(f"❌ Error getting historical data for {symbol}: {str(e)}", "white", "on_red")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return pd.DataFrame()

def market_buy(symbol, amount, slippage=None):
    """
    Execute a market buy order
    symbol: Trading pair (e.g. 'BTC-USD')
    amount: Amount in quote currency (e.g. USD)
    """
    try:
        # Check available balance first
        balance = get_account_balance("USD")
        if not balance or balance['available'] < amount:
            cprint(f"❌ Insufficient USD balance. Need ${amount:.2f}, have ${balance['available']:.2f}", "white", "on_red")
            return None

        # Round amount to 2 decimal places for USD
        rounded_amount = round(float(amount), 2)
        
        # Generate a unique client_order_id using timestamp
        client_order_id = f"buy_{symbol}_{int(time.time() * 1000)}"
        
        response = rest_client.create_order(
            client_order_id=client_order_id,
            product_id=symbol,
            side='BUY',
            order_configuration={
                'market_market_ioc': {
                    'quote_size': str(rounded_amount)  # Use rounded amount
                }
            }
        )
        
        # Check response format
        if hasattr(response, 'success') and response.success:
            order_details = response.success_response
            order_id = order_details.get('order_id')
            cprint(f"🚀 Market buy order placed for {symbol}: Order ID {order_id}", "white", "on_green")
            return order_details
        else:
            error_msg = getattr(response, 'error_response', {}).get('message', 'Unknown error')
            error_details = getattr(response, 'error_response', {}).get('error_details', '')
            preview_failure = getattr(response, 'error_response', {}).get('preview_failure_reason', '')
            cprint(f"❌ Failed to place market buy order: {error_msg}", "white", "on_red")
            if error_details:
                cprint(f"Error details: {error_details}", "white", "on_red")
            if preview_failure:
                cprint(f"Preview failure: {preview_failure}", "white", "on_red")
            return None
            
    except Exception as e:
        cprint(f"❌ Error placing market buy order: {str(e)}", "white", "on_red")
        return None

def market_sell(symbol, amount, slippage=None):
    """
    Execute a market sell order
    symbol: Trading pair (e.g. 'BTC-USD')
    amount: Amount in base currency (e.g. BTC)
    """
    try:
        # Check available balance first
        base_currency = symbol.split('-')[0]
        position = get_position(symbol)
        
        if position < amount:
            cprint(f"❌ Insufficient {base_currency} balance. Need {amount:.8f}, have {position:.8f}", "white", "on_red")
            return None

        # Generate a unique client_order_id using timestamp
        client_order_id = f"sell_{symbol}_{int(time.time() * 1000)}"
        
        response = rest_client.create_order(
            client_order_id=client_order_id,
            product_id=symbol,
            side='SELL',
            order_configuration={
                'market_market_ioc': {
                    'base_size': str(amount)
                }
            }
        )
        
        # Check response format
        if hasattr(response, 'success') and response.success:
            order_details = response.success_response
            order_id = order_details.get('order_id')
            cprint(f"📉 Market sell order placed for {symbol}: Order ID {order_id}", "white", "on_blue")
            return order_details
        else:
            error_msg = getattr(response, 'error_response', {}).get('message', 'Unknown error')
            cprint(f"❌ Failed to place market sell order: {error_msg}", "white", "on_red")
            return None
            
    except Exception as e:
        cprint(f"❌ Error placing market sell order: {str(e)}", "white", "on_red")
        return None

def get_account_balance(currency='USD'):
    """
    Get account balance for a specific currency
    Documentation: GET /brokerage/accounts
    """
    try:
        # Use the REST client instead of direct API call
        response = rest_client.get_accounts()
        
        if response and hasattr(response, 'accounts'):
            for account in response.accounts:
                if account.currency == currency:
                    return {
                        'currency': currency,
                        'balance': float(account.available_balance['value']),
                        'available': float(account.available_balance['value']),
                        'hold': float(account.hold['value'])
                    }
        return None
            
    except Exception as e:
        print(f"Error getting account balance: {str(e)}")
        return None

def get_position(symbol):
    """
    Get current position size for a given trading pair
    Returns amount in base currency
    """
    base_currency = symbol.split('-')[0]
    
    try:
        # Use the REST client instead of direct API call
        response = rest_client.get_accounts()
        
        if response and hasattr(response, 'accounts'):
            for account in response.accounts:
                if account.currency == base_currency:
                    return float(account.available_balance['value'])
            return 0.0  # Return 0 if currency not found
        else:
            cprint(f"❌ Failed to get position: No accounts found", "white", "on_red")
            return 0.0
            
    except Exception as e:
        cprint(f"❌ Error getting position: {str(e)}", "white", "on_red")
        return 0.0

def chunk_kill(symbol, max_usd_order_size, slippage=None):
    """
    Kill a position in chunks
    """
    cprint(f"\n🔪 Moon Dev's AI Agent initiating position exit for {symbol}...", "white", "on_cyan")
    
    position_size = get_position(symbol)
    if position_size <= 0:
        cprint("❌ No position found to exit", "white", "on_red")
        return
        
    current_price = get_product_price(symbol)
    position_value = position_size * current_price
    
    cprint(f"📊 Current position: {position_size:.8f} {symbol.split('-')[0]} (${position_value:.2f})", "white", "on_cyan")
    
    while position_size > 0:
        # Calculate chunk size based on max USD order size
        chunk_value = min(position_value, max_usd_order_size)
        chunk_size = chunk_value / current_price
        chunk_size = min(chunk_size, position_size)  # Don't sell more than we have
        
        cprint(f"\n💫 Executing sell chunk of {chunk_size:.8f} {symbol.split('-')[0]}...", "white", "on_cyan")
        
        try:
            result = market_sell(symbol, chunk_size, slippage)
            if result:
                cprint(f"✅ Chunk sell complete", "white", "on_green")
            else:
                cprint(f"❌ Chunk sell failed", "white", "on_red")
                break
                
            time.sleep(2)  # Small delay between chunks
            
            # Update position info
            position_size = get_position(symbol)
            current_price = get_product_price(symbol)
            position_value = position_size * current_price
            
        except Exception as e:
            cprint(f"❌ Error during chunk sell: {str(e)}", "white", "on_red")
            break
    
    final_position = get_position(symbol)
    if final_position <= 0:
        cprint("\n✨ Position successfully closed!", "white", "on_green")
    else:
        cprint(f"\n⚠️ Position partially closed. Remaining: {final_position:.8f}", "white", "on_yellow")

def round_down(value, decimals):
    """
    Round down to specified number of decimals
    """
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

def elegant_entry(symbol, buy_under, usd_size, max_usd_order_size, orders_per_open=3, slippage=None, tx_sleep=2):
    """
    Elegant entry function for Coinbase
    Buys in chunks when price is under specified level
    """
    pos = get_position(symbol)
    price = get_product_price(symbol)
    pos_usd = pos * price
    size_needed = usd_size - pos_usd
    
    if size_needed > max_usd_order_size:
        chunk_size = max_usd_order_size
    else:
        chunk_size = size_needed

    if pos_usd > (0.97 * usd_size):
        return

    while pos_usd < (0.97 * usd_size) and (price < buy_under):
        try:
            for i in range(orders_per_open):
                result = market_buy(symbol, chunk_size, slippage)
                if result:
                    cprint(f'chunk buy submitted of {symbol} size: {chunk_size}', 'white', 'on_blue')
                time.sleep(1)

            time.sleep(tx_sleep)

            # Update position info
            pos = get_position(symbol)
            price = get_product_price(symbol)
            pos_usd = pos * price
            size_needed = usd_size - pos_usd
            
            if size_needed > max_usd_order_size:
                chunk_size = max_usd_order_size
            else:
                chunk_size = size_needed

        except Exception as e:
            try:
                cprint(f'trying again to make the order in 30 seconds.....', 'light_blue', 'on_light_magenta')
                time.sleep(30)
                
                for i in range(orders_per_open):
                    result = market_buy(symbol, chunk_size, slippage)
                    if result:
                        cprint(f'chunk buy submitted of {symbol} size: {chunk_size}', 'white', 'on_blue')
                    time.sleep(1)

                time.sleep(tx_sleep)
                
                # Update position info
                pos = get_position(symbol)
                price = get_product_price(symbol)
                pos_usd = pos * price
                size_needed = usd_size - pos_usd
                
                if size_needed > max_usd_order_size:
                    chunk_size = max_usd_order_size
                else:
                    chunk_size = size_needed

            except:
                cprint(f'Final Error in the buy, restart needed', 'white', 'on_red')
                break

def breakout_entry(symbol, breakout_price, usd_size, max_usd_order_size, orders_per_open=3, slippage=None, tx_sleep=2):
    """
    Breakout entry function for Coinbase
    Buys in chunks when price breaks above specified level
    """
    pos = get_position(symbol)
    price = get_product_price(symbol)
    pos_usd = pos * price
    size_needed = usd_size - pos_usd
    
    if size_needed > max_usd_order_size:
        chunk_size = max_usd_order_size
    else:
        chunk_size = size_needed

    if pos_usd > (0.97 * usd_size):
        return

    while pos_usd < (0.97 * usd_size) and (price > breakout_price):
        try:
            for i in range(orders_per_open):
                result = market_buy(symbol, chunk_size, slippage)
                if result:
                    cprint(f'chunk buy submitted of {symbol} size: {chunk_size}', 'white', 'on_blue')
                time.sleep(1)

            time.sleep(tx_sleep)

            # Update position info
            pos = get_position(symbol)
            price = get_product_price(symbol)
            pos_usd = pos * price
            size_needed = usd_size - pos_usd
            
            if size_needed > max_usd_order_size:
                chunk_size = max_usd_order_size
            else:
                chunk_size = size_needed

        except Exception as e:
            try:
                cprint(f'trying again to make the order in 30 seconds.....', 'light_blue', 'on_light_magenta')
                time.sleep(30)
                
                for i in range(orders_per_open):
                    result = market_buy(symbol, chunk_size, slippage)
                    if result:
                        cprint(f'chunk buy submitted of {symbol} size: {chunk_size}', 'white', 'on_blue')
                    time.sleep(1)

                time.sleep(tx_sleep)
                
                # Update position info
                pos = get_position(symbol)
                price = get_product_price(symbol)
                pos_usd = pos * price
                size_needed = usd_size - pos_usd
                
                if size_needed > max_usd_order_size:
                    chunk_size = max_usd_order_size
                else:
                    chunk_size = size_needed

            except:
                cprint(f'Final Error in the buy, restart needed', 'white', 'on_red')
                break

def ai_entry(symbol, amount, max_usd_order_size, orders_per_open=3, slippage=None, tx_sleep=2):
    """
    AI agent entry function for Moon Dev's trading system 🤖
    """
    cprint("🤖 Moon Dev's AI Trading Agent initiating position entry...", "white", "on_blue")
    
    pos = get_position(symbol)
    price = get_product_price(symbol)
    pos_usd = pos * price
    
    cprint(f"🎯 Target allocation: ${amount:.2f} USD", "white", "on_blue")
    cprint(f"📊 Current position: ${pos_usd:.2f} USD", "white", "on_blue")
    
    if pos_usd >= (amount * 0.97):
        cprint("✋ Position already at or above target size!", "white", "on_blue")
        return
        
    size_needed = amount - pos_usd
    if size_needed <= 0:
        cprint("🛑 No additional size needed", "white", "on_blue")
        return
        
    if size_needed > max_usd_order_size:
        chunk_size = max_usd_order_size
    else:
        chunk_size = size_needed
    
    cprint(f"💫 Entry chunk size: ${chunk_size:.2f}", "white", "on_blue")

    while pos_usd < (amount * 0.97):
        try:
            for i in range(orders_per_open):
                result = market_buy(symbol, chunk_size, slippage)
                if result:
                    cprint(f"🚀 AI Agent placed order {i+1}/{orders_per_open} for {symbol}", "white", "on_blue")
                time.sleep(1)

            time.sleep(tx_sleep)
            
            # Update position info
            pos = get_position(symbol)
            price = get_product_price(symbol)
            pos_usd = pos * price
            
            if pos_usd >= (amount * 0.97):
                break
                
            size_needed = amount - pos_usd
            if size_needed <= 0:
                break
                
            if size_needed > max_usd_order_size:
                chunk_size = max_usd_order_size
            else:
                chunk_size = size_needed

        except Exception as e:
            cprint(f"❌ Error during entry: {str(e)}", "white", "on_red")
            break
    
    final_pos = get_position(symbol)
    final_price = get_product_price(symbol)
    final_pos_usd = final_pos * final_price
    
    cprint(f"\n📊 Final position: {final_pos:.8f} {symbol.split('-')[0]} (${final_pos_usd:.2f})", "white", "on_blue")

def supply_demand_zones(symbol, timeframe=3600, limit=100):
    """
    Calculate supply and demand zones for a given symbol
    timeframe: in seconds (e.g., 3600 for 1h)
    limit: number of candles to analyze
    """
    sd_df = pd.DataFrame()
    
    # Get historical data
    df = get_historical_data(symbol, granularity=timeframe, days_back=int(limit/24))
    
    # Only keep the data for as many bars as limit says
    df = df[-limit:]

    # Calculate support and resistance, excluding the last two rows
    if len(df) > 2:
        df['support'] = df[:-2]['close'].min()
        df['resis'] = df[:-2]['close'].max()
    else:
        df['support'] = df['close'].min()
        df['resis'] = df['close'].max()

    supp = df.iloc[-1]['support']
    resis = df.iloc[-1]['resis']

    df['supp_lo'] = df[:-2]['low'].min()
    supp_lo = df.iloc[-1]['supp_lo']

    df['res_hi'] = df[:-2]['high'].max()
    res_hi = df.iloc[-1]['res_hi']

    sd_df['dz'] = [supp_lo, supp]
    sd_df['sz'] = [res_hi, resis]

    return sd_df

def close_all_positions():
    """
    Close all open positions except USD
    """
    # Get all accounts
    request_path = '/accounts'
    headers = get_coinbase_headers('GET', request_path)
    
    try:
        response = requests.get(f"{BASE_URL}{request_path}", headers=headers)
        
        if response.status_code == 200:
            accounts = response.json()
            
            # Filter for non-USD accounts with balance > 0
            active_positions = [
                account for account in accounts 
                if account['currency'] != 'USD' 
                and float(account['balance']) > 0
            ]
            
            for position in active_positions:
                currency = position['currency']
                balance = float(position['balance'])
                symbol = f"{currency}-USD"  # Assuming USD as quote currency
                
                cprint(f"\n🔄 Closing position for {symbol}...", "white", "on_cyan")
                
                # Get current value in USD
                price = get_product_price(symbol)
                if price:
                    position_value = balance * price
                    
                    if position_value > 1.0:  # Only close if worth more than $1
                        chunk_kill(symbol, max_usd_order_size=50000)  # Using a safe max order size
                    else:
                        cprint(f"Skipping {symbol} - position value too small: ${position_value:.2f}", "white", "on_yellow")
                else:
                    cprint(f"❌ Could not get price for {symbol}", "white", "on_red")
            
            cprint("\n✨ All positions processed!", "white", "on_green")
            
        else:
            cprint(f"❌ Failed to get accounts: {response.text}", "white", "on_red")
            
    except Exception as e:
        cprint(f"❌ Error closing all positions: {str(e)}", "white", "on_red")

def get_wallet_holdings():
    """
    Get all current wallet holdings and their USD values
    """
    try:
        response = rest_client.get_accounts()
        holdings = []
        
        if response and hasattr(response, 'accounts'):
            for account in response.accounts:
                # Extract balance from the available_balance dictionary
                balance = float(account.available_balance['value'])
                currency = account.currency
                
                if balance > 0:
                    if currency == 'USD':
                        usd_value = balance
                    else:
                        symbol = f"{currency}-USD"
                        price = get_product_price(symbol)
                        usd_value = balance * price if price else 0
                        
                    if usd_value >= 0.01:  # Only include positions worth more than 1 cent
                        holdings.append({
                            'Currency': currency,
                            'Balance': balance,
                            'USD Value': usd_value
                        })
            
            df = pd.DataFrame(holdings)
            if not df.empty:
                df = df.sort_values('USD Value', ascending=False)
                cprint("\nWallet Holdings:", 'white', 'on_green')
                print(df)
                cprint(f'Total USD balance: ${df["USD Value"].sum():.2f}', 'white', 'on_green')
            else:
                cprint("No holdings to display.", 'white', 'on_red')
                
            return df
            
        return pd.DataFrame()
    except Exception as e:
        print(f"Error getting wallet holdings: {str(e)}")
        return pd.DataFrame()

def get_wallet_token_single(symbol):
    """
    Get holdings for a specific trading pair
    """
    base_currency = symbol.split('-')[0]
    df = get_wallet_holdings()
    
    if not df.empty:
        return df[df['Currency'] == base_currency]
    return pd.DataFrame()

# Update test function to handle empty responses
def test_account_functions():
    """Test account and position related functions with detailed debugging"""
    cprint("\n💰 Testing Account Functions...", "white", "on_blue")
    
    try:
        # Debug API credentials
        cprint("\n🔑 Debugging API Credentials:", "yellow")
        api_key = os.getenv("COINBASE_API_KEY")
        secret_key = os.getenv("COINBASE_API_SECRET")[:50] + "..."  # Show first 50 chars
        print(f"API Key: {api_key}")
        print(f"Secret Key (truncated): {secret_key}")
        
        # Debug request details
        request_path = '/api/v3/brokerage/accounts'
        method = 'GET'
        
        cprint("\n📡 Request Details:", "yellow")
        print(f"Request Path: {request_path}")
        print(f"Method: {method}")
        
        # Generate and display headers
        headers = get_coinbase_headers(method, request_path)
        cprint("\n📨 Generated Headers:", "yellow")
        safe_headers = {
            k: (v[:50] + '...' if k == 'Authorization' else v)
            for k, v in headers.items()
        }
        print_pretty_json(safe_headers)
        
        # Make the request with full debugging
        cprint("\n🔄 Making API Request...", "yellow")
        full_url = f"https://api.coinbase.com{request_path}"
        print(f"Full URL: {full_url}")
        
        response = requests.get(full_url, headers=headers)
        
        cprint("\n📊 Response Details:", "yellow")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers:")
        print_pretty_json(dict(response.headers))
        
        if response.text:  # Only try to parse JSON if there's content
            print(f"Response Body:")
            try:
                print_pretty_json(response.json())
            except:
                print(f"Raw response: {response.text}")
        else:
            print("Empty response body")
        
        if response.status_code == 200:
            cprint("✅ API request successful!", "green")
        else:
            cprint(f"❌ API request failed with status {response.status_code}", "red")
            
    except Exception as e:
        cprint(f"❌ Error in account tests: {str(e)}", "red")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())

def test_credentials():
    """
    Test if the API credentials are properly configured
    """
    print("\nTesting API Credentials:")
    
    try:
        if rest_client is None:
            print("❌ REST client not initialized")
            return False
            
        # Test basic API call
        response = rest_client.get_accounts()
        if response and response.accounts:
            print("✅ API credentials working correctly")
            return True
        else:
            print("❌ API response invalid")
            return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        print("Are you sure you generated your key at https://cloud.coinbase.com/access/api ?")
        return False

# Initialize WebSocket client (if needed)
def init_websocket():
    """
    Initialize WebSocket client with handlers
    """
    def on_message(message):
        print(f"📨 Received: {message}")
        
    def on_open():
        print("🔗 WebSocket connected")
        
    def on_close():
        print("🔌 WebSocket disconnected")
    
    try:
        ws_client = WSClient(
            api_key=os.getenv("COINBASE_API_KEY"),
            api_secret=os.getenv("COINBASE_API_SECRET"),
            on_message=on_message,
            on_open=on_open,
            on_close=on_close
        )
        return ws_client
    except Exception as e:
        print(f"Error initializing WebSocket: {str(e)}")
        return None

def debug_credentials():
    """Debug function to check API credentials"""
    print("\nDebugging API Credentials:")
    
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    
    print("\nAPI Key:")
    print(f"- Present: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"- Format correct: {'Yes' if api_key.startswith('organizations/') else 'No'}")
        print(f"- Length: {len(api_key)}")
    
    print("\nSecret Key:")
    print(f"- Present: {'Yes' if api_secret else 'No'}")
    print(api_secret)
    if api_secret:
        print(f"- Starts correctly: {'Yes' if '-----BEGIN EC PRIVATE KEY-----' in api_secret else 'No'}")
        print(f"- Ends correctly: {'Yes' if '-----END EC PRIVATE KEY-----' in api_secret else 'No'}")
        print(f"- Contains newlines: {'Yes' if chr(92) + 'n' in api_secret else 'No'}")
        print(f"- Length: {len(api_secret)}")

def verify_trading_pair(symbol):
    """
    Verify if a trading pair is valid and active on Coinbase
    """
    try:
        overview = get_product_overview(symbol)
        if overview:
            status = overview.get('status', 'unknown')
            cprint(f"✅ {symbol} is valid (Status: {status})", "green")
            return True
        else:
            cprint(f"❌ {symbol} is not a valid trading pair", "white", "on_red")
            return False
    except Exception as e:
        cprint(f"❌ Error verifying {symbol}: {str(e)}", "white", "on_red")
        return False

def get_token_balance_usd(symbol):
    """
    Get token balance in USD for a given symbol (e.g. 'BTC-USD')
    Returns the USD value of the position
    """
    try:
        # Extract base currency from symbol (e.g. 'BTC' from 'BTC-USD')
        base_currency = symbol.split('-')[0]
        
        # Get position size in base currency
        position = get_position(symbol)
        
        # Get current price
        price = get_product_price(symbol)
        
        if position is not None and price is not None:
            usd_value = position * price
            return usd_value
        return 0.0
            
    except Exception as e:
        print(f"Error getting token balance in USD: {str(e)}")
        return 0.0 