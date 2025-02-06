"""
🧪 Test Suite for Moon Dev's Coinbase Functions
"""

from nice_funcs_cb import *
import time
from termcolor import colored, cprint
import pandas as pd

def test_market_data_functions():
    """Test basic market data retrieval functions"""
    cprint("\n🔍 Testing Market Data Functions...", "white", "on_blue")
    
    try:
        test_symbol = "BTC-USD"   
        
        # Test get_available_products
        cprint("\nTesting get_available_products()...", "cyan")
        products = get_available_products()
        if products:
            cprint(f"✅ Found available products", "green")
            if isinstance(products, dict) and 'products' in products:
                # Only print the count, not the full product list
                cprint(f"Total number of products: {len(products.get('products', []))}", "green")
        else:
            cprint("❌ Failed to get available products", "red")
            
        # Test get_product_overview
        cprint(f"\nTesting get_product_overview() for {test_symbol}...", "cyan")
        overview = get_product_overview(test_symbol)
        if overview:
            cprint("✅ Successfully retrieved product overview:", "green")
            # Only print essential info instead of the full overview
            essential_info = {
                'product_id': overview.get('product_id'),
                'price': overview.get('price'),
                'status': overview.get('status'),
                'volume_24h': overview.get('volume_24h')
            }
            print_pretty_json(essential_info)
        else:
            cprint("❌ Failed to get product overview", "red")
            
        # Test get_product_price
        cprint(f"\nTesting get_product_price() for {test_symbol}...", "cyan")
        price = get_product_price(test_symbol)
        if price:
            cprint(f"✅ Current price: ${price:,.2f}", "green")
        else:
            cprint("❌ Failed to get price", "red")
            
        # Test get_historical_data
        cprint(f"\nTesting get_historical_data() for {test_symbol}...", "cyan")
        hist_data = get_historical_data(test_symbol, granularity=3600, days_back=1)
        if not hist_data.empty:
            cprint(f"✅ Retrieved {len(hist_data)} candles", "green")
            print(hist_data.head())
        else:
            cprint("❌ Failed to get historical data", "red")
            
        # Test supply_demand_zones
        cprint(f"\nTesting supply_demand_zones() for {test_symbol}...", "cyan")
        zones = supply_demand_zones(test_symbol, timeframe=3600, limit=24)
        if not zones.empty:
            cprint("✅ Successfully calculated supply and demand zones:", "green")
            print(zones)
        else:
            cprint("❌ Failed to calculate supply and demand zones", "red")
            
    except Exception as e:
        cprint(f"❌ Error in market data tests: {str(e)}", "red")

def test_account_functions():
    """Test account and wallet related functions"""
    cprint("\n💰 Testing Account Functions...", "white", "on_blue")
    
    try:
        # Test get_wallet_holdings
        cprint("\nTesting get_wallet_holdings()...", "cyan")
        holdings = get_wallet_holdings()
        if not holdings.empty:
            cprint("✅ Successfully retrieved wallet holdings", "green")
            print(holdings)
        else:
            cprint("⚠️ No holdings found or error occurred", "yellow")

        # Test get_wallet_token_single
        test_symbol = "TOSHI-USD"
        cprint(f"\nTesting get_wallet_token_single() for {test_symbol}...", "cyan")
        token_holdings = get_wallet_token_single(test_symbol)
        if not token_holdings.empty:
            cprint("✅ Successfully retrieved token holdings", "green")
            print(token_holdings)
        else:
            cprint("⚠️ No holdings found for this token", "yellow")

        # Test get_account_balance
        cprint("\nTesting get_account_balance()...", "cyan")
        balance = get_account_balance("USD")
        if balance:
            cprint("✅ Successfully retrieved account balance:", "green")
            print_pretty_json(balance)
        else:
            cprint("❌ Failed to get account balance", "red")

        # Test get_position
        cprint(f"\nTesting get_position() for {test_symbol}...", "cyan")
        position = get_position(test_symbol)
        if position is not None:
            cprint(f"✅ Current position: {position}", "green")
        else:
            cprint("❌ Failed to get position", "red")
            
    except Exception as e:
        cprint(f"❌ Error in account tests: {str(e)}", "red")

def test_order_functions():
    """Test order-related functions with minimal amounts"""
    cprint("\n📊 Testing Order Functions...", "white", "on_blue")
    
    try:
        test_symbol = "BTC-USD"
        
        # First check available balance
        cprint("\nChecking available USD balance...", "cyan")
        balance = get_account_balance("USD")
        if balance:
            available_usd = balance['available']
            cprint(f"Available USD balance: ${available_usd:.2f}", "green")
            
            # Only proceed with order tests if we have enough balance
            if available_usd < 1.0:
                cprint("⚠️ Insufficient USD balance for order tests (need at least $1.00)", "yellow")
                cprint("Please add funds to your account to test order functions", "yellow")
                return
                
            # Test market buy with 90% of available balance or $1.00, whichever is smaller
            test_buy_amount = min(round(available_usd * 0.9, 2), 1.00)  # Round to 2 decimals
            cprint(f"\nTesting market_buy() with ${test_buy_amount:.2f}...", "cyan")
            buy_result = market_buy(test_symbol, test_buy_amount)
            
            if buy_result:
                cprint("✅ Successfully placed market buy order", "green")
                print_pretty_json(buy_result)
                
                # Wait for order to settle
                time.sleep(2)
                
                # Get current position
                position = get_position(test_symbol)
                if position > 0:
                    cprint(f"\nTesting market_sell() with {position:.8f} BTC...", "cyan")
                    sell_result = market_sell(test_symbol, position)
                    if sell_result:
                        cprint("✅ Successfully placed market sell order", "green")
                        print_pretty_json(sell_result)
                    else:
                        cprint("❌ Failed to place market sell", "red")
                else:
                    cprint("⚠️ No position to sell after buy order", "yellow")
            else:
                cprint("❌ Failed to place market buy", "red")
        else:
            cprint("❌ Failed to get account balance", "red")
            
    except Exception as e:
        cprint(f"❌ Error in order tests: {str(e)}", "red")

def test_entry_functions():
    """Test entry strategy functions with minimal amounts"""
    cprint("\n📈 Testing Entry Functions...", "white", "on_blue")
    
    try:
        test_symbol = "BTC-USD"
        small_amount = 10  # $10 USD
        max_order_size = 5  # $5 USD per order
        
        # Test elegant_entry
        cprint("\nTesting elegant_entry()...", "cyan")
        current_price = get_product_price(test_symbol)
        if current_price:
            elegant_entry(
                symbol=test_symbol,
                buy_under=current_price * 1.1,  # Set higher than current price
                usd_size=small_amount,
                max_usd_order_size=max_order_size,
                orders_per_open=2
            )
        
        time.sleep(2)  # Wait between tests
        
        # Test breakout_entry
        cprint("\nTesting breakout_entry()...", "cyan")
        if current_price:
            breakout_entry(
                symbol=test_symbol,
                breakout_price=current_price * 0.9,  # Set lower than current price
                usd_size=small_amount,
                max_usd_order_size=max_order_size,
                orders_per_open=2
            )
        
        time.sleep(2)  # Wait between tests
        
        # Test ai_entry
        cprint("\nTesting ai_entry()...", "cyan")
        ai_entry(
            symbol=test_symbol,
            amount=small_amount,
            max_usd_order_size=max_order_size,
            orders_per_open=2
        )
        
    except Exception as e:
        cprint(f"❌ Error in entry function tests: {str(e)}", "red")

def run_all_tests():
    """Run all test functions"""
    cprint("\n🚀 Starting Moon Dev's Coinbase Function Tests...\n", "white", "on_green")
    
    try:
        # First test credentials
        debug_credentials()
        if not test_credentials():
            cprint("\n❌ Stopping tests due to credential failure", "red")
            return
            
        # Market data functions
        #test_market_data_functions()
        
        #time.sleep(1)  # Respect rate limits
        
        # Account functions
        #test_account_functions()
        time.sleep(1)  # Respect rate limits
        
        # Order functions (with minimal amounts)
        #test_order_functions()
        time.sleep(1)  # Respect rate limits
        
        # Entry strategy functions (with minimal amounts)
        #test_entry_functions()
        
        cprint("\n✨ All tests completed!", "white", "on_green")
        
    except Exception as e:
        cprint(f"\n❌ Test suite error: {str(e)}", "white", "on_red")

if __name__ == "__main__":
    run_all_tests() 