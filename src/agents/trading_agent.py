"""
🌙 Billy Bitcoin's Trading Agent
Coordinates strategy signals and executes trades
"""

# Keep only these prompts
TRADING_PROMPT = """
You are Billy Bitcoin's AI Trading Assistant 🌙

Analyze the provided market data and strategy signals (if available) to make a trading decision.

Market Data Criteria:
1. Price action relative to MA20 and MA40
2. RSI levels and trend
3. Volume patterns
4. Recent price movements

{strategy_context}

Respond in this exact format:
1. First line must be one of: BUY, SELL, or NOTHING (in caps)
2. Then explain your reasoning, including:
   - Technical analysis
   - Strategy signals analysis (if available)
   - Risk factors
   - Market conditions
   - Confidence level (as a percentage, e.g. 75%)

Remember: 
- Billy Bitcoin always prioritizes risk management! 🛡️
- Never trade USDC or SOL directly
- Consider both technical and strategy signals
"""

ALLOCATION_PROMPT = """
You are Billy Bitcoin's Portfolio Allocation Assistant 🌙

Given the total portfolio size and trading recommendations, allocate capital efficiently.
Consider:
1. Position sizing based on confidence levels
2. Risk distribution
3. Keep cash buffer as specified
4. Maximum allocation per position

Format your response as a Python dictionary:
{
    "token_address": allocated_amount,  # In USD
    ...
    "USDC_ADDRESS": remaining_cash  # Always use USDC_ADDRESS for cash
}

Remember:
- Total allocations must not exceed total_size
- Higher confidence should get larger allocations
- Never allocate more than {MAX_POSITION_PERCENTAGE}% to a single position
- Keep at least {CASH_PERCENTAGE}% in USDC as safety buffer
- Only allocate to BUY recommendations
- Cash must be stored as USDC using USDC_ADDRESS: {USDC_ADDRESS}
"""

import anthropic
import os
import pandas as pd
import json
from termcolor import colored, cprint
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# Local imports
from src.config import *
from src import config
from src import nice_funcs_cb as cb
#Sfrom src.data.ohlcv_collector import collect_all_tokens
from nice_funcs import get_data
from src.agents.base_agent import BaseAgent
from src.agents.chartanalysis_agent import ChartAnalysisAgent
from src.agents.strategy_agent import StrategyAgent

# Load environment variables
load_dotenv()

# Add at top of file with other constants
VALID_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
DEFAULT_TIMEFRAME = '15m'
DEFAULT_BARS = 24

class TradingAgent(BaseAgent):
    def __init__(self):
        """Initialize the Trading Agent"""
        super().__init__(agent_type='trading')
        
        # Get available trading pairs from Coinbase first
        self.symbols = self._get_trading_pairs()
        
        # Initialize other agents
        self.chart_agent = ChartAnalysisAgent(symbols=self.symbols)  # Pass symbols to chart agent
        self.strategy_agent = StrategyAgent()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        
        # Get initial portfolio state from Coinbase
        self.portfolio = self.get_portfolio_state()
        
        self.recommendations_df = pd.DataFrame(columns=['token', 'action', 'confidence', 'reasoning'])
        self.strategy_signals = {}  # Initialize empty strategy signals dict
        print("🤖 Billy Bitcoin's LLM Trading Agent initialized!")

    def _get_trading_pairs(self):
        """Get top trading pairs from Coinbase by volume"""
        try:
            # Use existing function from nice_funcs_cb
            products = cb.get_available_products()
            
            if not products:
                print("⚠️ No pairs found, defaulting to BTC-USD")
                return ["BTC-USD"]
            
            # Print pairs for visibility
            print("\n📊 Available USD trading pairs:")
            for pair in products[:2]:  # Show first 2 pairs
                print(f"🪙 {pair}")
            
            return products[:2]  # Return top pairs
            
        except Exception as e:
            print(f"❌ Error getting trading pairs: {str(e)}")
            print("⚠️ Defaulting to BTC-USD")
            return ["BTC-USD"]

    def get_portfolio_state(self):
        """Get current portfolio state from Coinbase"""
        try:
            # Get USD balance
            usd_balance = cb.get_usd_balance()
            portfolio = {'USDC-USD': usd_balance}
            
            # Get other token balances
            for token in self.symbols:
                if token != 'USDC-USD':
                    balance = cb.get_token_balance_usd(token)
                    if balance > 0:
                        portfolio[token] = balance
            
            return portfolio
        except Exception as e:
            print(f"❌ Error getting portfolio state: {e}")
            return {}

    def analyze_market_data(self, token, market_data, strategy_signals=None):
        """Analyze market data using Claude"""
        try:
            # Skip analysis for excluded tokens
            if token in EXCLUDED_TOKENS:
                print(f"⚠️ Skipping analysis for excluded token: {token}")
                return None
            
            # Check if strategy signals are required
            if config.REQUIRE_STRATEGY_SIGNALS and not strategy_signals:
                print(f"⏭️ Skipping {token} - No strategy signals available and REQUIRE_STRATEGY_SIGNALS=True")
                return None
            
            # Prepare strategy context
            strategy_context = ""
            if strategy_signals and token in strategy_signals:
                strategy_context = f"""
Strategy Signals Available:
{json.dumps(strategy_signals[token], indent=2)}
                """
            else:
                strategy_context = "No strategy signals available."
            
            message = self.client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,
                messages=[
                    {
                        "role": "user", 
                        "content": f"{TRADING_PROMPT.format(strategy_context=strategy_context)}\n\nMarket Data to Analyze:\n{market_data}"
                    }
                ]
            )
            
            # Parse the response - handle both string and list responses
            response = message.content
            if isinstance(response, list):
                # Extract text from TextBlock objects if present
                response = '\n'.join([
                    item.text if hasattr(item, 'text') else str(item)
                    for item in response
                ])
            
            lines = response.split('\n')
            action = lines[0].strip() if lines else "NOTHING"
            
            # Extract confidence from the response (assuming it's mentioned as a percentage)
            confidence = 0
            for line in lines:
                if 'confidence' in line.lower():
                    # Extract number from string like "Confidence: 75%"
                    try:
                        confidence = int(''.join(filter(str.isdigit, line)))
                    except:
                        confidence = 50  # Default if not found
            
            # Add to recommendations DataFrame with proper reasoning
            reasoning = '\n'.join(lines[1:]) if len(lines) > 1 else "No detailed reasoning provided"
            self.recommendations_df = pd.concat([
                self.recommendations_df,
                pd.DataFrame([{
                    'token': token,
                    'action': action,
                    'confidence': confidence,
                    'reasoning': reasoning
                }])
            ], ignore_index=True)
            
            print(f"🎯 Billy Bitcoin's AI Analysis Complete for {token[:4]}!")
            return response
            
        except Exception as e:
            print(f"❌ Error in AI analysis: {str(e)}")
            # Still add to DataFrame even on error, but mark as NOTHING with 0 confidence
            self.recommendations_df = pd.concat([
                self.recommendations_df,
                pd.DataFrame([{
                    'token': token,
                    'action': "NOTHING",
                    'confidence': 0,
                    'reasoning': f"Error during analysis: {str(e)}"
                }])
            ], ignore_index=True)
            return None
    
    def allocate_portfolio(self):
        """Get AI-recommended portfolio allocation"""
        try:
            cprint("\n💰 Calculating optimal portfolio allocation...", "cyan")
            max_position_size = usd_size * (MAX_POSITION_PERCENTAGE / 100)
            cprint(f"🎯 Maximum position size: ${max_position_size:.2f} ({MAX_POSITION_PERCENTAGE}% of ${usd_size:.2f})", "cyan")
            
            # Filter for BUY recommendations only
            buy_recommendations = self.recommendations_df[
                self.recommendations_df['action'] == "BUY"
            ]['token'].tolist()
            
            if not buy_recommendations:
                cprint("ℹ️ No BUY recommendations found - keeping funds in USDC", "yellow")
                return {"USDC-USD": usd_size}
            
            # Get available products from Coinbase
            available_products = cb.get_available_products()
            if not available_products:
                cprint("❌ Could not get available products from Coinbase", "red")
                return None
            
            # Filter trading pairs to only include BUY recommendations
            trading_pairs = [
                pair for pair in available_products 
                if pair in buy_recommendations and
                pair.endswith('-USD') and 
                not any(stable in pair for stable in ['USDC', 'USDT', 'DAI', 'UST'])
            ]
            
            # Get allocation from AI
            message = self.client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,
                messages=[{
                    "role": "user", 
                    "content": f"""You are Billy Bitcoin's Portfolio Allocation AI 🌙

Given:
- Total portfolio size: ${usd_size}
- Maximum position size: ${max_position_size} ({MAX_POSITION_PERCENTAGE}% of total)
- Minimum cash (USDC) buffer: {CASH_PERCENTAGE}%
- Available trading pairs: {trading_pairs}
- USDC-USD pair for cash allocation

Provide a portfolio allocation that:
1. Never exceeds max position size per token
2. Maintains minimum cash buffer in USD 
3. Returns allocation as a JSON object with trading pairs as keys and USD amounts as values
4. Uses 'USD' for cash allocation

Example format:
{{
    "BTC-USD": amount_in_usd,
    "ETH-USD": amount_in_usd,
    "USDC-USD": cash_amount
}}"""
                }]
            )
            
            # Parse the response
            allocations = self.parse_allocation_response(str(message.content))
            if not allocations:
                return None
            
            # Validate allocation totals
            total_allocated = sum(allocations.values())
            if total_allocated > usd_size:
                cprint(f"❌ Total allocation ${total_allocated:.2f} exceeds portfolio size ${usd_size:.2f}", "red")
                return None
            
            # Print allocations
            cprint("\n📊 Portfolio Allocation:", "green")
            for token, amount in allocations.items():
                token_display = "USDC" if token == 'USDC-USD' else token
                cprint(f"  • {token_display}: ${amount:.2f}", "green")
            
            return allocations
            
        except Exception as e:
            cprint(f"❌ Error in portfolio allocation: {str(e)}", "red")
            return None

    def execute_allocations(self, allocation_dict):
        """Execute the allocations using AI entry for each position"""
        try:
            print("\n🚀 Billy Bitcoin executing portfolio allocations...")
            
            for token, amount in allocation_dict.items():
                # Check recommendation before proceeding
                token_recommendation = self.recommendations_df[
                    self.recommendations_df['token'] == token
                ]
                
                if not token_recommendation.empty:
                    action = token_recommendation.iloc[0]['action']
                    if action != "BUY":
                        print(f"⏭️ Skipping {token} - AI recommends {action}, not BUY")
                        continue
                
                if token in EXCLUDED_TOKENS:
                    print(f"💵 Keeping ${amount:.2f} in {token}")
                    continue
                    
                print(f"\n🎯 Processing allocation for {token}...")
                
                try:
                    print("📊 Attempting to get position from Coinbase...")
                    current_position = None
                    
                    try:
                        current_position = cb.get_token_balance_usd(token)
                        print(f"✅ Successfully got position from Coinbase: ${current_position:.2f}")
                    except Exception as cb_error:
                        print(f"⚠️ Coinbase position check failed: {str(cb_error)}")
                        continue
                    
                    if current_position is None:
                        print("❌ Could not get current position from Coinbase")
                        continue
                        
                    target_allocation = amount
                    print(f"🎯 Target allocation: ${target_allocation:.2f} USD")
                    print(f"📊 Current position: ${current_position:.2f} USD")
                    
                    if current_position < target_allocation:
                        print(f"✨ Executing entry for {token}")
                        try:
                            available_usd = cb.get_usd_balance()
                            entry_amount = target_allocation - current_position
                            
                            if available_usd < entry_amount:
                                print(f"❌ Insufficient USD balance (${available_usd:.2f}) for entry (${entry_amount:.2f})")
                                continue
                                
                            # Simple market order for smaller amounts
                            if entry_amount <= 10000:
                                print(f"📈 Placing single market order for ${entry_amount:.2f}")
                                success = cb.market_buy(token, entry_amount)
                            else:
                                print(f"📈 Placing chunked order for ${entry_amount:.2f}")
                                # Pass max_usd_order_size from config
                                success = cb.ai_entry(token, entry_amount, max_usd_order_size=config.max_usd_order_size)
                                
                            if success:
                                print(f"✅ Entry successful for {token}")
                            else:
                                print(f"❌ Entry failed for {token}")
                                
                        except Exception as entry_error:
                            if "INSUFFICIENT_FUND" in str(entry_error):
                                print(f"❌ Insufficient funds for {token} entry - skipping")
                            else:
                                print(f"❌ Entry error: {str(entry_error)}")
                            continue
                    else:
                        print(f"⏸️ Position already at target size for {token}")

                except Exception as e:
                    print(f"❌ Error executing entry for {token}: {str(e)}")
                    print(f"Full error details: {type(e).__name__}")
                    continue  # Skip to next token on error
                
                time.sleep(2)  # Small delay between entries
                
        except Exception as e:
            print(f"❌ Error executing allocations: {str(e)}")
            print("🔧 Billy Bitcoin suggests checking the logs and trying again!")

    def handle_exits(self):
        """Check and exit positions based on SELL or NOTHING recommendations"""
        cprint("\n🔄 Checking for positions to exit...", "white", "on_blue")
        
        for _, row in self.recommendations_df.iterrows():
            token = row['token']
            
            if token in EXCLUDED_TOKENS:
                continue
                
            action = row['action']
            
            try:
                # Get position details
                current_position_usd = cb.get_token_balance_usd(token)
                if current_position_usd <= 0:
                    continue

                if action in ["SELL", "NOTHING"]:
                    cprint(f"\n🚫 AI Agent recommends {action} for {token}", "white", "on_yellow")
                    cprint(f"💰 Current position: ${current_position_usd:.2f}", "white", "on_blue")
                    
                    try:
                        cprint(f"📉 Attempting to close position...", "white", "on_cyan")
                        
                        # Get current price to calculate base amount
                        current_price = cb.get_product_price(token)
                        if current_price <= 0:
                            cprint(f"❌ Could not get current price for {token}", "white", "on_red")
                            continue
                            
                        # Calculate base amount (token amount) with extra precision
                        base_amount = current_position_usd / current_price
                        cprint(f"📊 Selling {base_amount:.8f} {token} @ ${current_price:.8f}", "white", "on_blue")
                        
                        # Execute market sell using base amount
                        if current_position_usd <= 10000:
                            # Use sell_token_amount instead of market_sell
                            success = cb.sell_token_amount(token, base_amount)
                        else:
                            success = cb.chunk_kill(token, max_usd_order_size=50000)
                        
                        # Verify the position was closed
                        time.sleep(3)  # Increased wait time
                        new_position = cb.get_token_balance_usd(token)
                        
                        if new_position <= 0.01:  # Allow for dust
                            cprint(f"✅ Position successfully closed for {token}", "white", "on_green")
                        else:
                            cprint(f"⚠️ Position may not be fully closed. Remaining balance: ${new_position:.2f}", "white", "on_yellow")
                            # Attempt one more time if partial fill
                            if new_position > 0.01:
                                cprint(f"🔄 Attempting to close remaining position...", "white", "on_cyan")
                                # Recalculate base amount for remaining position
                                remaining_base = new_position / cb.get_product_price(token)
                                success = cb.sell_token_amount(token, remaining_base)
                                time.sleep(3)
                                final_position = cb.get_token_balance_usd(token)
                                if final_position <= 0.01:
                                    cprint(f"✅ Remaining position closed successfully", "white", "on_green")
                                else:
                                    cprint(f"❌ Could not fully close position. Final balance: ${final_position:.2f}", "white", "on_red")
                                    
                    except Exception as e:
                        cprint(f"❌ Error closing position: {str(e)}", "white", "on_red")
                        cprint(f"Full error details: {type(e).__name__}", "white", "on_red")
            except Exception as e:
                cprint(f"❌ Error checking position for {token}: {str(e)}", "white", "on_red")
                continue

    def parse_allocation_response(self, response):
        """Parse the AI's allocation response and handle both string and TextBlock formats"""
        try:
            # Handle TextBlock format from Claude 3
            if isinstance(response, list):
                response = response[0].text if hasattr(response[0], 'text') else str(response[0])
            
            print("🔍 Raw response received:")
            print(response)
            
            # Find the JSON block between curly braces
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start:end]
            
            # More aggressive JSON cleaning
            json_str = (json_str
                .replace('\n', '')          # Remove newlines
                .replace('    ', '')        # Remove indentation
                .replace('\t', '')          # Remove tabs
                .replace('\\n', '')         # Remove escaped newlines
                .replace(' ', '')           # Remove all spaces
                .strip())                   # Remove leading/trailing whitespace
            
            print("\n🧹 Cleaned JSON string:")
            print(json_str)
            
            # Parse the cleaned JSON
            allocations = json.loads(json_str)
            
            print("\n📊 Parsed allocations:")
            for token, amount in allocations.items():
                print(f"  • {token}: ${amount}")
            
            # Validate amounts are numbers
            for token, amount in allocations.items():
                if not isinstance(amount, (int, float)):
                    raise ValueError(f"Invalid amount type for {token}: {type(amount)}")
                if amount < 0:
                    raise ValueError(f"Negative allocation for {token}: {amount}")
            
            return allocations
            
        except Exception as e:
            print(f"❌ Error parsing allocation response: {str(e)}")
            print("🔍 Raw response:")
            print(response)
            return None

    def parse_portfolio_allocation(self, allocation_text):
        """Parse portfolio allocation from text response"""
        try:
            # Clean up the response text
            cleaned_text = allocation_text.strip()
            if "```json" in cleaned_text:
                # Extract JSON from code block if present
                json_str = cleaned_text.split("```json")[1].split("```")[0]
            else:
                # Find the JSON object between curly braces
                start = cleaned_text.find('{')
                end = cleaned_text.rfind('}') + 1
                json_str = cleaned_text[start:end]
            
            # Parse the JSON
            allocations = json.loads(json_str)
            
            print("\n📊 Parsed allocations:")
            for token, amount in allocations.items():
                print(f"  • {token}: ${amount}")
            
            return allocations
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing allocation JSON: {e}")
            print(f"🔍 Raw text received:\n{allocation_text}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error parsing allocations: {e}")
            return None

    def get_data_from_sources(self, token, timeframe=DEFAULT_TIMEFRAME, bars=DEFAULT_BARS):
        """Helper function to get data from available sources, prioritizing Coinbase
        
        Args:
            token (str): Trading pair symbol
            timeframe (str): Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            bars (int): Number of candles to fetch
        """
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe. Must be one of: {VALID_TIMEFRAMES}")
        
        data = None
        
        # Try Coinbase first using get_historical_data
        try:
            # Convert timeframe to seconds for Coinbase
            granularity_map = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '1h': 3600,
                '4h': 14400,
                '1d': 86400
            }
            
            granularity = granularity_map[timeframe]
            
            data = cb.get_historical_data(
                symbol=token,
                granularity=granularity,  # Now using proper granularity mapping
                days_back=bars  # Coinbase uses 'limit' instead of 'bars'
            )
            if data is not None:
                print(f"✅ Got data from Coinbase for {token} using {timeframe} timeframe")
                return data
        except Exception as e:
            print(f"📝 Coinbase data fetch failed: {str(e)}")

        # Try other sources if Coinbase fails
        for module in [cb]:  # Add any new nice_funcs modules here
            try:
                data = module.get_data(token, timeframe=timeframe, bars=bars)
                if data is not None:
                    print(f"✅ Got data from alternate source for {token}")
                    return data
            except Exception:
                continue
            
        return data

    def get_products_from_sources(self):
        """Helper function to get available products, prioritizing Coinbase"""
        products = None
        
        # Try Coinbase first
        try:
            products = cb.get_available_products()
            if products:
                print("✅ Got products from Coinbase")
                return products
        except Exception as e:
            print(f"📝 Coinbase products fetch failed: {str(e)}")

        # Try other sources if Coinbase fails
        for module in [cb]:  # Add any new nice_funcs modules here
            try:
                products = module.get_available_products()
                if products:
                    print(f"✅ Got products from alternate source")
                    return products
            except Exception:
                continue
            
        return products

    def run(self):
        """Run the trading agent (implements BaseAgent interface)"""
        self.run_trading_cycle()

    def run_trading_cycle(self):
        """Run one trading cycle"""
        try:
            # 1. Get strategy signals
            signals = {}
            for token in self.symbols:
                token_signals = self.strategy_agent.get_signals(token)
                if token_signals:
                    signals[token] = token_signals
            
            # 2. Get chart analysis signals (now with proper error handling)
            chart_signals = self.chart_agent.run_monitoring_cycle() or {}
            
            # 3. Combine signals and execute trades
            for token in self.symbols:
                # Skip excluded tokens
                if token in EXCLUDED_TOKENS:
                    continue
                    
                # Combine signals from both sources
                token_signals = signals.get(token, [])
                token_chart = chart_signals.get(token, {})
                
                # If we have signals to act on
                if token_signals or any(chart.get('action') in ['BUY', 'SELL'] 
                                      for chart in token_chart.values()):
                    if token_signals:
                        # Execute strategy-based trades
                        self.execute_strategy_signals(token_signals)
                    elif any(chart.get('action') == 'SELL' for chart in token_chart.values()):
                        # Handle exits based on chart analysis
                        self.handle_exit(token)
                        
        except Exception as e:
            print(f"❌ Error in trading cycle: {str(e)}")

    def execute_strategy_signals(self, signals):
        """Execute trades based on strategy signals"""
        for signal in signals:
            token = signal['token']
            direction = signal['direction']
            target_size = self.calculate_position_size(signal)
            
            if direction == 'BUY':
                cb.ai_entry(token, target_size)
            elif direction == 'SELL':
                cb.chunk_kill(token, max_usd_order_size, slippage)

    def handle_exit(self, token):
        """Handle position exit"""
        try:
            current_position = cb.get_token_balance_usd(token)
            if current_position > 0:
                cb.chunk_kill(token, max_usd_order_size, slippage)
        except Exception as e:
            print(f"❌ Error exiting position: {str(e)}")

    def calculate_position_size(self, signal):
        """Calculate position size based on signal strength and portfolio size"""
        try:
            # Get signal confidence
            confidence = signal.get('confidence', 0)
            if isinstance(confidence, str):
                confidence = float(confidence.strip('%'))
            confidence = float(confidence) / 100  # Convert to decimal
            
            # Calculate maximum position size based on portfolio
            max_position = usd_size * (MAX_POSITION_PERCENTAGE / 100)
            
            # Scale position size by confidence
            position_size = max_position * confidence
            
            # Ensure minimum position size
            MIN_POSITION_SIZE = 10  # $10 minimum
            if position_size < MIN_POSITION_SIZE:
                return 0
            
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculating position size: {str(e)}")
            return 0

def main():
    """Main function to run the trading agent every 15 minutes"""
    cprint("🌙 Billy Bitcoin AI Trading System Starting Up! 🚀", "white", "on_blue")
    
    agent = TradingAgent()
    INTERVAL = SLEEP_BETWEEN_RUNS_MINUTES * 60  # Convert minutes to seconds
    
    while True:
        try:
            agent.run_trading_cycle()
            
            next_run = datetime.now() + timedelta(minutes=SLEEP_BETWEEN_RUNS_MINUTES)
            cprint(f"\n⏳ AI Agent run complete. Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}", "white", "on_green")
            
            # Sleep until next interval
            time.sleep(INTERVAL)
                
        except KeyboardInterrupt:
            cprint("\n👋 Billy Bitcoin AI Agent shutting down gracefully...", "white", "on_blue")
            break
        except Exception as e:
            cprint(f"\n❌ Error: {str(e)}", "white", "on_red")
            cprint("🔧 Billy Bitcoin suggests checking the logs and trying again!", "white", "on_blue")
            # Still sleep and continue on error
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main() 