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
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import asyncio
from functools import wraps
from prometheus_client import Counter, Gauge
from pydantic_settings import BaseSettings
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import *  # Import all config variables

# Local imports
from src import config    # This allows using config.CONSTANT_NAME if needed
from src import nice_funcs_cb as cb
from src.agents.base_agent import BaseAgent
from src.agents.chartanalysis_agent import ChartAnalysisAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.token_monitor_agent import TokenMonitorAgent

# Load environment variables
load_dotenv()

@dataclass
class TradingSignal:
    token: str
    chart_action: str
    chart_confidence: float
    strategy_signals: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failures = 0
        self.last_failure = None
        self.threshold = failure_threshold
        self.timeout = reset_timeout

    def can_execute(self) -> bool:
        if self.last_failure and (datetime.now() - self.last_failure).seconds > self.timeout:
            self.reset()
        return self.failures < self.threshold

def rate_limit(calls: int, period: int):
    def decorator(func):
        last_reset = time.time()
        calls_made = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_reset, calls_made
            
            current_time = time.time()
            if current_time - last_reset > period:
                calls_made = 0
                last_reset = current_time
                
            if calls_made >= calls:
                sleep_time = period - (current_time - last_reset)
                time.sleep(max(0, sleep_time))
                calls_made = 0
                last_reset = time.time()
                
            calls_made += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

class MetricsCollector:
    def __init__(self):
        self.trades_executed = Counter('trades_executed', 'Number of trades executed')
        self.position_size = Gauge('position_size', 'Current position size', ['token'])
        self.analysis_duration = Gauge('analysis_duration', 'Time taken for analysis')

class TradingAgent(BaseAgent):
    def __init__(self, 
                 client: anthropic.Anthropic,
                 token_monitor: TokenMonitorAgent,
                 chart_agent: ChartAnalysisAgent,
                 strategy_agent: Optional[StrategyAgent] = None):
        """Initialize the Trading Agent"""
        super().__init__(agent_type="trading")
        self.name = "Trading Agent"
        self.client = client
        self.token_monitor = token_monitor
        self.chart_agent = chart_agent
        self.strategy_agent = strategy_agent
        self.recommendations_df = pd.DataFrame(columns=['token', 'action', 'confidence'])
        self.last_update = datetime.now()
        
        # Initialize tokens list at startup (only once)
        cprint("\n🔄 Initializing monitored tokens list...", "white", "on_blue")
        self.tokens = self.token_monitor.run()
        self.last_token_update = datetime.now()
        
        # Initialize other attributes
        self.current_positions = {}
        self.trading_history = []
        self.last_analysis_time = None
        
        # Add cache for historical data
        self.data_cache = {}
        self.last_cache_update = {}
        self.circuit_breaker = CircuitBreaker()
        self.metrics = MetricsCollector()
        
        print("🤖 Trading Agent initialized!")

    def update_monitored_tokens(self):
        """Update the list of tokens to monitor"""
        try:
            # Use existing token_monitor instance instead of creating new one
            new_tokens = self.token_monitor.run()
            if new_tokens:
                self.tokens = new_tokens
                print(f"✅ Updated token list: {', '.join(self.tokens)}")
        except Exception as e:
            print(f"❌ Error updating token list: {str(e)}")

    def get_portfolio_state(self):
        """Get current portfolio state from Coinbase"""
        try:
            # Get USD balance
            usd_balance = cb.get_usd_balance()
            portfolio = {'USDC-USD': usd_balance}
            
            # Get other token balances
            for token in self.tokens:
                if token != 'USDC-USD':
                    balance = cb.get_token_balance_usd(token)
                    if balance > 0:
                        portfolio[token] = balance
            
            return portfolio
        except Exception as e:
            print(f"❌ Error getting portfolio state: {e}")
            return {}

    def analyze_market_data(
        self, 
        token: str, 
        market_data: pd.DataFrame, 
        strategy_signals: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Analyze market data using Claude"""
        try:
            # Skip analysis for excluded tokens
            if token in config.EXCLUDED_TOKENS:
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
                model=config.AI_MODEL,
                max_tokens=config.AI_MAX_TOKENS,
                temperature=config.AI_TEMPERATURE,
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
                    # Extract number including decimals from string
                    # like "Confidence: 75.5%" or "75.5% confidence"
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        confidence = float(numbers[0])
                        break
            
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
            max_position_size = config.usd_size * (config.MAX_POSITION_PERCENTAGE / 100)
            cprint(f"🎯 Maximum position size: ${max_position_size:.2f} ({config.MAX_POSITION_PERCENTAGE}% of ${config.usd_size:.2f})", "cyan")
            
            # Filter for BUY recommendations only
            buy_recommendations = self.recommendations_df[
                self.recommendations_df['action'] == "BUY"
            ]['token'].tolist()
            
            if not buy_recommendations:
                cprint("ℹ️ No BUY recommendations found - keeping funds in USDC", "yellow")
                return {"USDC-USD": config.usd_size}
            
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
                model=config.AI_MODEL,
                max_tokens=config.AI_MAX_TOKENS,
                temperature=config.AI_TEMPERATURE,
                messages=[{
                    "role": "user", 
                    "content": f"""You are Billy Bitcoin's Portfolio Allocation AI 🌙

Given:
- Total portfolio size: ${config.usd_size}
- Maximum position size: ${max_position_size} ({config.MAX_POSITION_PERCENTAGE}% of total)
- Minimum cash (USDC) buffer: {config.CASH_PERCENTAGE}%
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
            if total_allocated > config.usd_size:
                cprint(f"❌ Total allocation ${total_allocated:.2f} exceeds portfolio size ${config.usd_size:.2f}", "red")
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
        if not self.circuit_breaker.can_execute():
            print("⚠️ Circuit breaker active - skipping trades")
            return
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
                
                if token in config.EXCLUDED_TOKENS:
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
                                # Use ai_entry for larger orders which handles chunking
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
            
            if token in config.EXCLUDED_TOKENS:
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

    def get_cached_data(self, symbol, force_refresh=False):
        """Get data from cache or fetch new if needed"""
        current_time = datetime.now()
        cache_ttl = timedelta(minutes=config.SLEEP_BETWEEN_RUNS_MINUTES)  # Use config value

        # Check if we need to refresh the cache
        needs_refresh = (
            force_refresh or
            symbol not in self.data_cache or
            symbol not in self.last_cache_update or
            (current_time - self.last_cache_update[symbol]) > cache_ttl
        )

        if needs_refresh:
            data = cb.get_historical_data(symbol, granularity=3600, days_back=5)
            if not data.empty:
                self.data_cache[symbol] = data
                self.last_cache_update[symbol] = current_time
            return data
        
        return self.data_cache[symbol]

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

    async def run_trading_cycle(self):
        """Run a single trading cycle"""
        try:
            # Update tokens list only if interval has passed
            current_time = datetime.now()
            if (current_time - self.last_token_update).total_seconds() >= (config.SLEEP_BETWEEN_RUNS_MINUTES * 60):
                cprint("\n🔄 Updating monitored tokens list...", "white", "on_blue")
                self.tokens = self.token_monitor.run()
                self.last_token_update = current_time
            else:
                cprint("\n📋 Using existing tokens list...", "white", "on_blue")

            signal_data = []
            
            # Process tokens concurrently
            tasks = [self.analyze_token(token) for token in self.tokens]
            results = await asyncio.gather(*tasks)
            
            signal_data.extend([r for r in results if r])
            
            # Generate and display summary table
            if signal_data:
                # Create DataFrame for better formatting
                df = pd.DataFrame(signal_data)
                
                # Format strategy signals as string
                df['strategy_signals'] = df['strategy_signals'].apply(lambda x: '\n'.join(x) if x else 'No signals')
                
                # Print summary table
                cprint("\n📊 Signal Summary Table:", "white", "on_blue")
                print("\n" + df.to_string(index=False))
                
                # Save to CSV with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f'temp_data/signals_{timestamp}.csv'
                df.to_csv(csv_path, index=False)
                print(f"\n💾 Signal summary saved to: {csv_path}")

        except Exception as e:
            print(f"❌ Error in trading cycle: {str(e)}")
            traceback.print_exc()

    def execute_strategy_signals(self, signals):
        """Execute trades based on strategy signals"""
        for signal in signals:
            token = signal['token']
            direction = signal['direction']
            target_size = self.calculate_position_size(signal)
            
            if direction == 'BUY':
                cb.ai_entry(token, target_size)
            elif direction == 'SELL':
                cb.chunk_kill(token, config.max_usd_order_size, config.slippage)

    def handle_exit(self, token):
        """Handle position exit"""
        try:
            current_position = cb.get_token_balance_usd(token)
            if current_position > 0:
                cb.chunk_kill(token, config.max_usd_order_size, config.slippage)
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
            
            # Calculate maximum position size based on portfolio using config values
            max_position = config.usd_size * (config.MAX_POSITION_PERCENTAGE / 100)
            
            # Scale position size by confidence
            position_size = max_position * confidence
            
            # Ensure minimum position size using config value
            if position_size < config.MIN_TRADE_SIZE_USD:
                return 0
            
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculating position size: {str(e)}")
            return 0

    async def analyze_token(self, token):
        """Analyze a single token and return signal data"""
        try:
            # Get market data
            market_data = self.get_cached_data(token)
            if market_data.empty:
                print(f"⚠️ No market data available for {token}")
                return None
                
            # Run chart analysis first
            print(f"\n📊 Running chart analysis for {token}...")
            chart_analysis = self.chart_agent.analyze_chart(token, market_data)
            
            if not chart_analysis:
                print(f"⚠️ No chart analysis available for {token}")
                return None
            
            # Get strategy signals if strategy agent is enabled
            strategy_signals = None
            if self.strategy_agent:
                strategy_signals = await self.strategy_agent.get_signals(token)
            
            # Analyze market data
            analysis = self.analyze_market_data(token, market_data, strategy_signals)
            
            if analysis:
                return {
                    'token': token,
                    'chart_action': chart_analysis['action'],
                    'chart_confidence': chart_analysis['confidence'],
                    'strategy_signals': strategy_signals.get(token, []) if strategy_signals else []
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error analyzing {token}: {str(e)}")
            traceback.print_exc()  # Add this to see full error trace
            return None

    def extract_confidence(self, analysis_text: str) -> float:
        """Extract confidence percentage from analysis text"""
        try:
            # Look for confidence percentage in the text
            confidence = 0
            lines = analysis_text.split('\n')
            for line in lines:
                if 'confidence' in line.lower():
                    # Extract number including decimals from string
                    # like "Confidence: 75.5%" or "75.5% confidence"
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        confidence = float(numbers[0])
                        break
            
            # Ensure confidence is between 0 and 100
            confidence = max(0, min(100, confidence))
            return round(confidence, 2)  # Round to 2 decimal places
            
        except Exception as e:
            print(f"⚠️ Error extracting confidence: {str(e)}")
            return 50.0  # Return default confidence if extraction fails

async def main():
    """Main function to run the trading agent"""
    cprint("🌙 Billy Bitcoin AI Trading System Starting Up! 🚀", "white", "on_blue")
    
    # Initialize agents
    token_monitor = TokenMonitorAgent()
    chart_agent = ChartAnalysisAgent()
    strategy_agent = StrategyAgent() if config.ENABLE_STRATEGIES else None
    
    agent = TradingAgent(
        anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY")),
        token_monitor,
        chart_agent,
        strategy_agent
    )
    
    INTERVAL = config.SLEEP_BETWEEN_RUNS_MINUTES * 60  # Use config value
    
    while True:
        try:
            # Properly await the async function
            await agent.run_trading_cycle()
            
            next_run = datetime.now() + timedelta(minutes=config.SLEEP_BETWEEN_RUNS_MINUTES)
            cprint(f"\n⏳ AI Agent run complete. Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}", "white", "on_green")
            
            await asyncio.sleep(INTERVAL)
                
        except KeyboardInterrupt:
            cprint("\n👋 Billy Bitcoin AI Agent shutting down gracefully...", "white", "on_blue")
            break
        except Exception as e:
            cprint(f"\n❌ Error: {str(e)}", "white", "on_red")
            cprint("🔧 Billy Bitcoin suggests checking the logs and trying again!", "white", "on_blue")
            await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(main()) 