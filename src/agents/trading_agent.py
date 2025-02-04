"""
🌙 Moon Dev's LLM Trading Agent
Handles all LLM-based trading decisions
"""

# Keep only these prompts
TRADING_PROMPT = """
You are Moon Dev's AI Trading Assistant 🌙

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
- Moon Dev always prioritizes risk management! 🛡️
- Never trade USDC or SOL directly
- Consider both technical and strategy signals
"""

ALLOCATION_PROMPT = """
You are Moon Dev's Portfolio Allocation Assistant 🌙

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
from src import nice_funcs as n
from src import nice_funcs_hl as hl
from src import nice_funcs_cb as cb
#Sfrom src.data.ohlcv_collector import collect_all_tokens
from nice_funcs import get_data

# Load environment variables
load_dotenv()

class TradingAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        self.recommendations_df = pd.DataFrame(columns=['token', 'action', 'confidence', 'reasoning'])
        print("🤖 Moon Dev's LLM Trading Agent initialized!")

    def analyze_market_data(self, token, market_data):
        """Analyze market data using Claude"""
        try:
            # Skip analysis for excluded tokens
            if token in EXCLUDED_TOKENS:
                print(f"⚠️ Skipping analysis for excluded token: {token}")
                return None
            
            # Prepare strategy context
            strategy_context = ""
            if 'strategy_signals' in market_data:
                strategy_context = f"""
Strategy Signals Available:
{json.dumps(market_data['strategy_signals'], indent=2)}
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
            
            print(f"🎯 Moon Dev's AI Analysis Complete for {token[:4]}!")
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
            
            # Get available products from Coinbase
            available_products = cb.get_available_products()
            if not available_products:
                cprint("❌ Could not get available products from Coinbase", "red")
                return None
            
            # Filter out excluded tokens and stablecoins
            trading_pairs = [
                pair for pair in available_products 
                if pair.endswith('-USD') and 
                not any(stable in pair for stable in ['USDC', 'USDT', 'DAI', 'UST'])
            ]
            
            # Get allocation from AI
            message = self.client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,
                messages=[{
                    "role": "user", 
                    "content": f"""You are Moon Dev's Portfolio Allocation AI 🌙

Given:
- Total portfolio size: ${usd_size}
- Maximum position size: ${max_position_size} ({MAX_POSITION_PERCENTAGE}% of total)
- Minimum cash (USDC) buffer: {CASH_PERCENTAGE}%
- Available trading pairs: {trading_pairs}
- USDC-USD pair for cash allocation

Provide a portfolio allocation that:
1. Never exceeds max position size per token
2. Maintains minimum cash buffer in USDC-USD
3. Returns allocation as a JSON object with trading pairs as keys and USD amounts as values
4. Uses 'USDC-USD' for cash allocation

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
            print("\n🚀 Moon Dev executing portfolio allocations...")
            
            for token, amount in allocation_dict.items():
                if token in EXCLUDED_TOKENS:
                    print(f"💵 Keeping ${amount:.2f} in {token}")
                    continue
                    
                print(f"\n🎯 Processing allocation for {token}...")
                
                try:
                    # Debug which function we're using
                    print("📊 Attempting to get position from Coinbase...")
                    current_position = None
                    
                    try:
                        current_position = cb.get_token_balance_usd(token)
                        print(f"✅ Successfully got position from Coinbase: ${current_position:.2f}")
                    except Exception as cb_error:
                        print(f"⚠️ Coinbase position check failed: {str(cb_error)}")
                        continue  # Skip to next token if Coinbase fails
                    
                    if current_position is None:
                        print("❌ Could not get current position from Coinbase")
                        continue
                        
                    target_allocation = amount
                    print(f"🎯 Target allocation: ${target_allocation:.2f} USD")
                    print(f"📊 Current position: ${current_position:.2f} USD")
                    
                    if current_position < target_allocation:
                        print(f"✨ Executing entry for {token}")
                        try:
                            success = cb.ai_entry(token, amount, max_usd_order_size=50000)
                            if success:
                                print(f"✅ Entry successful using Coinbase")
                            else:
                                print(f"❌ Entry failed for {token}")
                        except Exception as entry_error:
                            print(f"❌ Coinbase entry failed: {str(entry_error)}")
                    else:
                        print(f"⏸️ Position already at target size for {token}")
                    
                except Exception as e:
                    print(f"❌ Error executing entry for {token}: {str(e)}")
                    print(f"Full error details: {type(e).__name__}")
                    import traceback
                    print(traceback.format_exc())
                
                time.sleep(2)  # Small delay between entries
                
        except Exception as e:
            print(f"❌ Error executing allocations: {str(e)}")
            print("🔧 Moon Dev suggests checking the logs and trying again!")

    def handle_exits(self):
        """Check and exit positions based on SELL or NOTHING recommendations"""
        cprint("\n🔄 Checking for positions to exit...", "white", "on_blue")
        
        for _, row in self.recommendations_df.iterrows():
            token = row['token']
            
            # Skip excluded tokens (USDC and SOL)
            if token in EXCLUDED_TOKENS:
                continue
                
            action = row['action']
            
            # Check if we have a position using Coinbase first
            try:
                current_position = cb.get_token_balance_usd(token)
            except:
                try:
                    current_position = n.get_token_balance_usd(token)  # Fallback to nice_funcs
                except:
                    current_position = 0
            
            if current_position > 0 and action in ["SELL", "NOTHING"]:
                cprint(f"\n🚫 AI Agent recommends {action} for {token}", "white", "on_yellow")
                cprint(f"💰 Current position: ${current_position:.2f}", "white", "on_blue")
                try:
                    cprint(f"📉 Closing position with chunk_kill...", "white", "on_cyan")
                    # Try Coinbase chunk_kill first
                    try:
                        cb.chunk_kill(token, max_usd_order_size=50000)
                    except:
                        n.chunk_kill(token, max_usd_order_size, slippage)  # Fallback to nice_funcs
                    cprint(f"✅ Successfully closed position", "white", "on_green")
                except Exception as e:
                    cprint(f"❌ Error closing position: {str(e)}", "white", "on_red")
            elif current_position > 0:
                cprint(f"✨ Keeping position for {token} (${current_position:.2f}) - AI recommends {action}", "white", "on_blue")

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

    def get_data_from_sources(self, token, timeframe='1h', bars=24):
        """Helper function to get data from available sources, prioritizing Coinbase"""
        data = None
        
        # Try Coinbase first using get_historical_data
        try:
            # Convert timeframe parameter to granularity for Coinbase
            granularity = timeframe.replace('m', '').replace('h', '* 60')
            granularity = eval(granularity)  # Safely convert string to number
            
            data = cb.get_historical_data(
                symbol=token,
                granularity=granularity,  # Coinbase uses seconds
                days_back=bars  # Coinbase uses 'limit' instead of 'bars'
            )
            if data is not None:
                print(f"✅ Got data from Coinbase for {token}")
                return data
        except Exception as e:
            print(f"📝 Coinbase data fetch failed: {str(e)}")

        # Try other sources if Coinbase fails
        for module in [n, hl]:  # Add any new nice_funcs modules here
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
        for module in [n, hl]:  # Add any new nice_funcs modules here
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

    def run_trading_cycle(self, strategy_signals=None):
        """Run one complete trading cycle"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cprint(f"\n⏰ AI Agent Run Starting at {current_time}", "white", "on_green")
            
            # Get available products
            cprint("📊 Getting available products...", "white", "on_blue")
            products = self.get_products_from_sources()
            
            if not products:
                raise Exception("Could not get products from any source")
            
            # Collect market data
            market_data = {}
            for token in products:
                if token in EXCLUDED_TOKENS:
                    continue
                    
                data = self.get_data_from_sources(token)
                if data is not None:
                    market_data[token] = data
            
            if not market_data:
                raise Exception("Could not get market data from any source")
            
            # Analyze each token's data
            for token, data in market_data.items():
                cprint(f"\n🤖 AI Agent Analyzing Token: {token}", "white", "on_green")
                
                # Include strategy signals in analysis if available
                if strategy_signals and token in strategy_signals:
                    cprint(f"📊 Including {len(strategy_signals[token])} strategy signals in analysis", "cyan")
                    data['strategy_signals'] = strategy_signals[token]
                
                analysis = self.analyze_market_data(token, data)
                print(f"\n📈 Analysis for contract: {token}")
                print(analysis)
                print("\n" + "="*50 + "\n")
            
            # Show recommendations summary
            cprint("\n📊 Moon Dev's Trading Recommendations:", "white", "on_blue")
            summary_df = self.recommendations_df[['token', 'action', 'confidence']].copy()
            print(summary_df.to_string(index=False))
            
            # Handle exits first
            self.handle_exits()
            
            # Then proceed with new allocations
            cprint("\n💰 Calculating optimal portfolio allocation...", "white", "on_blue")
            allocation = self.allocate_portfolio()
            
            if allocation:
                cprint("\n💼 Moon Dev's Portfolio Allocation:", "white", "on_blue")
                print(json.dumps(allocation, indent=4))
                
                cprint("\n🎯 Executing allocations...", "white", "on_blue")
                self.execute_allocations(allocation)
                cprint("\n✨ All allocations executed!", "white", "on_blue")
            else:
                cprint("\n⚠️ No allocations to execute!", "white", "on_yellow")
            
            # Clean up temp data
            cprint("\n🧹 Cleaning up temporary data...", "white", "on_blue")
            try:
                for file in os.listdir('temp_data'):
                    if file.endswith('_latest.csv'):
                        os.remove(os.path.join('temp_data', file))
                cprint("✨ Temp data cleaned successfully!", "white", "on_green")
            except Exception as e:
                cprint(f"⚠️ Error cleaning temp data: {str(e)}", "white", "on_yellow")
            
        except Exception as e:
            cprint(f"\n❌ Error in trading cycle: {str(e)}", "white", "on_red")
            cprint("🔧 Moon Dev suggests checking the logs and trying again!", "white", "on_blue")

def main():
    """Main function to run the trading agent every 15 minutes"""
    cprint("🌙 Moon Dev AI Trading System Starting Up! 🚀", "white", "on_blue")
    
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
            cprint("\n👋 Moon Dev AI Agent shutting down gracefully...", "white", "on_blue")
            break
        except Exception as e:
            cprint(f"\n❌ Error: {str(e)}", "white", "on_red")
            cprint("🔧 Moon Dev suggests checking the logs and trying again!", "white", "on_blue")
            # Still sleep and continue on error
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main() 