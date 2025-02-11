"""
🌙 Billy Bitcoin's Strategy Agent
Handles all strategy-based trading decisions
"""

from src.config import *
import json
from termcolor import cprint
import anthropic
import os
import importlib
import inspect
import time
from src import nice_funcs_cb as cb
import config

# 🎯 Strategy Evaluation Prompt
STRATEGY_EVAL_PROMPT = """
You are Billy Bitcoin's Strategy Validation Assistant 🌙

Analyze the following strategy signals and validate their recommendations:

Strategy Signals:
{strategy_signals}

Market Context:
{market_data}

Your task:
1. Evaluate each strategy signal's reasoning
2. Check if signals align with current market conditions
3. Look for confirmation/contradiction between different strategies
4. Consider risk factors

Respond in this format:
1. First line: EXECUTE or REJECT for each signal (e.g., "EXECUTE signal_1, REJECT signal_2")
2. Then explain your reasoning:
   - Signal analysis
   - Market alignment
   - Risk assessment
   - Confidence in each decision (0-100%)

Remember:
- Billy Bitcoin prioritizes risk management! 🛡️
- Multiple confirming signals increase confidence
- Contradicting signals require deeper analysis
- Better to reject a signal than risk a bad trade
"""

class StrategyAgent:
    def __init__(self):
        """Initialize the Strategy Agent"""
        self.enabled_strategies = []
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        
        if ENABLE_STRATEGIES:
            try:
                # Load strategies from custom directory
                strategy_dir = os.path.join('src', 'strategies', 'custom')
                
                if os.path.exists(strategy_dir):
                    for file in os.listdir(strategy_dir):
                        if file.endswith('_strategy.py'):
                            try:
                                # Convert file path to module path
                                module_path = f"src.strategies.custom.{file[:-3]}"
                                
                                # Import the module
                                module = importlib.import_module(module_path)
                                
                                # Find strategy class in module
                                for name, obj in inspect.getmembers(module):
                                    if (inspect.isclass(obj) and 
                                        name.endswith('Strategy') and 
                                        name != 'BaseStrategy'):
                                        self.enabled_strategies.append(obj())
                                        break
                                        
                            except Exception as e:
                                print(f"⚠️ Failed to load strategy from {file}: {e}")
                
                print(f"✅ Loaded {len(self.enabled_strategies)} strategies!")
                for strategy in self.enabled_strategies:
                    print(f"  • {strategy.name}")
                    
            except Exception as e:
                print(f"⚠️ Error loading strategies: {e}")
        else:
            print("🤖 Strategy Agent is disabled in config.py")
        
        print(f"🤖 Billy Bitcoin's Strategy Agent initialized with {len(self.enabled_strategies)} strategies!")

    def evaluate_signals(self, signals, market_data):
        """Have LLM evaluate strategy signals"""
        try:
            if not signals:
                return None
                
            # Format signals for prompt
            signals_str = json.dumps(signals, indent=2)
            
            message = self.client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,

                messages=[{
                    "role": "user",
                    "content": STRATEGY_EVAL_PROMPT.format(
                        strategy_signals=signals_str,
                        market_data=market_data
                    )
                }]
            )
            
            response = message.content
            if isinstance(response, list):
                response = response[0].text if hasattr(response[0], 'text') else str(response[0])
            
            # Parse response
            lines = response.split('\n')
            decisions = lines[0].strip().split(',')
            reasoning = '\n'.join(lines[1:])
            
            print("🤖 Strategy Evaluation:")
            print(f"Decisions: {decisions}")
            print(f"Reasoning: {reasoning}")
            
            return {
                'decisions': decisions,
                'reasoning': reasoning
            }
            
        except Exception as e:
            print(f"❌ Error evaluating signals: {e}")
            return None

    def _generate_signals(self, symbol, data):
        """Generate trading signals based on strategy rules"""
        try:
            signals = []
            
            # Run each enabled strategy
            for strategy in self.enabled_strategies:
                try:
                    signal = strategy.generate_signals(symbol, data)
                    if signal:
                        signals.append({
                            'strategy': strategy.name,
                            'token': symbol,
                            'signal': signal.get('strength', 0),
                            'direction': signal.get('direction', 'NOTHING'),
                            'metadata': signal.get('metadata', {})
                        })
                except Exception as e:
                    print(f"❌ Error in strategy {strategy.name}: {str(e)}")
                    continue
            
            return signals if signals else None
            
        except Exception as e:
            print(f"❌ Error generating signals: {str(e)}")
            return None

    def get_signals(self, symbol):
        """Get trading signals for a symbol"""
        try:
            # Use centralized data fetching from nice_funcs_cb
            data = cb.get_historical_data(symbol, granularity=3600, days_back=5)
            
            if data.empty:
                print(f"❌ No data available for {symbol}")
                return None
            
            # Calculate indicators and generate signals
            signals = self._generate_signals(symbol, data)
            return signals
            
        except Exception as e:
            print(f"❌ Error getting signals: {str(e)}")
            return None

    def combine_with_portfolio(self, signals, current_portfolio):
        """Combine strategy signals with current portfolio state"""
        try:
            final_allocations = current_portfolio.copy()
            
            for signal in signals:
                token = signal['token']
                strength = signal['signal']
                direction = signal['direction']
                
                if direction == 'BUY' and strength >= STRATEGY_MIN_CONFIDENCE:
                    print(f"🔵 Buy signal for {token} (strength: {strength})")
                    max_position = usd_size * (MAX_POSITION_PERCENTAGE / 100)
                    allocation = max_position * strength
                    final_allocations[token] = allocation
                elif direction == 'SELL' and strength >= STRATEGY_MIN_CONFIDENCE:
                    print(f"🔴 Sell signal for {token} (strength: {strength})")
                    final_allocations[token] = 0
            
            return final_allocations
            
        except Exception as e:
            print(f"❌ Error combining signals: {e}")
            return None 

    def execute_strategy_signals(self, approved_signals):
        """Execute trades based on approved strategy signals"""
        try:
            if not approved_signals:
                print("⚠️ No approved signals to execute")
                return

            print("\n🚀 Billy Bitcoin executing strategy signals...")
            print(f"📝 Received {len(approved_signals)} signals to execute")
            
            for signal in approved_signals:
                try:
                    print(f"\n🔍 Processing signal: {signal}")  # Debug output
                    
                    token = signal.get('token')
                    if not token:
                        print("❌ Missing token in signal")
                        print(f"Signal data: {signal}")
                        continue
                        
                    strength = signal.get('signal', 0)
                    direction = signal.get('direction', 'NOTHING')
                    
                    # Skip USDC and other excluded tokens
                    if token in EXCLUDED_TOKENS:
                        print(f"💵 Skipping {token} (excluded token)")
                        continue
                    
                    print(f"\n🎯 Processing signal for {token}...")
                    
                    # Calculate position size based on signal strength
                    max_position = usd_size * (MAX_POSITION_PERCENTAGE / 100)
                    target_size = max_position * strength
                    
                    # Enforce minimum trade size
                    if target_size < MIN_TRADE_SIZE_USD:
                        print(f"⚠️ Target size ${target_size:.2f} is below minimum trade size of ${MIN_TRADE_SIZE_USD}")
                        target_size = MIN_TRADE_SIZE_USD
                        print(f"📈 Adjusted target size to minimum: ${target_size:.2f}")
                    
                    # Get current position value using Coinbase function
                    current_position = cb.get_token_balance_usd(token)
                    
                    print(f"📊 Signal strength: {strength}")
                    print(f"🎯 Target position: ${target_size:.2f} USD")
                    print(f"📈 Current position: ${current_position:.2f} USD")
                    
                    if direction == 'BUY':
                        if current_position < target_size:
                            size_to_buy = target_size - current_position
                            if size_to_buy >= MIN_TRADE_SIZE_USD:
                                print(f"✨ Executing BUY for {token}")
                                self.execute_signal(signal)
                                print(f"✅ Entry complete for {token}")
                            else:
                                print(f"⚠️ Buy size ${size_to_buy:.2f} is below minimum trade size of ${MIN_TRADE_SIZE_USD}")
                        else:
                            print(f"⏸️ Position already at or above target size")
                            
                    elif direction == 'SELL':
                        if current_position > 0:
                            if current_position >= MIN_TRADE_SIZE_USD:
                                print(f"📉 Executing SELL for {token}")
                                self.execute_signal(signal)
                                print(f"✅ Exit complete for {token}")
                            else:
                                print(f"⚠️ Current position ${current_position:.2f} is below minimum trade size of ${MIN_TRADE_SIZE_USD}")
                        else:
                            print(f"⏸️ No position to sell")
                    
                    time.sleep(2)  # Small delay between trades
                    
                except Exception as e:
                    print(f"❌ Error processing signal: {str(e)}")
                    print(f"Signal data: {signal}")
                    continue
                
        except Exception as e:
            print(f"❌ Error executing strategy signals: {str(e)}")
            print("🔧 Billy Bitcoin suggests checking the logs and trying again!") 

    def calculate_position_size(self, signal_data):
        """Calculate position size based on signal strength and portfolio size"""
        try:
            # Get signal confidence from metadata or signal strength
            confidence = signal_data.get('metadata', {}).get('confidence', 0)
            if isinstance(confidence, str):
                confidence = float(confidence.strip('%'))
            
            # If no confidence in metadata, use signal strength
            if confidence == 0:
                confidence = float(signal_data.get('signal', 0))
            
            # Convert to decimal (0-1 range)
            confidence = confidence / 100 if confidence > 1 else confidence
            
            # Calculate maximum position size based on portfolio
            max_position = usd_size * (MAX_POSITION_PERCENTAGE / 100)
            
            # Scale position size by confidence
            position_size = max_position * confidence
            
            # Ensure minimum position size
            if position_size < MIN_TRADE_SIZE_USD:
                print(f"⚠️ Calculated position size ${position_size:.2f} below minimum ${MIN_TRADE_SIZE_USD}")
                return 0
            
            print(f"📊 Calculated position size: ${position_size:.2f} (confidence: {confidence:.2%})")
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculating position size: {str(e)}")
            return 0

    def execute_signal(self, signal_data):
        """Execute a trading signal"""
        try:
            token = signal_data['token']
            direction = signal_data['direction']
            
            # Get position size based on strategy confidence
            position_size = self.calculate_position_size(signal_data)
            
            if position_size == 0:
                print("⚠️ Position size too small - skipping trade")
                return False
            
            if direction == 'BUY':
                # Simple market order for smaller amounts
                if position_size <= 10000:
                    print(f"📈 Placing single market order for ${position_size:.2f}")
                    success = cb.market_buy(token, position_size)
                else:
                    print(f"📈 Placing chunked order for ${position_size:.2f}")
                    # Use ai_entry for larger orders which handles chunking
                    success = cb.ai_entry(token, position_size, max_usd_order_size=config.max_usd_order_size)
                    
            elif direction == 'SELL':
                success = cb.chunk_kill(token, max_usd_order_size=config.max_usd_order_size)
            
            return success
            
        except Exception as e:
            print(f"❌ Error processing signal: {str(e)}")
            print(f"Signal data: {signal_data}")
            return False 