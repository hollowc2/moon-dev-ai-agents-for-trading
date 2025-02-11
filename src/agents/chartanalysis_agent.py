"""
📊 Billy Bitcoin's Chart Analysis Agent
Built with love by Billy Bitcoin 🌙

Chuck the Chart Agent generates and analyzes trading charts using AI vision capabilities.
"""

import os
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import time
from dotenv import load_dotenv
import anthropic
import openai
from src import nice_funcs_cb as cb
from src.agents.base_agent import BaseAgent
import traceback
import base64
from io import BytesIO
import re
import pandas_ta as ta
import math

from src import config

# Register pandas_ta with pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# This line is crucial - it registers the ta namespace
pd.DataFrame.ta.cores = True  # Enable parallel processing for indicators

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
CHECK_INTERVAL_MINUTES = 10
TIMEFRAMES = ['15m', '1h']  # Coinbase supported timeframes: ['1m', '5m', '15m', '1h', '6h', '1d']
LOOKBACK_BARS = 300  # Changed from 320 to 300 to stay within Coinbase's limit

# Trading Pairs to Monitor - Update to use Coinbase format

SYMBOLS = ["BTC-USD"]  # Coinbase uses -USD format

# Chart Settings
CHART_STYLE = 'charles'  # mplfinance style
VOLUME_PANEL = True  # Show volume panel
INDICATORS = ['SMA20', 'SMA50', 'SMA200', 'RSI', 'MACD']  # Technical indicators to display

# AI Settings - Override config.py if set
from src import config

# Only set these if you want to override config.py settings
AI_MODEL = False  # Set to model name to override config.AI_MODEL
AI_TEMPERATURE = 0  # Set > 0 to override config.AI_TEMPERATURE 
AI_MAX_TOKENS = 0  # Set > 0 to override config.AI_MAX_TOKENS

# Voice settings
VOICE_MODEL = "tts-1"
VOICE_NAME = "shimmer" # Options: alloy, echo, fable, onyx, nova, shimmer
VOICE_SPEED = 1.0

# AI Analysis Prompt
CHART_ANALYSIS_PROMPT = """You must respond in exactly 3 lines:
Line 1: Only write BUY, SELL, or NOTHING
Line 2: One short reason why
Line 3: Only write "Confidence: X%" where X is 0-100

Analyze the chart data for {symbol} {timeframe}:

{chart_data}

Remember:
- Look for confluence between multiple indicators
- Volume should confirm price action
- Consider the timeframe context
- Look for confluence between multiple timeframes
"""

class ChartAnalysisAgent(BaseAgent):
    """Chuck the Chart Analysis Agent 📊"""
    
    def __init__(self):
        """Initialize the Chart Analysis Agent"""
        super().__init__(agent_type="chartanalysis")
        self.charts_dir = Path('temp_data/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = ["BTC-USD", "ETH-USD"]  # Default starting tokens
        self.last_update = datetime.now()
        
        print(f"📊 Chart Analysis Agent initialized with default tokens")
        
    def _convert_timeframe_to_seconds(self, timeframe):
        """Convert timeframe string to seconds for Coinbase API"""
        multiplier = int(timeframe[:-1])
        unit = timeframe[-1]
        
        if unit == 'm':
            return multiplier * 60
        elif unit == 'h':
            return multiplier * 3600
        elif unit == 'd':
            return multiplier * 86400
        return 3600  # Default to 1h if invalid

    def _get_timeframe_multiplier(self, timeframe):
        """Get multiplier for calculating days_back based on timeframe"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value / (24 * 60)
        elif unit == 'h':
            return value / 24
        elif unit == 'd':
            return value
        return 1  # Default multiplier

    def _generate_chart(self, symbol, data):
        """Generate chart for analysis"""
        try:
            # Ensure the charts directory exists
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert data to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Ensure data is properly formatted for mplfinance
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Calculate supply and demand zones
            sd_zones = cb.supply_demand_zones(symbol)
            if not sd_zones.empty:
                demand_zone = sd_zones['dz'].values
                supply_zone = sd_zones['sz'].values
            
            # Create chart filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"{symbol}_{timestamp}.png"
            chart_path = self.charts_dir / chart_filename
            
            print(f"📊 Generating chart: {chart_path}")
            
            # Configure chart style and indicators
            mc = mpf.make_marketcolors(
                up='green',
                down='red',
                edge='inherit',
                wick='inherit',
                volume='in',
                inherit=True
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='dotted',
                y_on_right=True
            )
            
            # Add technical indicators
            apds = []
            
            # Add SMA indicators
            sma20 = mpf.make_addplot(data['SMA20'], color='blue', width=0.7)
            sma50 = mpf.make_addplot(data['SMA50'], color='orange', width=0.7)
            apds.extend([sma20, sma50])
            
            # Add RSI if available
            if 'RSI' in data.columns:
                rsi = mpf.make_addplot(data['RSI'], panel=2, color='purple', width=0.7)
                apds.append(rsi)
            
            # Add supply and demand zones if available
            if not sd_zones.empty:
                # Create horizontal lines for demand zone
                demand_low = pd.Series(demand_zone[0], index=data.index)
                demand_high = pd.Series(demand_zone[1], index=data.index)
                apds.append(mpf.make_addplot(demand_low, color='green', width=1, linestyle='--'))
                apds.append(mpf.make_addplot(demand_high, color='green', width=1, linestyle='--'))
                
                # Create horizontal lines for supply zone
                supply_low = pd.Series(supply_zone[0], index=data.index)
                supply_high = pd.Series(supply_zone[1], index=data.index)
                apds.append(mpf.make_addplot(supply_low, color='red', width=1, linestyle='--'))
                apds.append(mpf.make_addplot(supply_high, color='red', width=1, linestyle='--'))
            
            # Generate and save the chart
            fig, axes = mpf.plot(
                data,
                type='candle',
                style=s,
                title=f'\n{symbol} Price Chart',
                volume=True,
                addplot=apds,
                panel_ratios=(6,2,2),  # Main chart, volume, RSI
                figsize=(15,10),
                savefig=chart_path,
                returnfig=True
            )
            
            plt.close(fig)  # Close the figure to free memory
            
            print(f"✅ Chart saved to: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            print(f"❌ Error generating chart: {str(e)}")
            traceback.print_exc()
            return None

    def _analyze_chart(self, symbol, timeframe, data):
        """Analyze chart data and return signals"""
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, unit='s')
            
            # Calculate indicators
            data['SMA20'] = data['close'].rolling(window=20).mean()
            data['SMA50'] = data['close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            data['MACD'], data['Signal'], data['Hist'] = self._calculate_macd(data['close'])
            
            # Get supply and demand zones
            sd_zones = cb.supply_demand_zones(symbol)
            
            # Calculate zone proximity factors
            def calculate_zone_proximity(price, zone_low, zone_high):
                """Calculate how close price is to a zone (0-1 scale)"""
                zone_middle = (zone_high + zone_low) / 2
                zone_range = zone_high - zone_low
                
                # Avoid division by zero
                if zone_range == 0:
                    return 0
                    
                # Calculate distance from price to zone middle
                distance = abs(price - zone_middle)
                
                # Normalize distance to 0-1 scale (closer = higher value)
                proximity = max(0, 1 - (distance / (zone_range * 2)))
                return proximity
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            sma20 = data['SMA20'].iloc[-1]
            sma50 = data['SMA50'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            signal = data['Signal'].iloc[-1]
            
            # Initialize base confidence from technical indicators
            tech_confidence = 0
            
            # Calculate technical indicator confidence (0-50)
            if current_price > sma20:
                tech_confidence += 10
            if current_price > sma50:
                tech_confidence += 10
            if rsi > 50:
                tech_confidence += 10
            if macd > signal:
                tech_confidence += 10
            if data['volume'].iloc[-1] > data['volume'].mean():
                tech_confidence += 10
            
            # Get zone values if available
            if not sd_zones.empty:
                demand_zone = sd_zones['dz'].values
                supply_zone = sd_zones['sz'].values
                demand_low, demand_high = demand_zone
                supply_low, supply_high = supply_zone
                
                # Calculate proximities
                demand_proximity = calculate_zone_proximity(current_price, demand_low, demand_high)
                supply_proximity = calculate_zone_proximity(current_price, supply_low, supply_high)
            else:
                demand_low = demand_high = supply_low = supply_high = None
                demand_proximity = supply_proximity = 0
            
            # Determine decimal precision based on price
            def get_precision(value):
                if value == 0:
                    return 8
                magnitude = abs(math.floor(math.log10(abs(value))))
                return magnitude + 8  # Add 8 decimal places after significant digits
            
            precision = get_precision(current_price)
            
            # Format analysis text with dynamic precision and zones
            analysis_text = (
                f"📊 {symbol} {timeframe} Analysis:\n"
                f"- Price: ${current_price:.{precision}f}\n"
                f"- SMA20: ${sma20:.{precision}f}\n"
                f"- SMA50: ${sma50:.{precision}f}\n"
                f"- RSI: {rsi:.2f}\n"
                f"- MACD: {macd:.{precision}f}\n"
            )
            
            if demand_low is not None:
                analysis_text += (
                    f"- Demand Zone: ${demand_low:.{precision}f} - ${demand_high:.{precision}f}\n"
                    f"- Supply Zone: ${supply_low:.{precision}f} - ${supply_high:.{precision}f}\n"
                )
            
            # Determine signals with enhanced zone-based confidence
            action = "NOTHING"
            confidence = 0
            direction = "SIDEWAYS"
            
            # Zone-based confidence (0-50)
            zone_confidence = 0
            
            if demand_low is not None and supply_low is not None:
                # Calculate price position relative to zones
                price_range = supply_high - demand_low
                if price_range > 0:
                    relative_position = (current_price - demand_low) / price_range
                    
                    # Bullish conditions
                    if current_price > sma20 and current_price > sma50:
                        action = "BUY"
                        direction = "BULLISH"
                        
                        # Higher confidence near demand zone
                        zone_confidence = 50 * (1 - relative_position)
                        
                    # Bearish conditions
                    elif current_price < sma20 and current_price < sma50:
                        action = "SELL"
                        direction = "BEARISH"
                        
                        # Higher confidence near supply zone
                        zone_confidence = 50 * relative_position
                        
                    else:
                        action = "NOTHING"
                        direction = "SIDEWAYS"
                        zone_confidence = 25  # Moderate confidence for no action
                
            # Calculate final confidence (technical + zone-based)
            confidence = tech_confidence + zone_confidence
            
            # Adjust confidence based on RSI extremes
            if rsi > 70 and action == "SELL":
                confidence *= 1.2  # 20% boost for overbought conditions
            elif rsi < 30 and action == "BUY":
                confidence *= 1.2  # 20% boost for oversold conditions
            
            # Ensure confidence stays within 0-100 range
            confidence = min(max(confidence, 0), 100)
            
            return {
                'action': action,
                'direction': direction,
                'confidence': round(confidence, 2),  # Round to 2 decimal places
                'analysis': analysis_text,
                'data': {
                    'price': current_price,
                    'sma20': sma20,
                    'sma50': sma50,
                    'rsi': rsi,
                    'macd': macd,
                    'volume': data['volume'].iloc[-1],
                    'demand_zone': [demand_low, demand_high] if demand_low is not None else None,
                    'supply_zone': [supply_low, supply_high] if supply_low is not None else None,
                    'zone_proximities': {
                        'demand': demand_proximity if demand_low is not None else 0,
                        'supply': supply_proximity if supply_low is not None else 0
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ Error in chart analysis: {str(e)}")
            traceback.print_exc()
            return None

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _format_announcement(self, symbol, timeframe, analysis):
        """Format analysis results into announcement"""
        try:
            # Get price with proper precision
            price = analysis['data']['price']
            precision = 8 if price < 0.01 else 6  # Use 8 decimals for small numbers
            
            announcement = (
                f"hi, Billy Bitcoin. In the Morning! Chart analysis for {symbol} "
                f"on the {timeframe} timeframe! The trend is {analysis['direction']}. "
                f"📊 {symbol} {timeframe} Analysis:\n"
                f"- Price: ${price:.{precision}f}\n"
                f"- RSI: {analysis['data']['rsi']:.2f}\n"
                f" AI suggests to {analysis['action']} with {analysis['confidence']}% confidence! "
            )
            return announcement
        except Exception as e:
            print(f"❌ Error formatting announcement: {str(e)}")
            return None
            
    def _announce(self, message):
        """Announce message using OpenAI TTS"""
        if not message:
            return
            
        try:
            print(f"\n📢 Announcing: {message}")
            
            # Generate speech
            response = self.openai_client.audio.speech.create(
                model=VOICE_MODEL,
                voice=VOICE_NAME,
                input=message,
                speed=VOICE_SPEED
            )
            
            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = self.audio_dir / f"chart_alert_{timestamp}.mp3"
            
            response.stream_to_file(str(audio_file))
            
            # Try to play with mpg123 first (most Linux distros)
            try:
                result = os.system(f"which mpg123 > /dev/null 2>&1")
                if result == 0:  # mpg123 is installed
                    os.system(f"mpg123 -q {audio_file}")  # -q for quiet mode
                else:
                    # Try ffplay as fallback (comes with ffmpeg)
                    result = os.system(f"which ffplay > /dev/null 2>&1")
                    if result == 0:
                        os.system(f"ffplay -nodisp -autoexit -loglevel quiet {audio_file}")
                    else:
                        print("⚠️ Please install either mpg123 or ffmpeg for audio playback:")
                        print("   sudo apt-get install mpg123    # For Ubuntu/Debian")
                        print("   sudo apt-get install ffmpeg    # Alternative option")
                        print(f"💾 Audio file saved to: {audio_file}")
            except Exception as e:
                print(f"⚠️ Could not play audio: {str(e)}")
                print(f"💾 Audio file saved to: {audio_file}")
            
        except Exception as e:
            print(f"❌ Error in announcement: {str(e)}")
            
    def analyze_chart(self, symbol, data=None):
        """Public method for chart analysis - wrapper for _analyze_chart"""
        try:
            print(f"\n📊 Starting chart analysis for {symbol}...")
            
            # If no data provided, fetch it
            if data is None:
                print(f"🔄 Fetching historical data for {symbol}...")
                data = cb.get_historical_data(
                    symbol=symbol,
                    granularity=self._convert_timeframe_to_seconds('15m'),
                    days_back=int(LOOKBACK_BARS * self._get_timeframe_multiplier('15m'))
                )
            
            if data is None or data.empty:
                print(f"❌ No data available for {symbol}")
                return None

            # Ensure data is properly formatted
            if not isinstance(data.index, pd.DatetimeIndex):
                print("🔄 Converting timestamps...")
                data.set_index('start', inplace=True)
                data.index = pd.to_datetime(data.index)

            # Calculate indicators before generating chart
            print("📈 Calculating technical indicators...")
            data['SMA20'] = data['close'].rolling(window=20).mean()
            data['SMA50'] = data['close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            
            # Generate chart
            print(f"🎨 Generating chart for {symbol}...")
            chart_path = self._generate_chart(symbol, data)
            if chart_path:
                print(f"✅ Chart generated: {chart_path}")
            else:
                print("⚠️ Failed to generate chart")
            
            # Analyze chart
            print("🔍 Analyzing chart patterns...")
            analysis = self._analyze_chart(symbol, '15m', data)
            
            if analysis:
                print("\n" + "╔" + "═" * 50 + "╗")
                print(f"║    🌙 Billy Bitcoin's Chart Analysis - {symbol}   ║")
                print("╠" + "═" * 50 + "╣")
                print(f"║  Direction: {analysis['direction']:<36} ║")
                print(f"║  Action: {analysis['action']:<39} ║")
                print(f"║  Confidence: {analysis['confidence']:.2f}%{' ' * 29}║")
                print("╚" + "═" * 50 + "╝")
            
            return analysis

        except Exception as e:
            print(f"❌ Error in chart analysis for {symbol}: {str(e)}")
            traceback.print_exc()
            return None

    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle for all symbols"""
        try:
            results = {}
            for symbol in self.symbols:
                print(f"\n📊 Fetching historical data for {symbol}...")
                analysis = self.analyze_chart(symbol)
                if analysis:
                    results[symbol] = analysis
            return results
        except Exception as e:
            print(f"❌ Error in monitoring cycle: {str(e)}")
            traceback.print_exc()
            return None

    def run(self):
        """Run the chart analysis monitor continuously"""
        try:
            print("\n🚀 Starting chart analysis monitoring...")
            
            while True:
                try:
                    # Update token list every hour
                    if (datetime.now() - self.last_update).total_seconds() > 3600:  # 1 hour
                        self.symbols = config.get_top_trading_pairs()  # Get fresh list
                        self.last_update = datetime.now()
                        print(f"\n🔄 Updated monitoring list: {', '.join(self.symbols)}")
                    
                    results = self.run_monitoring_cycle()
                    print(f"\n💤 Sleeping for {CHECK_INTERVAL_MINUTES} minutes...")
                    time.sleep(CHECK_INTERVAL_MINUTES * 60)
                    
                except KeyboardInterrupt:
                    print("\n👋 Chart Analysis Agent shutting down gracefully...")
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {str(e)}")
                    time.sleep(60)  # Sleep for a minute before retrying
                    
        except Exception as e:
            print(f"❌ Fatal error in run method: {str(e)}")
            raise

    def update_symbols(self, new_symbols):
        """Update the list of symbols to monitor"""
        if new_symbols:
            self.symbols = new_symbols
            print(f"🔄 Updated symbols list: {', '.join(self.symbols)}")


if __name__ == "__main__":
    # Create and run the agent
    print("\n🌙 Billy Bitcoin's Chart Analysis Agent Starting Up...")
    print("👋 Hey! I'm Chuck, your friendly chart analysis agent! 📊")
    print(f"🎯 Monitoring {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}")
    agent = ChartAnalysisAgent()
    
    # Run the continuous monitoring cycle
    agent.run()
