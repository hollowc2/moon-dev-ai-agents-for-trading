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
TIMEFRAMES = ['5m']  # Coinbase supported timeframes: ['1m', '5m', '15m', '1h', '6h', '1d']
LOOKBACK_BARS = 300

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
            # Convert data to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Create chart filename
            chart_filename = f"{symbol}_{int(time.time())}.png"
            chart_path = os.path.join(self.charts_dir, chart_filename)
            
            # Generate chart using mplfinance
            mpf.plot(
                data,
                type='candle',
                volume=True,
                title=f'\n{symbol} Chart',
                style='charles',
                savefig=chart_path
            )
            
            return chart_path
            
        except Exception as e:
            print(f"❌ Error generating chart: {str(e)}")
            return None

    def _analyze_chart(self, symbol, timeframe, data):
        """Analyze chart data and return signals"""
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, unit='s')  # Convert Unix timestamps to datetime
            
            # Calculate indicators first
            data['SMA20'] = data['close'].rolling(window=20).mean()
            data['SMA50'] = data['close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            data['MACD'], data['Signal'], data['Hist'] = self._calculate_macd(data['close'])
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            sma20 = data['SMA20'].iloc[-1]
            sma50 = data['SMA50'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            signal = data['Signal'].iloc[-1]
            
            # Determine decimal precision based on price
            def get_precision(value):
                if value == 0:
                    return 8
                magnitude = abs(math.floor(math.log10(abs(value))))
                return magnitude + 8  # Add 8 decimal places after significant digits
            
            precision = get_precision(current_price)
            
            # Format analysis text with dynamic precision
            analysis_text = (
                f"📊 {symbol} {timeframe} Analysis:\n"
                f"- Price: ${current_price:.{precision}f}\n"
                f"- SMA20: ${sma20:.{precision}f}\n"
                f"- SMA50: ${sma50:.{precision}f}\n"
                f"- RSI: {rsi:.2f}\n"
                f"- MACD: {macd:.{precision}f}\n"
            )
            
            # Print market data with proper precision
            print("\n" + "╔" + "═" * 100 + "╗")
            print(f"║    🌙 Chart Data for {symbol} {timeframe} - Last 5 Candles" + " " * 45 + "║")
            print("╠" + "═" * 100 + "╣")
            print("║ Time                │ Open          │ High          │ Low           │ Close         │ Volume      ║")
            print("╟" + "─" * 100 + "╢")
            
            last_5 = data.tail(5)
            for idx, row in last_5.iterrows():
                # Format timestamp to string
                time_str = idx.strftime('%Y-%m-%d %H:%M')
                print(f"║ {time_str:<16} │ {row['open']:.{precision}f} │ {row['high']:.{precision}f} │ "
                      f"{row['low']:.{precision}f} │ {row['close']:.{precision}f} │ {row['volume']:10.2f} ║")
            
            # Determine signals
            action = "NOTHING"
            confidence = 0
            direction = "SIDEWAYS"
            
            # Simple strategy logic
            if current_price > sma20 and current_price > sma50 and rsi > 50 and macd > signal:
                action = "BUY"
                direction = "BULLISH"
                confidence = min(((rsi - 50) / 20) * 100, 100)  # Scale confidence
            elif current_price < sma20 and current_price < sma50 and rsi < 50 and macd < signal:
                action = "SELL"
                direction = "BEARISH"
                confidence = int(min(((50 - rsi) / 20) * 100, 100))  # Scale confidence
            
            return {
                'action': action,
                'direction': direction,
                'confidence': confidence,
                'analysis': analysis_text,
                'data': {
                    'price': current_price,
                    'sma20': sma20,
                    'sma50': sma50,
                    'rsi': rsi,
                    'macd': macd
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
            # If no data provided, fetch it
            if data is None:
                data = cb.get_historical_data(
                    symbol=symbol,
                    granularity=self._convert_timeframe_to_seconds('15m'),  # Default to 15m
                    days_back=int(LOOKBACK_BARS * self._get_timeframe_multiplier('15m'))
                )
            
            if data is None or data.empty:
                print(f"❌ No data available for {symbol}")
                return None

            # Ensure data is properly formatted
            if not isinstance(data.index, pd.DatetimeIndex):
                data.set_index('start', inplace=True)
                data.index = pd.to_datetime(data.index)

            # Generate chart
            chart_path = self._generate_chart(symbol, data)
            
            # Analyze chart
            analysis = self._analyze_chart(symbol, '15m', data)  # Default to 15m timeframe
            
            if analysis:
                # Format and print analysis
                print("\n" + "╔" + "═" * 50 + "╗")
                print(f"║    🌙 Billy Bitcoin's Chart Analysis - {symbol}    ║")
                print("╠" + "═" * 50 + "╣")
                print(f"║  Direction: {analysis['direction']:<41} ║")
                print(f"║  Action: {analysis['action']:<44} ║")
                print(f"║  Confidence: {analysis['confidence']}%{' ' * 37}║")
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
