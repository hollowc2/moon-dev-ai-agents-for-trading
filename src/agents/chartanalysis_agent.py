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
    
    def __init__(self, symbols=None):
        """Initialize Chuck the Chart Agent"""
        super().__init__('chartanalysis')
        
        # Set up directories
        self.charts_dir = PROJECT_ROOT / "src" / "data" / "charts"
        self.audio_dir = PROJECT_ROOT / "src" / "audio"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Use passed symbols or default to BTC-USD
        self.symbols = symbols if symbols else ["BTC-USD"]
        
        # Load environment variables
        load_dotenv()
        
        # Initialize API clients
        openai_key = os.getenv("OPENAI_KEY")
        anthropic_key = os.getenv("ANTHROPIC_KEY")
        
        if not openai_key or not anthropic_key:
            raise ValueError("🚨 API keys not found in environment variables!")
            
        self.openai_client = openai.OpenAI(api_key=openai_key)  # For TTS only
        self.client = anthropic.Anthropic(api_key=anthropic_key)
        
        # Set AI parameters - use config values unless overridden
        self.ai_model = AI_MODEL if AI_MODEL else config.AI_MODEL
        self.ai_temperature = AI_TEMPERATURE if AI_TEMPERATURE > 0 else config.AI_TEMPERATURE
        self.ai_max_tokens = AI_MAX_TOKENS if AI_MAX_TOKENS > 0 else config.AI_MAX_TOKENS
        
        print("📊 Chuck the Chart Agent initialized!")
        print(f"🤖 Using AI Model: {self.ai_model}")
        if AI_MODEL or AI_TEMPERATURE > 0 or AI_MAX_TOKENS > 0:
            print("⚠️ Note: Using some override settings instead of config.py defaults")
        print(f"🎯 Analyzing {len(TIMEFRAMES)} timeframes: {', '.join(TIMEFRAMES)}")
        print(f"📈 Using indicators: {', '.join(INDICATORS)}")
        print(f"🪙 Monitoring {len(self.symbols)} symbols: {', '.join(self.symbols)}")
        
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

    def _generate_chart(self, symbol, timeframe, data):
        """Generate a chart using mplfinance"""
        try:
            # Prepare data
            df = data.copy()
            df.index = pd.to_datetime(df.index)
            
            # Check if data is valid
            if df.empty:
                print("❌ No data available for chart generation")
                return None
                
            # Calculate indicators directly using pandas_ta on the DataFrame
            # Use min_periods parameter to start calculating as soon as possible
            if 'SMA20' in INDICATORS:
                df['SMA20'] = df['close'].rolling(window=20, min_periods=1).mean()
            if 'SMA50' in INDICATORS:
                df['SMA50'] = df['close'].rolling(window=50, min_periods=1).mean()
            if 'SMA200' in INDICATORS:
                df['SMA200'] = df['close'].rolling(window=200, min_periods=1).mean()
            if 'RSI' in INDICATORS:
                df['RSI'] = ta.rsi(df['close'])
            if 'MACD' in INDICATORS:
                macd = ta.macd(df['close'])
                if isinstance(macd, pd.DataFrame):
                    df = pd.concat([df, macd], axis=1)
            
            # Get supply and demand zones
            sd_df = cb.supply_demand_zones(symbol, timeframe=self._convert_timeframe_to_seconds(timeframe))
            demand_zone = sd_df['dz'].values
            supply_zone = sd_df['sz'].values

            # Create addplot for indicators with better visibility
            ap = []
            colors = ['blue', 'orange', 'purple']
            linewidths = [1.5, 1.5, 1.5]  # Increased line width for better visibility
            for i, sma in enumerate(['SMA20', 'SMA50', 'SMA200']):
                if sma in INDICATORS and sma in df.columns:
                    if not df[sma].isna().all():
                        ap.append(mpf.make_addplot(df[sma], 
                                                 color=colors[i],
                                                 width=linewidths[i],
                                                 secondary_y=False))
            
            # Add horizontal lines for supply and demand zones
            for dz in demand_zone:
                ap.append(mpf.make_addplot([dz] * len(df), color='green', linestyle='--'))
            for sz in supply_zone:
                ap.append(mpf.make_addplot([sz] * len(df), color='red', linestyle='--'))
            
            # Save chart with improved styling
            filename = f"{symbol}_{timeframe}_{int(time.time())}.png"
            chart_path = self.charts_dir / filename
            
            # Create the chart with better styling
            mpf.plot(df,
                    type='candle',
                    style=CHART_STYLE,
                    volume=VOLUME_PANEL,
                    addplot=ap if ap else None,
                    title=f"\n{symbol} {timeframe} Chart Analysis by Billy Bitcoin 🌙",
                    figsize=(12, 8),  # Larger figure size
                    panel_ratios=(3, 1),  # Better ratio between price and volume
                    datetime_format='%m-%d %H:%M',  # Cleaner date format
                    xrotation=45,  # Angled dates for better readability
                    savefig=chart_path)
            
            # Print nicely formatted data table with wider columns
            print("\n" + "╔" + "═" * 100 + "╗")
            print(f"║    🌙 Chart Data for {symbol} {timeframe} - Last 5 Candles" + " " * 45 + "║")
            print("╠" + "═" * 100 + "╣")
            print("║ Time                │ Open          │ High          │ Low           │ Close         │ Volume      ║")
            print("╟" + "─" * 100 + "╢")
            
            # Print last 5 candles with proper formatting and wider columns
            last_5 = df.tail(5)
            for idx, row in last_5.iterrows():
                time_str = idx.strftime('%Y-%m-%d %H:%M')
                print(f"║ {time_str:<16} │ {row['open']:12.2f} │ {row['high']:12.2f} │ {row['low']:12.2f} │ {row['close']:12.2f} │ {row['volume']:10.0f} ║")
            
            print("║" + " " * 100 + "║")
            print("║ Technical Indicators:" + " " * 79 + "║")
            print(f"║ SMA20: {df['SMA20'].iloc[-1]:.2f}" + " " * 85 + "║")
            print(f"║ SMA50: {df['SMA50'].iloc[-1]:.2f}" + " " * 85 + "║")
            print(f"║ SMA200: {df['SMA200'].iloc[-1] if not pd.isna(df['SMA200'].iloc[-1]) else 'Not enough data'}" + " " * 75 + "║")
            print(f"║ 24h High: {df['high'].max():.2f}" + " " * 83 + "║")
            print(f"║ 24h Low: {df['low'].min():.2f}" + " " * 84 + "║")
            print(f"║ Volume Trend: {'Increasing' if df['volume'].iloc[-1] > df['volume'].mean() else 'Decreasing'}" + " " * 75 + "║")
            print("╚" + "═" * 100 + "╝")
            
            return chart_path
            
        except Exception as e:
            print(f"❌ Error generating chart: {str(e)}")
            traceback.print_exc()
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
                confidence = min(((50 - rsi) / 20) * 100, 100)  # Scale confidence
            
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
            
    def analyze_symbol(self, symbol, timeframe):
        """Analyze a single symbol on a specific timeframe"""
        try:
            # Try getting data from Coinbase first
            data = None
            try:
                data = cb.get_historical_data(
                    symbol=symbol,
                    granularity=self._convert_timeframe_to_seconds(timeframe),
                    days_back=int(LOOKBACK_BARS * self._get_timeframe_multiplier(timeframe))
                )
                
                # Debug the raw data
                print("\nRaw data from Coinbase:")
                print(data.head())
                
            except Exception as e:
                print(f"📝 Coinbase data fetch failed: {str(e)}")

            if data is None or data.empty:
                print(f"❌ No data available for {symbol} {timeframe} from Coinbase")
                return

            # Set the 'start' column as the index
            data.set_index('start', inplace=True)
            
            # Calculate additional indicators
            if 'SMA20' not in data.columns:
                data['SMA20'] = data['close'].rolling(window=20).mean()
            if 'SMA50' not in data.columns:
                data['SMA50'] = data['close'].rolling(window=50).mean()
            if 'SMA200' not in data.columns:
                data['SMA200'] = data['close'].rolling(window=200).mean()
            
            # Generate and save chart first
            print(f"\n📊 Generating chart for {symbol} {timeframe}...")
            chart_path = self._generate_chart(symbol, timeframe, data)
            if chart_path:
                print(f"📈 Chart saved to: {chart_path}")
            
            # Print market data once
            print("\n" + "╔" + "═" * 100 + "╗")
            print(f"║    🌙 Chart Data for {symbol} {timeframe} - Last 5 Candles" + " " * 45 + "║")
            print("╠" + "═" * 100 + "╣")
            print("║ Time                │ Open          │ High          │ Low           │ Close         │ Volume      ║")
            print("╟" + "─" * 100 + "╢")
            
            last_5 = data.tail(5)
            for idx, row in last_5.iterrows():
                # Ensure the index is a datetime
                time_str = idx.strftime('%Y-%m-%d %H:%M')
                
                print(f"║ {time_str:<16} │ {row['open']:12.8f} │ {row['high']:12.8f} │ {row['low']:12.8f} │ {row['close']:12.8f} │ {row['volume']:10.0f} ║")
            
            print("║" + " " * 100 + "║")
            print("║ Technical Indicators:" + " " * 79 + "║")
            print(f"║ SMA20: {data['SMA20'].iloc[-1]:.8f}" + " " * 85 + "║")
            print(f"║ SMA50: {data['SMA50'].iloc[-1]:.8f}" + " " * 85 + "║")
            print(f"║ SMA200: {data['SMA200'].iloc[-1]:.8f}" + " " * 84 + "║")
            print(f"║ 24h High: {data['high'].max():.8f}" + " " * 83 + "║")
            print(f"║ 24h Low: {data['low'].min():.8f}" + " " * 84 + "║")
            print(f"║ Volume Trend: {'Increasing' if data['volume'].iloc[-1] > data['volume'].mean() else 'Decreasing'}" + " " * 75 + "║")
            print("╚" + "═" * 100 + "╝")
                
            # Analyze with AI
            print(f"\n🔍 Analyzing {symbol} {timeframe}...")
            analysis = self._analyze_chart(symbol, timeframe, data)
            
            if analysis and all(k in analysis for k in ['direction', 'analysis', 'action', 'confidence']):
                # Format and announce
                message = self._format_announcement(symbol, timeframe, analysis)
                if message:
                    self._announce(message)
                    
                # Print analysis in a nice box
                print("\n" + "╔" + "═" * 50 + "╗")
                print(f"║    🌙 Billy Bitcoin's Chart Analysis - {symbol} {timeframe}   ║")
                print("╠" + "═" * 50 + "╣")
                print(f"║  Direction: {analysis['direction']:<41} ║")
                print(f"║  Action: {analysis['action']:<44} ║")
                print(f"║  Confidence: {analysis['confidence']}%{' ' * 37}║")
                print("╟" + "─" * 50 + "╢")
                print(f"║  Analysis: {analysis['analysis']:<41} ║")
                print("╚" + "═" * 50 + "╝")
            else:
                print("❌ Invalid analysis result")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol} {timeframe}: {str(e)}")
            traceback.print_exc()
            
    def _cleanup_old_charts(self):
        """Remove all existing charts from the charts directory"""
        try:
            for chart in self.charts_dir.glob("*.png"):
                chart.unlink()
            print("🧹 Cleaned up old charts")
        except Exception as e:
            print(f"⚠️ Error cleaning up charts: {str(e)}")

    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        try:
            # Clean up old charts before starting new cycle
            self._cleanup_old_charts()
            
            results = {}  # Store results for each symbol/timeframe
            
            for symbol in self.symbols:
                for timeframe in TIMEFRAMES:
                    # Get analysis result
                    data = None
                    try:
                        data = cb.get_historical_data(
                            symbol=symbol,
                            granularity=self._convert_timeframe_to_seconds(timeframe),
                            days_back=int(LOOKBACK_BARS * self._get_timeframe_multiplier(timeframe))
                        )
                    except Exception as e:
                        print(f"Error getting data: {str(e)}")
                        continue

                    if data is not None:
                        analysis = self._analyze_chart(symbol, timeframe, data)
                        if analysis:
                            # Store results
                            if symbol not in results:
                                results[symbol] = {}
                            results[symbol][timeframe] = analysis
                            
                            # Format and announce if needed
                            message = self._format_announcement(symbol, timeframe, analysis)
                            if message:
                                self._announce(message)
                
                    time.sleep(2)  # Small delay between analyses
                    
            return results  # Return the analysis results
                    
        except Exception as e:
            print(f"❌ Error in monitoring cycle: {str(e)}")
            return {}  # Return empty dict on error

    def run(self):
        """Run the chart analysis monitor continuously"""
        print("\n🚀 Starting chart analysis monitoring...")
        
        while True:
            try:
                results = self.run_monitoring_cycle()
                print(f"\n💤 Sleeping for {CHECK_INTERVAL_MINUTES} minutes...")
                time.sleep(CHECK_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                print("\n👋 Chuck the Chart Agent shutting down gracefully...")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {str(e)}")
                time.sleep(60)  # Sleep for a minute before retrying

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
