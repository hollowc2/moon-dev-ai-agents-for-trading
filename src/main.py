"""
🌙 Billy Bitcoin's AI Trading System
Main entry point for running trading agents
"""

import os
import sys
from termcolor import cprint
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
from config import *
from anthropic import Anthropic
import asyncio  # Add this import at the top


# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import agents
from src.agents.trading_agent import TradingAgent
from src.agents.risk_agent import RiskAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.copybot_agent import CopyBotAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.chartanalysis_agent import ChartAnalysisAgent
from src.agents.token_monitor_agent import TokenMonitorAgent

# Load environment variables
load_dotenv()

# Agent Configuration
ACTIVE_AGENTS = {
    'risk': False,      # Risk management agent
    'trading': True,   # LLM trading agent
    'strategy': False,  # Strategy-based trading agent
    'copybot': False,   # CopyBot agent
    'sentiment': False, # Run sentiment_agent.py directly instead
    #'chartanalysis': True, # Chart Analysis Agent
    # whale_agent is run from whale_agent.py
    # Add more agents here as we build them:
    # 'portfolio': False,  # Future portfolio optimization agent
}

async def run_agents():
    """Run all active agents in sequence"""
    try:
        # Initialize active agents
        if ACTIVE_AGENTS['trading']:
            # Create required components for TradingAgent
            anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
            token_monitor = TokenMonitorAgent()
            
            # Initialize chart agent with proper configuration
            print("\n🎯 Initializing Chart Analysis Agent...")
            chart_agent = ChartAnalysisAgent()
            
            # Create trading agent with all components
            trading_agent = TradingAgent(
                client=anthropic_client,
                token_monitor=token_monitor,
                chart_agent=chart_agent,
                strategy_agent=None  # Add strategy agent if needed
            )
            print("✅ Trading Agent initialized with Chart Analysis capabilities")
        else:
            trading_agent = None
            
        risk_agent = RiskAgent() if ACTIVE_AGENTS['risk'] else None
        strategy_agent = StrategyAgent() if ACTIVE_AGENTS['strategy'] else None
        copybot_agent = CopyBotAgent() if ACTIVE_AGENTS['copybot'] else None
        sentiment_agent = SentimentAgent() if ACTIVE_AGENTS['sentiment'] else None
        
        while True:
            try:
                # Run Trading Analysis (which includes Chart Analysis)
                if trading_agent:
                    cprint("\n🤖 Running Trading Analysis...", "cyan")
                    await trading_agent.run_trading_cycle()  # Add await here
                
                # Run Risk Management
                if risk_agent:
                    cprint("\n🛡️ Running Risk Management...", "cyan")
                    risk_agent.run()

                # Run Strategy Analysis
                if strategy_agent:
                    cprint("\n📊 Running Strategy Analysis...", "cyan")
                    for token in trading_agent.tokens:  # Use trading agent's token list
                        if token not in EXCLUDED_TOKENS:
                            cprint(f"\n🔍 Analyzing {token}...", "cyan")
                            strategy_agent.get_signals(token)

                # Run CopyBot Analysis
                if copybot_agent:
                    cprint("\n🤖 Running CopyBot Portfolio Analysis...", "cyan")
                    copybot_agent.run_analysis_cycle()

                # Run Sentiment Analysis
                if sentiment_agent:
                    cprint("\n🎭 Running Sentiment Analysis...", "cyan")
                    sentiment_agent.run()

                # Sleep until next cycle
                next_run = datetime.now() + timedelta(minutes=SLEEP_BETWEEN_RUNS_MINUTES)
                cprint(f"\n😴 Sleeping until {next_run.strftime('%H:%M:%S')}", "cyan")
                await asyncio.sleep(60 * SLEEP_BETWEEN_RUNS_MINUTES)  # Use asyncio.sleep instead of time.sleep

            except Exception as e:
                cprint(f"\n❌ Error running agents: {str(e)}", "red")
                cprint("🔄 Continuing to next cycle...", "yellow")
                await asyncio.sleep(60)  # Use asyncio.sleep here too

    except KeyboardInterrupt:
        cprint("\n👋 Gracefully shutting down...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error in main loop: {str(e)}", "red")
        raise

if __name__ == "__main__":
    cprint("\n🌙 Billy Bitcoin AI Agent Trading System Starting...", "white", "on_blue")
    cprint("\n📊 Active Agents:", "white", "on_blue")
    for agent, active in ACTIVE_AGENTS.items():
        status = "✅ ON" if active else "❌ OFF"
        cprint(f"  • {agent.title()}: {status}", "white", "on_blue")
    print("\n")

    # Run the async main function
    asyncio.run(run_agents())