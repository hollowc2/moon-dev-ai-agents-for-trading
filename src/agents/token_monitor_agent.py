"""
🌙 Billy Bitcoin's Token Monitor Agent
Monitors and selects tokens based on various criteria
"""



##
## Future Ideas:
## tokens by market cap
## tokens by volatility
## tokens by price
## tokens by volume
## tokens by age
## tokens by trading activity
## tokens by market sentiment


from src.agents.base_agent import BaseAgent
from src import nice_funcs_cb as cb
from src.config import TOP_TOKENS_TO_MONITOR, STABLECOIN_IDENTIFIERS
import time

class TokenMonitorAgent(BaseAgent):
    def __init__(self):
        """Initialize the Token Monitor Agent"""
        super().__init__(agent_type="token_monitor")
        self.tokens = ["BTC-USD", "ETH-USD"]  # Default starting tokens
        print("🎯 Token Monitor Agent initialized!")

    def get_top_volume_tokens(self):
        """Get top tokens by normalized volume"""
        try:
            print("🔄 Updating monitored tokens...")
            # Get all USD trading pairs
            all_pairs = cb.get_all_products()
            
            if not all_pairs:
                print("⚠️ Could not fetch trading pairs, keeping current list")
                return self.tokens
                

                
            # Filter for USD pairs and get their volumes with price normalization
            usd_pairs = []
            for pair in all_pairs:
                try:
                    # Check if it's a USD pair and not a stablecoin
                    if (pair.endswith('-USD') and 
                        not any(stable in pair for stable in STABLECOIN_IDENTIFIERS)):
                        
                        # Get 24h volume and current price
                        volume = cb.get_product_24h_volume(pair)
                        price = cb.get_product_price(pair)
                        
                        # Validate volume and price
                        if volume and price and isinstance(volume, (int, float)) and isinstance(price, (int, float)):
                            normalized_volume = volume * price  # Normalize volume by price
                            usd_pairs.append({
                                'pair': pair,
                                'volume': volume,
                                'price': price,
                                'normalized_volume': normalized_volume
                            })
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Error processing {pair}: {str(e)}")
                    continue
            
            # Sort by normalized volume and get top N pairs
            top_pairs = sorted(usd_pairs, key=lambda x: x['normalized_volume'], reverse=True)
            top_pairs = top_pairs[:TOP_TOKENS_TO_MONITOR]
            
            if not top_pairs:
                print("⚠️ No valid pairs found, keeping current list")
                return self.tokens
            
            print(f"\n📊 Top {TOP_TOKENS_TO_MONITOR} trading pairs by normalized volume (24h):")
            for i, pair in enumerate(top_pairs, 1):
                print(f"{i}. {pair['pair']}")
                print(f"   • Raw Volume: {pair['volume']:.2f}")
                print(f"   • Price: ${pair['price']:.8f}")
                print(f"   • Normalized Volume: ${pair['normalized_volume']:,.2f}")
            
            # Update tokens list
            self.tokens = [pair['pair'] for pair in top_pairs]
            print(f"\n🔄 Updated monitoring list: {', '.join(self.tokens)}")
            
            return self.tokens
            
        except Exception as e:
            print(f"❌ Error updating monitored tokens: {str(e)}")
         
            return self.tokens

    

    def run(self):
        """Run the token monitor agent"""
        return self.get_top_volume_tokens()
