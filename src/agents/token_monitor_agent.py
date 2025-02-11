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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

# Coinbase API Rate Limits
PUBLIC_REQUESTS_PER_SEC = 8  # Keep under 10 limit for safety
PRIVATE_REQUESTS_PER_SEC = 12  # Keep under 15 limit for safety
REQUEST_DELAY = 1.0 / PUBLIC_REQUESTS_PER_SEC  # Delay between requests

class TokenMonitorAgent(BaseAgent):
    def __init__(self):
        """Initialize the Token Monitor Agent"""
        super().__init__(agent_type="token_monitor")
        self.tokens = []  # Start with empty list
        self.cache_lock = threading.Lock()
        self.last_update = None  # Set to None initially
        self.cache_ttl = timedelta(minutes=15)  # Cache TTL
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = time.time()
        print("🎯 Token Monitor Agent initialized!")

    def _rate_limit_delay(self):
        """Enforce rate limiting between requests"""
        with self.rate_limit_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)
            self.last_request_time = time.time()

    @lru_cache(maxsize=100)
    def get_token_metrics(self, pair):
        """Get volume and price for a single token pair (cached)"""
        try:
            # Enforce rate limiting
            self._rate_limit_delay()
            
            volume = cb.get_product_24h_volume(pair)
            
            # Enforce rate limiting between requests
            self._rate_limit_delay()
            
            price = cb.get_product_price(pair)
            
            if volume and price and isinstance(volume, (int, float)) and isinstance(price, (int, float)):
                return {
                    'pair': pair,
                    'volume': volume,
                    'price': price,
                    'normalized_volume': volume * price
                }
        except Exception as e:
            print(f"⚠️ Error getting metrics for {pair}: {str(e)}")
        return None

    def process_token_pair(self, pair):
        """Process a single token pair"""
        try:
            if (pair.endswith('-USD') and 
                not any(stable in pair for stable in STABLECOIN_IDENTIFIERS)):
                return self.get_token_metrics(pair)
        except Exception as e:
            print(f"⚠️ Error processing {pair}: {str(e)}")
        return None

    def get_top_volume_tokens(self):
        """Get top tokens by normalized volume using rate-limited parallel processing"""
        try:
            print("🔄 Updating monitored tokens...")
            
            # Check if cache is valid
            if (self.last_update is not None and 
                (datetime.now() - self.last_update) < self.cache_ttl and 
                self.tokens):
                print("✨ Using cached token list")
                return self.tokens

            print("📊 Fetching fresh token data...")
            
            # Rate limit the initial products request
            self._rate_limit_delay()
            all_pairs = cb.get_all_products()
            
            if not all_pairs:
                print("⚠️ Could not fetch trading pairs, keeping current list")
                return self.tokens

            # Process pairs in parallel with rate limiting
            valid_pairs = []
            # Limit concurrent threads to respect rate limits
            max_workers = min(PUBLIC_REQUESTS_PER_SEC, 8)  # Conservative limit
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(self.process_token_pair, pair): pair 
                    for pair in all_pairs
                }
                
                for future in as_completed(future_to_pair):
                    result = future.result()
                    if result:
                        valid_pairs.append(result)

            # Sort by normalized volume and get top N pairs
            if valid_pairs:
                top_pairs = sorted(
                    valid_pairs, 
                    key=lambda x: x['normalized_volume'], 
                    reverse=True
                )[:TOP_TOKENS_TO_MONITOR]

                print(f"\n📊 Top {TOP_TOKENS_TO_MONITOR} trading pairs by normalized volume (24h):")
                for i, pair in enumerate(top_pairs, 1):
                    print(f"{i}. {pair['pair']}")
                    print(f"   • Raw Volume: {pair['volume']:,.2f}")
                    print(f"   • Price: ${pair['price']:,.8f}")
                    print(f"   • Normalized Volume: ${pair['normalized_volume']:,.2f}")

                # Update tokens list with lock
                with self.cache_lock:
                    self.tokens = [pair['pair'] for pair in top_pairs]
                    self.last_update = datetime.now()

                print(f"\n🔄 Updated monitoring list: {', '.join(self.tokens)}")
                return self.tokens

            return self.tokens

        except Exception as e:
            print(f"❌ Error updating monitored tokens: {str(e)}")
            return self.tokens

    def clear_cache(self):
        """Clear the metrics cache"""
        self.get_token_metrics.cache_clear()

    def run(self):
        """Run the token monitor agent"""
        try:
            tokens = self.get_top_volume_tokens()
            # Clear cache after successful run
            self.clear_cache()
            return tokens
        except Exception as e:
            print(f"❌ Error in token monitor run: {str(e)}")
            return self.tokens
