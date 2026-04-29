import sys
import os
from datetime import datetime

# Add the project root to path so we can import our client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingestion.crypto_client import CryptoClient

def run_smoketest():
    print("🚀 Starting CryptoClient Smoketest...\n")
    client = CryptoClient()
    test_symbol = "BTC-USD"

    try:
        # 1. Test get_price
        print(f"--- Testing get_price('{test_symbol}') ---")
        price = client.get_price(test_symbol)
        assert isinstance(price, float), "Price should be a float"
        print(f"✅ Success: Current {test_symbol} price is ${price:,.2f}\n")

        # 2. Test get_24h_stats
        print(f"--- Testing get_24h_stats('{test_symbol}') ---")
        stats = client.get_24h_stats(test_symbol)
        required_keys = ["last_price", "volume_24h", "price_change_pct_24h"]
        for key in required_keys:
            assert key in stats, f"Stats missing key: {key}"
        print(f"✅ Success: 24h Volume is {stats['volume_24h']}")
        print(f"✅ Success: 24h Change is {stats['price_change_pct_24h']}%\n")

        # 3. Test get_klines
        print(f"--- Testing get_klines('{test_symbol}', limit=10) ---")
        candles = client.get_klines(test_symbol, limit=10)
        assert len(candles) <= 10, f"Expected max 10 candles, got {len(candles)}"
        assert "close" in candles[0], "Candle missing 'close' price"
        assert isinstance(candles[0]["open_time"], datetime), "open_time should be datetime object"
        print(f"✅ Success: Retrieved {len(candles)} candles.")
        print(f"✅ Success: Oldest candle time: {candles[0]['open_time']}\n")

        # 4. Test compute_price_changes
        print(f"--- Testing compute_price_changes('{test_symbol}') ---")
        changes = client.compute_price_changes(test_symbol)
        print(f"📊 Current: ${changes['current_price']:,.2f}")
        print(f"📊 1h Change: {changes['price_change_1h']:.4%}")
        print(f"📊 6h Change: {changes['price_change_6h']:.4%}")
        
        assert -1.0 < changes['price_change_1h'] < 1.0, "1h change seems unrealistic (>100%)"
        print("\n✅ ALL TESTS PASSED!")

    except Exception as e:
        print(f"\n❌ SMOKETEST FAILED: {str(e)}")
        # Print more context for debugging
        if "APIKey" in str(e):
            print("💡 Tip: Check your .env file for COINBASE_API_KEY/SECRET.")
        sys.exit(1)

if __name__ == "__main__":
    run_smoketest()