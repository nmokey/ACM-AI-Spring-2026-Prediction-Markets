import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import the existing logging function and schema
from execution.dry_run import log_dry_run_trade, CONFIG
from data.schema import TradeRecord
from live_signals import compute_live_signal_for_today

logger = logging.getLogger(__name__)

class LiveDryRunBot:
    def __init__(self, threshold_f: float = 85.0, lookback_days: int = 30):
        self.threshold_f = threshold_f
        self.lookback_days = lookback_days
        
        # Track active tracking positions in-memory: {contract_id: position_details}
        self.active_positions = {} 
        
    def fetch_historical_weather_data(self) -> pd.DataFrame:
        """
        Mock or real ingestion layer: Retrieves up-to-date historical daily TMAX.
        In production, replace with an automated NOAA API call or updated CSV read.
        """
        # Example loading real local dataset
        try:
            return pd.read_csv("data/historical_temps.csv")
        except FileNotFoundError:
            # Fallback mock dataframe for isolated execution
            dates = pd.date_range(end=datetime.now(), periods=60)
            return pd.DataFrame({
                "DATE": dates,
                "TMAX": np.random.uniform(70, 95, size=60)
            })

    def fetch_kalshi_open_contracts(self) -> list[dict]:
        """
        Simulates fetching actual open weather contracts from Kalshi API.
        Returns active contracts with their current market mid/yes prices.
        """
        # Real implementation would use: requests.get("https://api.kalshi.com/v2/markets"...)
        # Filtering for weather contracts (e.g., KXWEATHER)
        return [
            {
                "contract_id": "WEATHER-26MAY19-T85",
                "title": "Will Los Angeles hit >85F on 2026-05-19?",
                "yes_price": 0.48, # Live market price for YES contract
                "is_open": True,
                "target_date": "2026-05-19"
            },
            {
                "contract_id": "WEATHER-26MAY20-T85",
                "title": "Will Los Angeles hit >85F on 2026-05-20?",
                "yes_price": 0.52,
                "is_open": True,
                "target_date": "2026-05-20"
            }
        ]

    def check_resolutions(self, historical_data: pd.DataFrame):
        """
        Monitors active positions against newly recorded data points to see if they settled.
        """
        resolved_contracts = []
        df_sorted = historical_data.sort_values("DATE")

        for contract_id, pos in list(self.active_positions.items()):
            target_date_str = pos["target_date"]
            
            # Check if our historical database now has the true outcome for this target date
            day_row = df_sorted[df_sorted["DATE"].astype(str) == target_date_str]
            
            if not day_row.empty:
                actual_tmax = day_row.iloc[0]["TMAX"]
                resolved_yes = 1 if actual_tmax > self.threshold_f else 0
                
                # Determine PnL metadata
                final_status = "WIN" if (pos["side"] == "YES" and resolved_yes == 1) or \
                                       (pos["side"] == "NO" and resolved_yes == 0) else "LOSS"
                
                logger.info(f"Contract {contract_id} Resolved! Result: {final_status}. TMAX was {actual_tmax}°F.")
                
                # Append to structural log or specific settlement file if needed
                resolved_contracts.append(contract_id)
                del self.active_positions[contract_id]

    def run_iteration(self):
        logger.info("Starting live dry-run evaluation loop iteration...")
        
        # 1. Fetch data updates
        weather_df = self.fetch_historical_weather_data()
        open_contracts = self.fetch_kalshi_open_contracts()
        
        # 2. Check on older positions to see if they can be closed/resolved
        self.check_resolutions(weather_df)
        
        # 3. Calculate today's internal model metric
        p_model = compute_live_signal_for_today(weather_df, self.threshold_f, self.lookback_days)
        logger.info(f"Current P_Model for threshold {self.threshold_f}°F is: {p_model:.4f}")
        
        # 4. Evaluate live contracts
        for contract in open_contracts:
            contract_id = contract["contract_id"]
            
            # Skip if we already hold a mock position in this exact contract
            if contract_id in self.active_positions:
                continue
                
            market_price = contract["yes_price"]
            edge = p_model - market_price
            
            # Trading Conditions
            side = None
            if edge > 0.05:
                side = "YES"
            elif edge < -0.05:
                side = "NO"
                
            if side:
                absolute_edge = abs(edge)
                # Build object conforming to schema
                # Convert your float decimal prices (e.g., 0.48) to integer cents (e.g., 48)
                limit_price_cents = int(round(market_price * 100))
                market_price_cents = int(round(market_price * 100))

                # Note: Check if your schema also expects p_model and edge as integers (cents)
                # If p_model is 0.5667, multiplying by 100 and rounding yields 57 cents
                p_model_cents = int(round(p_model * 100))
                edge_cents = int(round(absolute_edge * 100))

                record = TradeRecord(
                    contract_id=contract_id,
                    timestamp=datetime.now(),
                    side=side,
                    size=10,  
                    limit_price=limit_price_cents,      # Now an Integer (e.g., 48)
                    p_model=p_model_cents,              # Now an Integer (e.g., 57)
                    market_price=market_price_cents,    # Now an Integer (e.g., 48)
                    edge=edge_cents,                    # Now an Integer (e.g., 9)
                    mode="dry_run"
                )
                
                # Safely execute mirror writing via your shared dry_run script
                log_dry_run_trade(record)
                logger.info(f"Executed Dry-Run Order: {side.upper()} on {contract_id} | Edge: {absolute_edge:.4f}")
                
                # Store position to trace lifecycle execution
                self.active_positions[contract_id] = {
                    "side": side,
                    "entry_price": market_price,
                    "target_date": contract["target_date"],
                    "timestamp": record.timestamp
                }

    def start_polling(self, poll_interval_seconds: int = 3600):
        """
        Infinite polling sequence. Weather contracts shift slowly, 
        so defaults to check hourly.
        """
        logger.info("Initializing Live Dry Run Bot Polling System...")
        while True:
            try:
                self.run_iteration()
            except Exception as e:
                logger.error(f"Error encountered during runtime cycle: {e}", exc_info=True)
            
            # Sleep until next check
            time.sleep(poll_interval_seconds)

if __name__ == "__main__":
    # Setup baseline terminal logging visibility
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    bot = LiveDryRunBot(threshold_f=85.0, lookback_days=30)
    # Set to shorter interval (e.g., 10 seconds) for immediate local debugging/sanity check
    bot.start_polling(poll_interval_seconds=10)