import pandas as pd
import os

def run_integration_pipeline():
    print(" Starting Feature Engineering Smoke Test...")

    # 1. Define Paths
    RAW_DIR = "data/raw"
    FEATURES_DIR = "data/features"
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 2. Load the Raw Data
    try:
        crypto = pd.read_csv(f"{RAW_DIR}/coinbase_data.csv")
        kalshi = pd.read_csv(f"{RAW_DIR}/kalshi_data.csv")
        weather = pd.read_csv(f"{RAW_DIR}/weather_data.csv")
    except FileNotFoundError as e:
        print(f" Error: Could not find raw files. Did you run the clients first?\n{e}")
        return

    # 3. Clean and Align Timestamps
    for df in [crypto, kalshi, weather]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)

    # 4. The Merge (The "Integration" part of the test)
    # We join everything together based on the timestamp
    print(" Integrating datasets...")
    merged = pd.merge_asof(crypto, kalshi, on='timestamp', direction='backward')
    final_df = pd.merge_asof(merged, weather, on='timestamp', direction='backward')

    # 5. Create Features (The "Engineering" part)
    # Example: Price momentum
    final_df['price_momentum'] = final_df['price'].pct_change()
    
    # 6. Export for Team 2
    output_path = f"{FEATURES_DIR}/live_features.parquet"
    final_df.to_parquet(output_path, index=False)
    
    print(f" Smoke Test Success! Created: {output_path}")
    print(f" Total Features Produced: {len(final_df.columns)}")

if __name__ == "__main__":
    run_integration_pipeline()