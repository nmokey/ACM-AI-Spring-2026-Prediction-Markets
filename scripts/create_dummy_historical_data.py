from pathlib import Path

import pandas as pd

path = Path(__file__).parents[1] / "data" / "features" / "historical_features.parquet"
path.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(
    {
        "contract_id": [
            "CID001",
            "CID002",
            "CID003",
            "CID004",
            "CID005",
            "CID006",
            "CID007",
            "CID008",
            "CID009",
            "CID010",
        ],
        "market_price": [0.10, 0.25, 0.60, 0.45, 0.80, 0.05, 0.30, 0.90, 0.15, 0.55],
        "volume_24h": [100, 50, 150, 80, 120, 20, 40, 200, 10, 90],
        "days_to_resolution": [5, 10, 3, 7, 1, 12, 8, 2, 6, 4],
        "btc_change_1h": [0.001, -0.002, 0.003, -0.001, 0.002, 0.0, -0.001, 0.004, 0.001, -0.002],
        "btc_change_6h": [0.005, 0.002, -0.003, 0.001, 0.004, -0.001, 0.0, 0.006, -0.002, 0.003],
        "eth_change_1h": [0.002, 0.001, -0.001, 0.003, 0.002, 0.0, 0.001, 0.004, -0.001, 0.002],
        "eth_change_6h": [0.006, 0.003, 0.0, -0.002, 0.005, 0.001, 0.002, 0.007, -0.001, 0.004],
        "precip_prob_new_york": [0.2, 0.8, 0.1, 0.4, 0.6, 0.3, 0.5, 0.0, 0.7, 0.2],
        "precip_prob_los_angeles": [0.1, 0.2, 0.0, 0.3, 0.1, 0.4, 0.2, 0.0, 0.5, 0.2],
        "precip_prob_chicago": [0.3, 0.4, 0.2, 0.6, 0.1, 0.7, 0.5, 0.2, 0.8, 0.3],
        "sentiment_score": [0.0, 0.1, -0.1, 0.3, -0.2, 0.0, 0.2, 0.4, -0.3, 0.1],
        "sentiment_confidence": [0.5, 0.7, 0.4, 0.8, 0.6, 0.3, 0.9, 0.7, 0.5, 0.6],
        "resolved_yes": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    }
)

df.to_parquet(path, index=False)
print(f"Created dummy file: {path}")
