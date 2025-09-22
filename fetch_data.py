import requests
import pandas as pd
from datetime import datetime
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# FPL API
API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_fpl_data():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
    except Exception as e:
        print("Error fetching FPL data:", e)
        return None

    data = response.json()
    players = data["elements"]

    df = pd.DataFrame(players)

    # Välj relevanta kolumner
    df = df[[
        "id", "first_name", "second_name", "team", "element_type",
        "now_cost", "selected_by_percent", "total_points", "status",
        "transfers_in_event", "transfers_out_event"
    ]]

    df.rename(columns={
        "id": "player_id",
        "first_name": "first_name",
        "second_name": "last_name",
        "element_type": "position",
        "now_cost": "price",
        "selected_by_percent": "ownership",
        "total_points": "points",
        "status": "status",
        "transfers_in_event": "transfers_in",
        "transfers_out_event": "transfers_out"
    }, inplace=True)

    # Beräkna net transfers
    df['net_transfers'] = df['transfers_in'] - df['transfers_out']

    # Spara daglig CSV
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(DATA_DIR, f"{today}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")
    return df


if __name__ == "__main__":
    fetch_fpl_data()
