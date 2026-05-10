import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import ClosePositionRequest

load_dotenv()

api_key = os.getenv("APCA_API_KEY_ID")
api_secret = os.getenv("APCA_API_SECRET_KEY")

# TradingClient automatically uses paper if paper=True or if the base URL is paper
client = TradingClient(api_key, api_secret, paper=True)

symbols_to_flatten = ["BITO", "CORZ", "BMNU", "FXI"]

print(f"Starting force-flatten of {symbols_to_flatten}...")

for symbol in symbols_to_flatten:
    try:
        print(f"Attempting to close position for {symbol}...")
        client.close_position(symbol)
        print(f"Successfully submitted close order for {symbol}.")
    except Exception as e:
        if "position does not exist" in str(e).lower():
            print(f"No open position found for {symbol}, skipping.")
        else:
            print(f"Failed to close {symbol}: {e}")

print("Flattening script complete.")
