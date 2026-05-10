import os
import pytz
import sqlite3
import json
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca_day_bot.config import load_settings

class PnlTracker:
    def __init__(self):
        self.positions = {} # symbol -> (qty, avg_price)
        self.realized_pnl = 0.0
        self.trade_pnl = {} # ts -> pnl

    def add_fill(self, ts, symbol, side, qty, price):
        qty = float(qty)
        price = float(price)
        side = side.lower()
        
        # Normalize symbol for matching (remove slashes)
        sym = (symbol or "").strip().upper().replace("/", "")
        
        if sym not in self.positions:
            self.positions[sym] = [0.0, 0.0]
        
        q0, a0 = self.positions[sym]
        
        if side == "buy":
            # Opening or adding to long
            q1 = q0 + qty
            a1 = (a0 * q0 + price * qty) / q1 if q1 > 0 else 0
            self.positions[sym] = [q1, a1]
            return None
        else:
            # Selling (Closing or reducing long)
            sell_qty = min(qty, q0)
            if sell_qty > 0:
                pnl = (price - a0) * sell_qty
                self.realized_pnl += pnl
                q1 = q0 - sell_qty
                if q1 <= 0.0001:
                    del self.positions[sym]
                else:
                    self.positions[sym][0] = q1
                return pnl
            return 0.0

def fmt_money(x):
    return f"${x:,.2f}" if x is not None else "$0.00"

def main():
    settings = load_settings()
    # Support dual-instance reporting via STATE_DIR env var
    state_dir = os.getenv("STATE_DIR", "state")
    db_path = os.path.join(state_dir, "ledger.sqlite3")
    tz = pytz.timezone('America/Chicago')
    now = datetime.now(tz)
    today = now.date()
    
    # 1. Reconstruct Day from Ledger
    tracker = PnlTracker()
    transactions = []
    
    try:
        conn = sqlite3.connect(db_path)
        # Get all fills from the last 24h to ensure we catch entries for today's exits
        start_utc = (datetime.now(pytz.utc) - timedelta(hours=24)).isoformat()
        query = "SELECT ts, symbol, raw_json FROM trade_updates WHERE ts >= ? AND event = 'fill' ORDER BY ts ASC"
        rows = conn.execute(query, (start_utc,)).fetchall()
        conn.close()
        
        for r in rows:
            ts_str, symbol, raw_str = r
            raw = json.loads(raw_str)
            order = raw.get("payload", {}).get("order", {})
            side = order.get("side", "buy")
            qty = float(order.get("filled_qty", 0) or 0)
            price = float(order.get("filled_avg_price", 0) or 0)
            
            pnl = tracker.add_fill(ts_str, symbol, side, qty, price)
            
            ts_dt = datetime.fromisoformat(ts_str).astimezone(tz)
            if ts_dt.date() == today:
                transactions.append({
                    'ts': ts_dt,
                    'symbol': symbol,
                    'side': side.upper(),
                    'qty': qty,
                    'price': price,
                    'pnl': pnl
                })
    except Exception as e:
        print(f" [!] Error processing ledger: {e}")

    # 2. Get Live Open Positions
    unrealized_pnl = 0.0
    open_positions_display = []
    
    # Filter for symbols managed by THIS instance
    managed_symbols = set()
    if hasattr(settings, "symbols"):
        for s in settings.symbols:
            s_clean = s.strip().upper()
            managed_symbols.add(s_clean)
            # Add version without slash (e.g., BTC/USD -> BTCUSD)
            managed_symbols.add(s_clean.replace("/", ""))

    try:
        tc = TradingClient(settings.apca_api_key_id, settings.apca_api_secret_key, paper=True)
        positions = tc.get_all_positions()
        for p in positions:
            sym = p.symbol.strip().upper()
            if managed_symbols and sym not in managed_symbols:
                continue
                
            u = float(p.unrealized_pl)
            unrealized_pnl += u
            open_positions_display.append({
                'symbol': p.symbol,
                'qty': p.qty,
                'value': float(p.market_value),
                'pnl': u
            })
    except Exception as e:
        print(f" [!] Error fetching positions: {e}")

    # 3. Print Report
    print("\n" + "="*60)
    print(f" 📊 BOT PERFORMANCE UPDATE: {now.strftime('%Y-%m-%d %H:%M:%S')} CT")
    print("="*60)
    
    net_pnl = tracker.realized_pnl + unrealized_pnl
    print(f" Net P&L Today:   {net_pnl:+.2f} USD")
    print(f" (Realized: {tracker.realized_pnl:+.2f} | Unrealized: {unrealized_pnl:+.2f})")
    print(f" Total Fills Today: {len(transactions)}")

    print("\n --- OPEN POSITIONS ---")
    if not open_positions_display:
        print(" No open positions.")
    else:
        for p in open_positions_display:
            print(f" {p['symbol']:<6} | Qty: {p['qty']:>5} | Value: {fmt_money(p['value']):>10} | P&L: {p['pnl']:+.2f}")

    print("\n --- RECENT TRANSACTIONS (Today) ---")
    if not transactions:
        print(" No transactions today.")
    else:
        # Show last 15
        for t in reversed(transactions[-15:]):
            pnl_str = f" | P&L: {t['pnl']:+.2f}" if t['pnl'] is not None else ""
            print(f" {t['ts'].strftime('%H:%M:%S')} | {t['symbol']:<5} | {t['side']:<4} | {int(t['qty']):>5} @ ${t['price']:>7.2f}{pnl_str}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
