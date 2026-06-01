import os
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, AssetClass

def main():
    # Load environment
    env_path = Path(__file__).parent.parent / '.env.stocks_trial'
    load_dotenv(env_path)
    
    api_key = os.getenv('APCA_API_KEY_ID')
    api_secret = os.getenv('APCA_API_SECRET_KEY')
    
    if not api_key or not api_secret:
        print("Error: Alpaca API credentials not found in .env.stocks_trial")
        return

    client = TradingClient(api_key, api_secret, paper=True)
    
    try:
        # Starting Capital for the Stocks Trial Portfolio
        STARTING_CAPITAL = 2500.0
        
        # 1. Fetch Open Positions
        positions = client.get_all_positions()
        stock_positions = [p for p in positions if p.asset_class == AssetClass.US_EQUITY]
        
        total_unrealized = 0.0
        for p in stock_positions:
            unrealized = float(p.unrealized_pl)
            total_unrealized += unrealized
            
        # 2. Fetch Today's Orders
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50)
        orders = client.get_orders(filter=req)
        
        today = datetime.date.today()
        filled_today = []
        for o in orders:
            if o.filled_at and o.filled_at.date() == today and o.asset_class == AssetClass.US_EQUITY:
                filled_today.append(o)
                
        # Group fills by symbol to calculate realized P&L
        trade_log = {}
        realized_pl = 0.0
        for o in filled_today:
            sym = o.symbol
            qty = int(float(o.qty))
            price = float(o.filled_avg_price)
            if sym not in trade_log:
                trade_log[sym] = []
            trade_log[sym].append((o.side.name, qty, price))
            
        for sym, trades in trade_log.items():
            buys = sum(q*p for side, q, p in trades if side == 'BUY')
            buy_qty = sum(q for side, q, p in trades if side == 'BUY')
            sells = sum(q*p for side, q, p in trades if side == 'SELL')
            sell_qty = sum(q for side, q, p in trades if side == 'SELL')
            
            matched_qty = min(buy_qty, sell_qty)
            if matched_qty > 0:
                avg_buy = buys / buy_qty if buy_qty > 0 else 0
                avg_sell = sells / sell_qty if sell_qty > 0 else 0
                pnl = (avg_sell - avg_buy) * matched_qty
                realized_pl += pnl

        net_today_pnl = realized_pl + total_unrealized
        virtual_equity = STARTING_CAPITAL + net_today_pnl
        daily_change_pct = (net_today_pnl / STARTING_CAPITAL) * 100
        
        print("\n" + "="*70)
        print(f" 📈 AETHERIS LIVE PERFORMANCE REPORT (TRIAL) - {datetime.date.today()}")
        print("="*70)
        print(f" Trial Capital Baseline:  ${STARTING_CAPITAL:,.2f}")
        print(f" Virtual Equity:          ${virtual_equity:,.2f}")
        print(f" Today's Change:          ${net_today_pnl:+,.2f} ({daily_change_pct:+.2f}%)")
        print("-" * 70)
        
        # 1. Open Positions (Stocks Only)
        print(" 🟢 OPEN POSITIONS (STOCKS)")
        print(f" {'SYMBOL':<8} | {'QTY':<8} | {'AVG ENTRY':<10} | {'CURRENT':<10} | {'UNREALIZED P&L':<15}")
        print("-" * 70)
        
        if not stock_positions:
            print("  No open stock positions.")
        else:
            for p in stock_positions:
                unrealized = float(p.unrealized_pl)
                print(f" {p.symbol:<8} | {int(float(p.qty)):<8} | ${float(p.avg_entry_price):<9.2f} | ${float(p.current_price):<9.2f} | {unrealized:<+14.2f}")
                
        print("-" * 70)
        print(f" Total Open Stock Positions: {len(stock_positions)}")
        print(f" Total Unrealized P&L:       ${total_unrealized:+,.2f}")
        print("="*70)
        
        # 2. Closed Trades Today (Filled Orders)
        print(" 🔴 CLOSED / FILLED TRADES TODAY (STOCKS)")
        print(f" {'TIME (UTC)':<10} | {'SYMBOL':<8} | {'SIDE':<6} | {'QTY':<8} | {'PRICE':<10} | {'STATUS':<10}")
        print("-" * 70)
        
        if not filled_today:
            print("  No filled stock orders today.")
        else:
            # Sort by fill time ascending
            filled_today.sort(key=lambda x: x.filled_at)
            for o in filled_today:
                fill_time = o.filled_at.strftime("%H:%M:%S")
                print(f" {fill_time:<10} | {o.symbol:<8} | {o.side.name:<6} | {int(float(o.qty)):<8} | ${float(o.filled_avg_price):<9.2f} | {o.status.name:<10}")
                
        print("-" * 70)
        print(" 📊 REALIZED P&L BREAKDOWN (STOCKS)")
        print("-" * 70)
        
        has_realized = False
        for sym, trades in trade_log.items():
            buys = sum(q*p for side, q, p in trades if side == 'BUY')
            buy_qty = sum(q for side, q, p in trades if side == 'BUY')
            sells = sum(q*p for side, q, p in trades if side == 'SELL')
            sell_qty = sum(q for side, q, p in trades if side == 'SELL')
            
            matched_qty = min(buy_qty, sell_qty)
            if matched_qty > 0:
                has_realized = True
                avg_buy = buys / buy_qty if buy_qty > 0 else 0
                avg_sell = sells / sell_qty if sell_qty > 0 else 0
                pnl = (avg_sell - avg_buy) * matched_qty
                print(f" {sym:<8}: Realized P&L on {matched_qty} shares: ${pnl:+,.2f} (Bought avg: ${avg_buy:.2f}, Sold avg: ${avg_sell:.2f})")
                
        if not has_realized:
            print("  No round-trip trades completed today yet to realize P&L.")
            
        print("-" * 70)
        print(f" Total Realized Stock P&L:   ${realized_pl:+,.2f}")
        print(f" Net Today Stock P&L (Est):  ${net_today_pnl:+,.2f}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Error fetching data from Alpaca: {e}")

if __name__ == "__main__":
    main()
