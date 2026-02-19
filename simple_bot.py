"""
MULTI-TIMEFRAME TRADING BOT
Top-down analysis from Monthly → Daily → Hourly → Minutes
With ON/OFF switches for each timeframe
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import ta

# ============================================================================
# 🎛️ TIMEFRAME CONFIGURATION
# ============================================================================

class TimeframeConfig:
    """Configure which timeframes to use"""
    
    # ========== API CREDENTIALS ==========
    ALPACA_API_KEY = ""
    ALPACA_SECRET_KEY = ""
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    
    # ========== TRADING SYMBOLS ==========
    SYMBOLS = ['SPY', 'QQQ', 'DIA', 'NASDAQ']
    
    # ========== TIMEFRAME ON/OFF SWITCHES ==========
    # Turn on/off each timeframe analysis
    USE_MONTHLY = True      # Long-term trend (cycles, accumulation)
    USE_WEEKLY = True       # Major trend direction
    USE_DAILY = True        # Market structure (HH/LL), immediate bias
    USE_4HOUR = True        # Entry setup refinement
    USE_1HOUR = True        # Bridge between long/short term
    USE_15MIN = False       # Precise entries (can be noisy)
    USE_5MIN = False        # Very precise but very noisy
    
    # ========== TREND ALIGNMENT RULES ==========
    REQUIRE_HIGHER_TF_ALIGNMENT = True  # Must align with higher timeframes?
    MIN_TIMEFRAMES_BULLISH = 3          # How many TF must be bullish to buy?
    
    # ========== INDICATORS ==========
    USE_EMA = True
    USE_RSI = True
    USE_MACD = True
    USE_VOLUME = True
    USE_SUPPORT_RESISTANCE = True
    
    # EMA Settings
    MA_FAST = 9
    MA_SLOW = 21
    MA_LONG = 50  # For longer timeframes
    
    # RSI Settings
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    # Volume
    VOLUME_THRESHOLD = 1.5
    
    # ========== RISK MANAGEMENT ==========
    POSITION_SIZE = 0.10
    STOP_LOSS = 0.02
    
    # ========== TIMING ==========
    CHECK_INTERVAL = 180  # Check every 3 minutes
    
    # ========== DEBUG MODE ==========
    SHOW_ANALYSIS_DETAILS = True  # Show detailed timeframe analysis

config = TimeframeConfig()

# ============================================================================
# TIMEFRAME MAPPING
# ============================================================================

TIMEFRAME_MAP = {
    'monthly': tradeapi.TimeFrame.Month,
    'weekly': tradeapi.TimeFrame.Week,
    'daily': tradeapi.TimeFrame.Day,
    '4hour': tradeapi.TimeFrame(4, tradeapi.TimeFrameUnit.Hour),
    '1hour': tradeapi.TimeFrame.Hour,
    '15min': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
    '5min': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
}

TIMEFRAME_BARS = {
    'monthly': 24,   # 2 years
    'weekly': 52,    # 1 year
    'daily': 100,    # 100 days
    '4hour': 60,     # 10 days
    '1hour': 168,    # 7 days
    '15min': 96,     # 1 day
    '5min': 78,      # 6.5 hours
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log(message, detail=False):
    """Print with timestamp"""
    if detail and not config.SHOW_ANALYSIS_DETAILS:
        return
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def get_timeframe_data(api, symbol, timeframe_name, bars_back=100):
    """Get data for specific timeframe"""
    try:
        timeframe = TIMEFRAME_MAP[timeframe_name]
        
        # Calculate lookback period
        if timeframe_name == 'monthly':
            start = datetime.now() - timedelta(days=730)  # 2 years
        elif timeframe_name == 'weekly':
            start = datetime.now() - timedelta(days=365)  # 1 year
        elif timeframe_name == 'daily':
            start = datetime.now() - timedelta(days=150)
        elif timeframe_name == '4hour':
            start = datetime.now() - timedelta(days=15)
        elif timeframe_name == '1hour':
            start = datetime.now() - timedelta(days=10)
        elif timeframe_name == '15min':
            start = datetime.now() - timedelta(days=2)
        else:  # 5min
            start = datetime.now() - timedelta(days=1)
        
        bars = api.get_bars(
            symbol,
            timeframe,
            start=start.strftime('%Y-%m-%d'),
            limit=TIMEFRAME_BARS.get(timeframe_name, 100),
            feed='iex'
        )
        
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            })
        
        if len(data) == 0:
            return None
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        log(f"Error getting {timeframe_name} data for {symbol}: {e}")
        return None

def analyze_timeframe(df, timeframe_name):
    """Analyze trend and structure for a timeframe"""
    try:
        if df is None or len(df) < 30:
            return None
        
        # Calculate EMAs
        df['EMA_fast'] = df['close'].ewm(span=config.MA_FAST, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=config.MA_SLOW, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=config.MA_LONG, adjust=False).mean()
        
        # Calculate RSI
        if config.USE_RSI:
            rsi = ta.momentum.RSIIndicator(df['close'], window=config.RSI_PERIOD).rsi()
            rsi_value = rsi.iloc[-1]
        else:
            rsi_value = 50
        
        # Calculate MACD
        if config.USE_MACD:
            macd = ta.trend.MACD(df['close'])
            macd_value = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_bullish = macd_value > macd_signal
        else:
            macd_bullish = True
        
        # Determine trend
        current_price = df['close'].iloc[-1]
        ema_fast = df['EMA_fast'].iloc[-1]
        ema_slow = df['EMA_slow'].iloc[-1]
        ema_long = df['EMA_long'].iloc[-1]
        
        # Trend determination
        if ema_fast > ema_slow > ema_long:
            trend = "STRONG BULLISH"
            trend_score = 3
        elif ema_fast > ema_slow:
            trend = "BULLISH"
            trend_score = 2
        elif ema_fast < ema_slow < ema_long:
            trend = "STRONG BEARISH"
            trend_score = -3
        elif ema_fast < ema_slow:
            trend = "BEARISH"
            trend_score = -2
        else:
            trend = "NEUTRAL"
            trend_score = 0
        
        # Market structure (Higher Highs / Lower Lows)
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        
        if recent_highs.iloc[-1] > recent_highs.iloc[-3]:
            structure = "HH"  # Higher High
        elif recent_lows.iloc[-1] < recent_lows.iloc[-3]:
            structure = "LL"  # Lower Low
        else:
            structure = "Ranging"
        
        # Volume trend
        if config.USE_VOLUME and len(df) >= 20:
            vol_ma = df['volume'].rolling(20).mean()
            volume_increasing = df['volume'].iloc[-1] > vol_ma.iloc[-1] * config.VOLUME_THRESHOLD
        else:
            volume_increasing = False
        
        return {
            'timeframe': timeframe_name,
            'trend': trend,
            'trend_score': trend_score,
            'structure': structure,
            'rsi': rsi_value,
            'macd_bullish': macd_bullish,
            'volume_strong': volume_increasing,
            'price': current_price,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'ema_long': ema_long,
            'is_bullish': trend_score > 0,
            'is_bearish': trend_score < 0
        }
        
    except Exception as e:
        log(f"Error analyzing {timeframe_name}: {e}")
        return None

def multi_timeframe_analysis(api, symbol):
    """Perform top-down multi-timeframe analysis"""
    
    log(f"\n{'='*70}", detail=True)
    log(f"📊 MULTI-TIMEFRAME ANALYSIS: {symbol}", detail=True)
    log(f"{'='*70}", detail=True)
    
    timeframes_to_check = []
    
    if config.USE_MONTHLY:
        timeframes_to_check.append('monthly')
    if config.USE_WEEKLY:
        timeframes_to_check.append('weekly')
    if config.USE_DAILY:
        timeframes_to_check.append('daily')
    if config.USE_4HOUR:
        timeframes_to_check.append('4hour')
    if config.USE_1HOUR:
        timeframes_to_check.append('1hour')
    if config.USE_15MIN:
        timeframes_to_check.append('15min')
    if config.USE_5MIN:
        timeframes_to_check.append('5min')
    
    if not timeframes_to_check:
        log("⚠️  No timeframes enabled!")
        return None
    
    analyses = {}
    
    # Analyze each timeframe (top-down)
    for tf in timeframes_to_check:
        log(f"\n  Analyzing {tf.upper()}...", detail=True)
        df = get_timeframe_data(api, symbol, tf)
        
        if df is not None:
            analysis = analyze_timeframe(df, tf)
            if analysis:
                analyses[tf] = analysis
                
                # Display analysis
                log(f"    Trend: {analysis['trend']}", detail=True)
                log(f"    Structure: {analysis['structure']}", detail=True)
                log(f"    RSI: {analysis['rsi']:.1f}", detail=True)
                log(f"    Price: ${analysis['price']:.2f}", detail=True)
                log(f"    EMA: Fast ${analysis['ema_fast']:.2f} | Slow ${analysis['ema_slow']:.2f} | Long ${analysis['ema_long']:.2f}", detail=True)
        else:
            log(f"    ⚠️  No data available", detail=True)
    
    if not analyses:
        log("  ❌ No timeframe data available")
        return None
    
    # Count bullish/bearish timeframes
    bullish_count = sum(1 for a in analyses.values() if a['is_bullish'])
    bearish_count = sum(1 for a in analyses.values() if a['is_bearish'])
    total_count = len(analyses)
    
    # Determine overall bias
    if bullish_count >= config.MIN_TIMEFRAMES_BULLISH:
        overall_bias = "BULLISH"
    elif bearish_count >= config.MIN_TIMEFRAMES_BULLISH:
        overall_bias = "BEARISH"
    else:
        overall_bias = "NEUTRAL"
    
    log(f"\n  {'='*66}", detail=True)
    log(f"  📈 BULLISH Timeframes: {bullish_count}/{total_count}", detail=True)
    log(f"  📉 BEARISH Timeframes: {bearish_count}/{total_count}", detail=True)
    log(f"  🎯 OVERALL BIAS: {overall_bias}", detail=True)
    log(f"  {'='*66}", detail=True)
    
    # Check alignment if required
    aligned = True
    if config.REQUIRE_HIGHER_TF_ALIGNMENT:
        # Check if higher timeframes agree with lower ones
        higher_tf_bullish = []
        
        if 'monthly' in analyses:
            higher_tf_bullish.append(analyses['monthly']['is_bullish'])
        if 'weekly' in analyses:
            higher_tf_bullish.append(analyses['weekly']['is_bullish'])
        if 'daily' in analyses:
            higher_tf_bullish.append(analyses['daily']['is_bullish'])
        
        if higher_tf_bullish:
            # Most higher TFs should agree
            aligned = sum(higher_tf_bullish) / len(higher_tf_bullish) >= 0.5
            
            if not aligned:
                log(f"  ⚠️  Higher timeframes NOT aligned!", detail=True)
    
    return {
        'symbol': symbol,
        'analyses': analyses,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'total_timeframes': total_count,
        'overall_bias': overall_bias,
        'aligned': aligned,
        'tradeable': aligned and (bullish_count >= config.MIN_TIMEFRAMES_BULLISH or 
                                  bearish_count >= config.MIN_TIMEFRAMES_BULLISH)
    }

def get_position(api, symbol):
    """Check position"""
    try:
        positions = api.list_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None
    except:
        return None

def place_order(api, symbol, side, qty):
    """Place order"""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        log(f"  {'🟢' if side == 'buy' else '🔴'} {side.upper()}: {symbol} - {qty} shares")
        return order
    except Exception as e:
        log(f"  ❌ Error placing order: {e}")
        return None

def trading_decision(api, symbol, mtf_analysis):
    """Make trading decision based on multi-timeframe analysis"""
    
    if not mtf_analysis or not mtf_analysis['tradeable']:
        log(f"{symbol}: Not tradeable (bias: {mtf_analysis['overall_bias'] if mtf_analysis else 'N/A'})")
        return
    
    position = get_position(api, symbol)
    bias = mtf_analysis['overall_bias']
    
    # Get daily analysis for current price
    daily_analysis = mtf_analysis['analyses'].get('daily')
    if not daily_analysis:
        log(f"{symbol}: No daily data")
        return
    
    current_price = daily_analysis['price']
    
    # Display summary
    log(f"{symbol}: ${current_price:.2f} | Bias: {bias} | TF: {mtf_analysis['bullish_count']}/{mtf_analysis['total_timeframes']} bullish")
    
    # Trading logic
    if position:
        # Have position - check if should sell
        entry = float(position.avg_entry_price)
        pnl_pct = ((current_price - entry) / entry) * 100
        
        log(f"  💼 Position: {position.qty} shares @ ${entry:.2f} | P&L: {pnl_pct:+.2f}%")
        
        # Stop loss
        if pnl_pct <= -(config.STOP_LOSS * 100):
            log(f"  🛑 STOP LOSS HIT")
            place_order(api, symbol, 'sell', int(position.qty))
            return
        
        # Sell if bias changed
        if bias == "BEARISH":
            log(f"  📉 SELL: Bias turned bearish")
            place_order(api, symbol, 'sell', int(position.qty))
    
    else:
        # No position - check if should buy
        if bias == "BULLISH" and mtf_analysis['aligned']:
            log(f"  📈 BUY SIGNAL: {mtf_analysis['bullish_count']}/{mtf_analysis['total_timeframes']} timeframes bullish")
            
            # Calculate quantity
            account = api.get_account()
            buying_power = float(account.buying_power)
            qty = int((buying_power * config.POSITION_SIZE) / current_price)
            
            if qty > 0:
                place_order(api, symbol, 'buy', qty)
            else:
                log(f"  ⚠️  Not enough buying power")
        elif bias == "BULLISH":
            log(f"  ⏸️  Bullish but not aligned across timeframes")
        else:
            log(f"  ⏸️  No buy signal (bias: {bias})")

def print_config():
    """Print configuration"""
    log("="*70)
    log("🎛️  MULTI-TIMEFRAME BOT CONFIGURATION")
    log("="*70)
    log(f"Symbols: {', '.join(config.SYMBOLS)}")
    log("")
    log("Timeframes Enabled:")
    log(f"  📅 Monthly (Cycles):      {'✅ ON' if config.USE_MONTHLY else '❌ OFF'}")
    log(f"  📅 Weekly (Major Trend):  {'✅ ON' if config.USE_WEEKLY else '❌ OFF'}")
    log(f"  📅 Daily (Structure):     {'✅ ON' if config.USE_DAILY else '❌ OFF'}")
    log(f"  🕐 4-Hour (Entry Setup):  {'✅ ON' if config.USE_4HOUR else '❌ OFF'}")
    log(f"  🕐 1-Hour (Refinement):   {'✅ ON' if config.USE_1HOUR else '❌ OFF'}")
    log(f"  🕐 15-Min (Precision):    {'✅ ON' if config.USE_15MIN else '❌ OFF'}")
    log(f"  🕐 5-Min (Day Trading):   {'✅ ON' if config.USE_5MIN else '❌ OFF'}")
    log("")
    log("Indicators:")
    log(f"  EMA: {'✅ ON' if config.USE_EMA else '❌ OFF'}")
    log(f"  RSI: {'✅ ON' if config.USE_RSI else '❌ OFF'}")
    log(f"  MACD: {'✅ ON' if config.USE_MACD else '❌ OFF'}")
    log(f"  Volume: {'✅ ON' if config.USE_VOLUME else '❌ OFF'}")
    log("")
    log(f"Require Higher TF Alignment: {'Yes' if config.REQUIRE_HIGHER_TF_ALIGNMENT else 'No'}")
    log(f"Min Timeframes Bullish: {config.MIN_TIMEFRAMES_BULLISH}")
    log(f"Check Interval: Every {config.CHECK_INTERVAL} seconds")
    log("="*70)
    log("")

def main():
    """Main function"""
    
    if "YOUR_ALPACA_API_KEY" in config.ALPACA_API_KEY:
        print("\n❌ Add your API keys!")
        return
    
    api = tradeapi.REST(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                        config.ALPACA_BASE_URL, api_version='v2')
    
    log("🤖 MULTI-TIMEFRAME BOT STARTED")
    print_config()
    
    while True:
        try:
            clock = api.get_clock()
            
            if not clock.is_open:
                log(f"💤 Market closed. Opens {clock.next_open}")
                time.sleep(300)
                continue
            
            log("\n" + "="*70)
            log(f"🔍 SCANNING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log("="*70)
            
            for symbol in config.SYMBOLS:
                try:
                    # Multi-timeframe analysis
                    mtf_analysis = multi_timeframe_analysis(api, symbol)
                    
                    # Trading decision
                    if mtf_analysis:
                        trading_decision(api, symbol, mtf_analysis)
                    
                    log("")  # Spacing
                    
                except Exception as e:
                    log(f"❌ Error analyzing {symbol}: {e}")
            
            log(f"⏳ Waiting {config.CHECK_INTERVAL} seconds...\n")
            time.sleep(config.CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            log("\n🛑 Bot stopped by user\n")
            break
        except Exception as e:
            log(f"❌ Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()