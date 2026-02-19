"""
PROFESSIONAL ALGORITHMIC TRADING SYSTEM
Complete enterprise-grade system in one file
All settings at the top for easy configuration
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import logging
import ta

# ============================================================================
# 🎛️ MASTER CONTROL PANEL - CHANGE ALL SETTINGS HERE
# ============================================================================

class CONFIG:
    """Master configuration - All settings in one place"""
    
    # ========== 🔑 API CREDENTIALS ==========
    ALPACA_API_KEY = ""
    ALPACA_SECRET_KEY = ""
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    
    # ========== 📊 TIMEFRAMES ON/OFF ==========
    USE_MONTHLY = True          # Long-term cycles (accumulation/distribution)
    USE_WEEKLY = True           # Major trend direction
    USE_DAILY = True            # Market structure (HH/LL)
    USE_4HOUR = True            # Entry setup refinement
    USE_1HOUR = True            # Bridge between long/short term
    USE_15MIN = False           # Precise entries (noisy)
    USE_5MIN = False            # Day trading (very noisy)
    
    # ========== 📈 TECHNICAL INDICATORS ON/OFF ==========
    USE_EMA = True              # Exponential Moving Average (trend)
    USE_RSI = True              # Relative Strength Index (momentum)
    USE_MACD = True             # MACD (momentum)
    USE_BOLLINGER = False       # Bollinger Bands (volatility)
    USE_ATR = True              # Average True Range (volatility)
    USE_VOLUME = True           # Volume confirmation
    USE_SUPPORT_RESISTANCE = True  # Support/Resistance levels
    
    # ========== 🎯 INDICATOR PARAMETERS ==========
    EMA_FAST = 9                # Fast EMA period
    EMA_SLOW = 21               # Slow EMA period
    EMA_LONG = 50               # Long EMA period
    
    RSI_PERIOD = 14             # RSI calculation period
    RSI_OVERSOLD = 30           # RSI oversold level
    RSI_OVERBOUGHT = 70         # RSI overbought level
    
    MACD_FAST = 12              # MACD fast period
    MACD_SLOW = 26              # MACD slow period
    MACD_SIGNAL = 9             # MACD signal line period
    
    ATR_PERIOD = 14             # ATR period
    VOLUME_THRESHOLD = 1.5      # Volume must be 1.5x average
    
    # ========== 🎲 TRADING LOGIC ==========
    MIN_TIMEFRAMES_BULLISH = 3  # How many timeframes must agree
    REQUIRE_HIGHER_TF_ALIGNMENT = True  # Must align with higher TF?
    
    # ========== 💰 POSITION SIZING ==========
    POSITION_SIZE_PCT = 10.0    # Use 10% of portfolio per trade
    MAX_POSITION_SIZE_PCT = 15.0  # Maximum 15% per position
    MAX_POSITIONS = 5           # Maximum open positions
    
    # Position sizing method: 'fixed' or 'volatility_adjusted' or 'kelly'
    POSITION_SIZE_METHOD = 'volatility_adjusted'
    
    # ========== 🛡️ RISK MANAGEMENT ==========
    USE_STOP_LOSS = True        # Use stop loss?
    STOP_LOSS_PCT = 2.0         # Stop loss percentage (2%)
    STOP_LOSS_ATR_MULTIPLE = 2.0  # Or use ATR: stop at 2x ATR
    USE_ATR_STOP = True         # Use ATR-based stop (more dynamic)
    
    USE_TRAILING_STOP = True    # Use trailing stop?
    TRAILING_STOP_ACTIVATION_PCT = 2.0  # Activate at 2% profit
    TRAILING_STOP_DISTANCE_PCT = 1.0    # Trail by 1%
    
    USE_PROFIT_TARGET = True    # Use profit target?
    PROFIT_TARGET_PCT = 6.0     # Take profit at 6%
    RISK_REWARD_RATIO = 3.0     # Or use 3:1 risk/reward
    USE_RISK_REWARD = True      # Use risk/reward instead of fixed %
    
    MAX_PORTFOLIO_RISK_PCT = 2.0  # Max 2% portfolio risk per trade
    MAX_DAILY_LOSS_PCT = 5.0    # Stop trading if lose 5% in one day
    MAX_DRAWDOWN_PCT = 10.0     # Emergency stop at 10% drawdown
    
    # ========== ⏱️ TIMING ==========
    SCAN_INTERVAL = 180       # Check every 180 seconds (3 minutes)
    DATA_UPDATE_INTERVAL = 1800  # Update data half- hour
    TRADE_COOLDOWN_HOURS = 24   # Wait 24h before trading same symbol again
    
    ENABLE_PREMARKET = False    # Trade in pre-market?
    ENABLE_AFTERHOURS = False   # Trade after hours?
    
    # ========== 📊 ANALYTICS & REPORTING ==========
    SHOW_DETAILED_ANALYSIS = True  # Show detailed TF analysis
    LOG_TRADES_TO_DATABASE = True  # Save trades to DB
    CALCULATE_PERFORMANCE = True   # Calculate real-time metrics
    PRINT_PERFORMANCE_INTERVAL = 3600  # Print metrics every hour
    
    # ========== 🔔 ALERTS ==========
    ENABLE_ALERTS = False       # Enable alerts (future: Telegram/email)
    ALERT_ON_TRADE = True       # Alert when trade executed
    ALERT_ON_STOP_LOSS = True   # Alert on stop loss
    ALERT_ON_PROFIT = True      # Alert on profit target
    
    # ========== 🐛 DEBUG ==========
    LOG_LEVEL = "INFO"          # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TimeframeAnalysis:
    """Analysis result for single timeframe"""
    timeframe: str
    trend: str  # STRONG_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG_BEARISH
    trend_score: int  # 3, 2, 0, -2, -3
    price: float
    ema_fast: float
    ema_slow: float
    ema_long: float
    rsi: float
    macd: float
    macd_signal: float
    atr: float
    volume_ratio: float
    structure: str  # HH, LL, Ranging
    is_bullish: bool
    is_bearish: bool

@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    symbol: str
    side: str
    qty: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    profit_target: float
    risk_amount: float

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class Logger:
    """Professional logging"""
    
    def __init__(self):
        Path("logs").mkdir(exist_ok=True)
        log_file = Path("logs") / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=getattr(logging, CONFIG.LOG_LEVEL),
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingSystem')
    
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class Database:
    """Trade history database"""
    
    def __init__(self):
        Path("data").mkdir(exist_ok=True)
        self.db_path = Path("data") / "trades.db"
        self.init_db()
    
    def init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                entry_price REAL,
                exit_price REAL,
                entry_time TEXT,
                exit_time TEXT,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                timestamp TEXT,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                total_return REAL,
                portfolio_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_data):
        """Save completed trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', trade_data)
        conn.commit()
        conn.close()
    
    def get_performance(self):
        """Get performance stats"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_time DESC LIMIT 100", conn)
        conn.close()
        return df

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Risk management system"""
    
    def __init__(self, api, logger):
        self.api = api
        self.logger = logger
        self.initial_capital = None
        self.peak_value = None
        self.last_trade_time = {}
    
    def initialize(self):
        """Initialize risk manager"""
        account = self.api.get_account()
        self.initial_capital = float(account.portfolio_value)
        self.peak_value = self.initial_capital
        self.logger.info(f"Risk Manager initialized | Capital: ${self.initial_capital:,.2f}")
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, atr=None):
        """Calculate optimal position size"""
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)
        buying_power = float(account.buying_power)
        
        if CONFIG.POSITION_SIZE_METHOD == 'fixed':
            # Fixed percentage
            position_value = portfolio_value * (CONFIG.POSITION_SIZE_PCT / 100)
            qty = int(position_value / entry_price)
        
        elif CONFIG.POSITION_SIZE_METHOD == 'volatility_adjusted':
            # Based on risk per trade
            risk_per_trade = portfolio_value * (CONFIG.MAX_PORTFOLIO_RISK_PCT / 100)
            price_risk = abs(entry_price - stop_loss)
            qty = int(risk_per_trade / price_risk) if price_risk > 0 else 0
        
        else:  # kelly
            # Kelly Criterion (simplified)
            win_rate = 0.55
            avg_win_loss_ratio = 2.0
            kelly_pct = (win_rate - (1 - win_rate) / avg_win_loss_ratio) * 100
            kelly_pct = min(kelly_pct, 25)  # Cap at 25%
            position_value = portfolio_value * (kelly_pct / 100)
            qty = int(position_value / entry_price)
        
        # Apply max position size
        max_value = portfolio_value * (CONFIG.MAX_POSITION_SIZE_PCT / 100)
        max_qty = int(max_value / entry_price)
        qty = min(qty, max_qty)
        
        # Check buying power
        if qty * entry_price > buying_power:
            qty = int(buying_power * 0.95 / entry_price)
        
        return max(0, qty)
    
    def calculate_stop_loss(self, entry_price, atr, side='buy'):
        """Calculate stop loss"""
        if not CONFIG.USE_STOP_LOSS:
            return 0.0
        
        if CONFIG.USE_ATR_STOP and atr:
            stop_distance = atr * CONFIG.STOP_LOSS_ATR_MULTIPLE
        else:
            stop_distance = entry_price * (CONFIG.STOP_LOSS_PCT / 100)
        
        if side == 'buy':
            return round(entry_price - stop_distance, 2)
        else:
            return round(entry_price + stop_distance, 2)
    
    def calculate_profit_target(self, entry_price, stop_loss, side='buy'):
        """Calculate profit target"""
        if not CONFIG.USE_PROFIT_TARGET:
            return None
        
        if CONFIG.USE_RISK_REWARD:
            risk = abs(entry_price - stop_loss)
            reward = risk * CONFIG.RISK_REWARD_RATIO
            if side == 'buy':
                return round(entry_price + reward, 2)
            else:
                return round(entry_price - reward, 2)
        else:
            target_pct = CONFIG.PROFIT_TARGET_PCT / 100
            if side == 'buy':
                return round(entry_price * (1 + target_pct), 2)
            else:
                return round(entry_price * (1 - target_pct), 2)
    
    def check_drawdown(self):
        """Check drawdown limit"""
        account = self.api.get_account()
        current_value = float(account.portfolio_value)
        
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown_pct = ((self.peak_value - current_value) / self.peak_value) * 100
        
        if drawdown_pct >= CONFIG.MAX_DRAWDOWN_PCT:
            self.logger.critical(f"⛔ DRAWDOWN LIMIT HIT: {drawdown_pct:.2f}%")
            return False
        
        return True
    
    def check_cooldown(self, symbol):
        """Check trade cooldown"""
        if symbol not in self.last_trade_time:
            return True
        
        hours_passed = (datetime.now() - self.last_trade_time[symbol]).total_seconds() / 3600
        return hours_passed >= CONFIG.TRADE_COOLDOWN_HOURS
    
    def record_trade(self, symbol):
        """Record trade time"""
        self.last_trade_time[symbol] = datetime.now()

# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """Manage market data"""
    
    def __init__(self, api, logger):
        self.api = api
        self.logger = logger
        self.cache = {}
        self.last_update = {}
    
    def get_timeframe_data(self, symbol, timeframe, bars=100):
        """Get data for timeframe"""
        cache_key = f"{symbol}_{timeframe}"
        now = time.time()
        
        # Check cache
        if cache_key in self.last_update:
            if (now - self.last_update[cache_key]) < CONFIG.DATA_UPDATE_INTERVAL:
                return self.cache.get(cache_key)
        
        try:
            # Define timeframe mapping
            tf_map = {
                'monthly': (tradeapi.TimeFrame.Month, 24),
                'weekly': (tradeapi.TimeFrame.Week, 52),
                'daily': (tradeapi.TimeFrame.Day, 100),
                '4hour': (tradeapi.TimeFrame(4, tradeapi.TimeFrameUnit.Hour), 60),
                '1hour': (tradeapi.TimeFrame.Hour, 168),
                '15min': (tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute), 96),
                '5min': (tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute), 78),
            }
            
            tf_obj, limit = tf_map.get(timeframe, (tradeapi.TimeFrame.Day, 100))
            
            # Calculate start date
            if timeframe == 'monthly':
                start = datetime.now() - timedelta(days=730)
            elif timeframe == 'weekly':
                start = datetime.now() - timedelta(days=365)
            else:
                start = datetime.now() - timedelta(days=150)
            
            bars = self.api.get_bars(
                symbol,
                tf_obj,
                start=start.strftime('%Y-%m-%d'),
                limit=limit,
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
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Cache it
            self.cache[cache_key] = df
            self.last_update[cache_key] = now
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting {timeframe} data for {symbol}: {e}")
            return None

# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    """Multi-timeframe strategy"""
    
    def __init__(self, data_manager, logger):
        self.data_manager = data_manager
        self.logger = logger
    
    def analyze_timeframe(self, df, timeframe):
        """Analyze single timeframe"""
        if df is None or len(df) < 30:
            return None
        
        try:
            # Calculate EMAs
            if CONFIG.USE_EMA:
                df['EMA_fast'] = df['close'].ewm(span=CONFIG.EMA_FAST, adjust=False).mean()
                df['EMA_slow'] = df['close'].ewm(span=CONFIG.EMA_SLOW, adjust=False).mean()
                df['EMA_long'] = df['close'].ewm(span=CONFIG.EMA_LONG, adjust=False).mean()
                
                ema_fast = df['EMA_fast'].iloc[-1]
                ema_slow = df['EMA_slow'].iloc[-1]
                ema_long = df['EMA_long'].iloc[-1]
            else:
                ema_fast = ema_slow = ema_long = 0
            
            # Calculate RSI
            if CONFIG.USE_RSI:
                rsi = ta.momentum.RSIIndicator(df['close'], window=CONFIG.RSI_PERIOD).rsi()
                rsi_value = rsi.iloc[-1]
            else:
                rsi_value = 50
            
            # Calculate MACD
            if CONFIG.USE_MACD:
                macd = ta.trend.MACD(df['close'], window_fast=CONFIG.MACD_FAST, 
                                    window_slow=CONFIG.MACD_SLOW, window_sign=CONFIG.MACD_SIGNAL)
                macd_value = macd.macd().iloc[-1]
                macd_signal = macd.macd_signal().iloc[-1]
            else:
                macd_value = macd_signal = 0
            
            # Calculate ATR
            if CONFIG.USE_ATR:
                atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 
                                                     window=CONFIG.ATR_PERIOD).average_true_range()
                atr_value = atr.iloc[-1]
            else:
                atr_value = 0
            
            # Volume
            if CONFIG.USE_VOLUME:
                vol_ma = df['volume'].rolling(20).mean()
                volume_ratio = df['volume'].iloc[-1] / vol_ma.iloc[-1]
            else:
                volume_ratio = 1.0
            
            # Determine trend
            current_price = df['close'].iloc[-1]
            
            if CONFIG.USE_EMA:
                if ema_fast > ema_slow > ema_long:
                    trend = "STRONG_BULLISH"
                    trend_score = 3
                elif ema_fast > ema_slow:
                    trend = "BULLISH"
                    trend_score = 2
                elif ema_fast < ema_slow < ema_long:
                    trend = "STRONG_BEARISH"
                    trend_score = -3
                elif ema_fast < ema_slow:
                    trend = "BEARISH"
                    trend_score = -2
                else:
                    trend = "NEUTRAL"
                    trend_score = 0
            else:
                trend = "NEUTRAL"
                trend_score = 0
            
            # Market structure
            recent_highs = df['high'].tail(10)
            recent_lows = df['low'].tail(10)
            
            if recent_highs.iloc[-1] > recent_highs.iloc[-3]:
                structure = "HH"
            elif recent_lows.iloc[-1] < recent_lows.iloc[-3]:
                structure = "LL"
            else:
                structure = "Ranging"
            
            return TimeframeAnalysis(
                timeframe=timeframe,
                trend=trend,
                trend_score=trend_score,
                price=current_price,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_long=ema_long,
                rsi=rsi_value,
                macd=macd_value,
                macd_signal=macd_signal,
                atr=atr_value,
                volume_ratio=volume_ratio,
                structure=structure,
                is_bullish=trend_score > 0,
                is_bearish=trend_score < 0
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {timeframe}: {e}")
            return None
    
    def multi_timeframe_analysis(self, symbol):
        """Analyze multiple timeframes"""
        
        timeframes = {
            'monthly': CONFIG.USE_MONTHLY,
            'weekly': CONFIG.USE_WEEKLY,
            'daily': CONFIG.USE_DAILY,
            '4hour': CONFIG.USE_4HOUR,
            '1hour': CONFIG.USE_1HOUR,
            '15min': CONFIG.USE_15MIN,
            '5min': CONFIG.USE_5MIN
        }
        
        enabled_tf = [tf for tf, enabled in timeframes.items() if enabled]
        
        if not enabled_tf:
            self.logger.warning("No timeframes enabled!")
            return None
        
        analyses = {}
        
        if CONFIG.SHOW_DETAILED_ANALYSIS:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"📊 ANALYZING: {symbol}")
            self.logger.info(f"{'='*70}")
        
        for tf in enabled_tf:
            df = self.data_manager.get_timeframe_data(symbol, tf)
            if df is not None:
                analysis = self.analyze_timeframe(df, tf)
                if analysis:
                    analyses[tf] = analysis
                    
                    if CONFIG.SHOW_DETAILED_ANALYSIS:
                        self.logger.info(f"  {tf.upper():8} | {analysis.trend:15} | "
                                       f"Price: ${analysis.price:.2f} | RSI: {analysis.rsi:.1f}")
        
        if not analyses:
            return None
        
        # Count votes
        bullish_count = sum(1 for a in analyses.values() if a.is_bullish)
        bearish_count = sum(1 for a in analyses.values() if a.is_bearish)
        total = len(analyses)
        
        # Determine bias
        if bullish_count >= CONFIG.MIN_TIMEFRAMES_BULLISH:
            bias = "BULLISH"
        elif bearish_count >= CONFIG.MIN_TIMEFRAMES_BULLISH:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        
        # Check alignment
        aligned = True
        if CONFIG.REQUIRE_HIGHER_TF_ALIGNMENT:
            higher_tf = [a for tf, a in analyses.items() if tf in ['monthly', 'weekly', 'daily']]
            if higher_tf:
                higher_bullish = sum(1 for a in higher_tf if a.is_bullish)
                aligned = higher_bullish / len(higher_tf) >= 0.5
        
        if CONFIG.SHOW_DETAILED_ANALYSIS:
            self.logger.info(f"  {'='*66}")
            self.logger.info(f"  BULLISH: {bullish_count}/{total} | BEARISH: {bearish_count}/{total} | "
                           f"BIAS: {bias} | ALIGNED: {aligned}")
            self.logger.info(f"  {'='*66}")
        
        # Get daily analysis for ATR
        daily_analysis = analyses.get('daily')
        atr = daily_analysis.atr if daily_analysis else None
        
        return {
            'symbol': symbol,
            'bias': bias,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'total': total,
            'aligned': aligned,
            'tradeable': aligned and (bullish_count >= CONFIG.MIN_TIMEFRAMES_BULLISH or 
                                     bearish_count >= CONFIG.MIN_TIMEFRAMES_BULLISH),
            'analyses': analyses,
            'current_price': daily_analysis.price if daily_analysis else None,
            'atr': atr
        }

# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """Handle order execution"""
    
    def __init__(self, api, risk_manager, logger, db):
        self.api = api
        self.risk_manager = risk_manager
        self.logger = logger
        self.db = db
        self.active_trades = {}
    
    def get_position(self, symbol):
        """Get current position"""
        try:
            return self.api.get_position(symbol)
        except:
            return None
    
    def place_order(self, symbol, side, qty):
        """Place market order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            self.logger.info(f"{'🟢' if side=='buy' else '🔴'} ORDER: {side.upper()} {qty} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Order error: {e}")
            return None
    
    def open_position(self, symbol, analysis):
        """Open new position"""
        
        if not analysis['tradeable']:
            return
        
        # Check if we already have position
        if self.get_position(symbol):
            return
        
        # Check cooldown
        if not self.risk_manager.check_cooldown(symbol):
            self.logger.info(f"{symbol}: In cooldown period")
            return
        
        # Check max positions
        positions = self.api.list_positions()
        if len(positions) >= CONFIG.MAX_POSITIONS:
            self.logger.info(f"Max positions ({CONFIG.MAX_POSITIONS}) reached")
            return
        
        # Calculate entry, stop, target
        entry_price = analysis['current_price']
        atr = analysis['atr']
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, 'buy')
        profit_target = self.risk_manager.calculate_profit_target(entry_price, stop_loss, 'buy')
        qty = self.risk_manager.calculate_position_size(symbol, entry_price, stop_loss, atr)
        
        if qty == 0:
            self.logger.info(f"{symbol}: Position size = 0")
            return
        
        # Execute order
        order = self.place_order(symbol, 'buy', qty)
        
        if order:
            # Record trade
            trade = Trade(
                trade_id=order.id,
                symbol=symbol,
                side='buy',
                qty=qty,
                entry_price=entry_price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                profit_target=profit_target,
                risk_amount=(entry_price - stop_loss) * qty
            )
            
            self.active_trades[symbol] = trade
            self.risk_manager.record_trade(symbol)
            
            self.logger.info(f"  Entry: ${entry_price:.2f} | Stop: ${stop_loss:.2f} | "
                           f"Target: ${profit_target:.2f} | Risk: ${trade.risk_amount:.2f}")
    
    def manage_position(self, symbol):
        """Manage existing position"""
        
        position = self.get_position(symbol)
        if not position:
            return
        
        current_price = float(position.current_price)
        entry_price = float(position.avg_entry_price)
        qty = int(position.qty)
        
        # Get trade info
        trade = self.active_trades.get(symbol)
        if not trade:
            # Position exists but no trade record - create one
            return
        
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Check stop loss
        if CONFIG.USE_STOP_LOSS and current_price <= trade.stop_loss:
            self.logger.info(f"🛑 {symbol}: STOP LOSS HIT at ${current_price:.2f} ({pnl_pct:.2f}%)")
            self.close_position(symbol, 'stop_loss')
            return
        
        # Check profit target
        if CONFIG.USE_PROFIT_TARGET and trade.profit_target and current_price >= trade.profit_target:
            self.logger.info(f"🎯 {symbol}: PROFIT TARGET HIT at ${current_price:.2f} ({pnl_pct:.2f}%)")
            self.close_position(symbol, 'profit_target')
            return
        
        # Update trailing stop
        if CONFIG.USE_TRAILING_STOP:
            if pnl_pct >= CONFIG.TRAILING_STOP_ACTIVATION_PCT:
                trail_stop = current_price * (1 - CONFIG.TRAILING_STOP_DISTANCE_PCT / 100)
                if trail_stop > trade.stop_loss:
                    trade.stop_loss = trail_stop
    
    def close_position(self, symbol, reason):
        """Close position"""
        
        position = self.get_position(symbol)
        if not position:
            return
        
        qty = int(position.qty)
        exit_price = float(position.current_price)
        
        order = self.place_order(symbol, 'sell', qty)
        
        if order:
            trade = self.active_trades.get(symbol)
            if trade:
                pnl = (exit_price - trade.entry_price) * qty
                pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                
                self.logger.info(f"  Exit: ${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
                               f"Reason: {reason}")
                
                # Save to database
                if CONFIG.LOG_TRADES_TO_DATABASE:
                    trade_data = (
                        trade.trade_id,
                        symbol,
                        'buy',
                        qty,
                        trade.entry_price,
                        exit_price,
                        trade.entry_time.isoformat(),
                        datetime.now().isoformat(),
                        pnl,
                        pnl_pct,
                        reason
                    )
                    self.db.save_trade(trade_data)
                
                # Remove from active
                del self.active_trades[symbol]

# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.last_print = time.time()
    
    def calculate_metrics(self):
        """Calculate performance"""
        
        df = self.db.get_performance()
        
        if len(df) == 0:
            return None
        
        total_trades = len(df)
        winning = df[df['pnl'] > 0]
        losing = df[df['pnl'] <= 0]
        
        win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0
        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing['pnl'].mean()) if len(losing) > 0 else 0
        profit_factor = (avg_win * len(winning)) / (avg_loss * len(losing)) if len(losing) > 0 and avg_loss > 0 else 0
        total_return = df['pnl'].sum()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return
        }
    
    def print_performance(self, force=False):
        """Print performance metrics"""
        
        now = time.time()
        if not force and (now - self.last_print) < CONFIG.PRINT_PERFORMANCE_INTERVAL:
            return
        
        metrics = self.calculate_metrics()
        if not metrics:
            return
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"📊 PERFORMANCE METRICS")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  Total Trades: {metrics['total_trades']}")
        self.logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
        self.logger.info(f"  Avg Win: ${metrics['avg_win']:.2f}")
        self.logger.info(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
        self.logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        self.logger.info(f"  Total P&L: ${metrics['total_return']:.2f}")
        self.logger.info(f"{'='*70}\n")
        
        self.last_print = now

# ============================================================================
# MAIN TRADING SYSTEM
# ============================================================================

class TradingSystem:
    """Main orchestrator"""
    
    def __init__(self, symbols):
        self.symbols = symbols
        self.logger = Logger()
        self.db = Database()
        
        # Initialize components
        self.api = tradeapi.REST(CONFIG.ALPACA_API_KEY, CONFIG.ALPACA_SECRET_KEY,
                                 CONFIG.ALPACA_BASE_URL, api_version='v2')
        
        self.risk_manager = RiskManager(self.api, self.logger)
        self.data_manager = DataManager(self.api, self.logger)
        self.strategy = StrategyEngine(self.data_manager, self.logger)
        self.execution = ExecutionEngine(self.api, self.risk_manager, self.logger, self.db)
        self.performance = PerformanceTracker(self.db, self.logger)
    
    def print_config(self):
        """Print system configuration"""
        
        self.logger.info("="*70)
        self.logger.info("🤖 PROFESSIONAL TRADING SYSTEM")
        self.logger.info("="*70)
        self.logger.info(f"Symbols: {', '.join(self.symbols)}")
        self.logger.info("")
        
        self.logger.info("Timeframes:")
        tfs = {
            'Monthly': CONFIG.USE_MONTHLY,
            'Weekly': CONFIG.USE_WEEKLY,
            'Daily': CONFIG.USE_DAILY,
            '4-Hour': CONFIG.USE_4HOUR,
            '1-Hour': CONFIG.USE_1HOUR,
            '15-Min': CONFIG.USE_15MIN,
            '5-Min': CONFIG.USE_5MIN
        }
        for name, enabled in tfs.items():
            self.logger.info(f"  {name:10} {'✅ ON' if enabled else '❌ OFF'}")
        
        self.logger.info("")
        self.logger.info("Indicators:")
        inds = {
            'EMA': CONFIG.USE_EMA,
            'RSI': CONFIG.USE_RSI,
            'MACD': CONFIG.USE_MACD,
            'ATR': CONFIG.USE_ATR,
            'Volume': CONFIG.USE_VOLUME,
            'S/R': CONFIG.USE_SUPPORT_RESISTANCE
        }
        for name, enabled in inds.items():
            self.logger.info(f"  {name:10} {'✅ ON' if enabled else '❌ OFF'}")
        
        self.logger.info("")
        self.logger.info(f"Position Size: {CONFIG.POSITION_SIZE_PCT}% ({CONFIG.POSITION_SIZE_METHOD})")
        self.logger.info(f"Stop Loss: {CONFIG.STOP_LOSS_PCT}% (ATR: {CONFIG.USE_ATR_STOP})")
        self.logger.info(f"Profit Target: {CONFIG.PROFIT_TARGET_PCT}% (R:R {CONFIG.RISK_REWARD_RATIO}:1)")
        self.logger.info(f"Max Positions: {CONFIG.MAX_POSITIONS}")
        self.logger.info(f"Scan Interval: {CONFIG.SCAN_INTERVAL}s")
        self.logger.info("="*70)
        self.logger.info("")
    
    def run(self):
        """Main trading loop"""
        
        self.print_config()
        self.risk_manager.initialize()
        
        self.logger.info("🚀 System started - Waiting for market open...\n")
        
        while True:
            try:
                # Check market hours
                clock = self.api.get_clock()
                if not clock.is_open:
                    self.logger.info(f"💤 Market closed. Opens: {clock.next_open}")
                    time.sleep(300)
                    continue
                
                # Check drawdown
                if not self.risk_manager.check_drawdown():
                    self.logger.critical("⛔ EMERGENCY STOP: Max drawdown exceeded")
                    break
                
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"🔍 SCANNING - {datetime.now().strftime('%H:%M:%S')}")
                self.logger.info(f"{'='*70}")
                
                # Scan each symbol
                for symbol in self.symbols:
                    try:
                        # Analyze
                        analysis = self.strategy.multi_timeframe_analysis(symbol)
                        
                        if not analysis:
                            continue
                        
                        # Display summary
                        self.logger.info(f"\n{symbol}: ${analysis['current_price']:.2f} | "
                                       f"Bias: {analysis['bias']} | "
                                       f"{analysis['bullish_count']}/{analysis['total']} bullish")
                        
                        # Check if have position
                        position = self.execution.get_position(symbol)
                        
                        if position:
                            # Manage existing position
                            self.execution.manage_position(symbol)
                            
                            # Check if should close based on signal reversal
                            if analysis['bias'] == 'BEARISH':
                                self.logger.info(f"  ⚠️  Bias turned bearish")
                                self.execution.close_position(symbol, 'signal_reversal')
                        else:
                            # Look for entry
                            if analysis['bias'] == 'BULLISH':
                                self.logger.info(f"  📈 BULLISH SIGNAL")
                                self.execution.open_position(symbol, analysis)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                
                # Print performance
                if CONFIG.CALCULATE_PERFORMANCE:
                    self.performance.print_performance()
                
                # Wait
                self.logger.info(f"\n⏳ Next scan in {CONFIG.SCAN_INTERVAL}s...\n")
                time.sleep(CONFIG.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                self.logger.info("\n🛑 System stopped by user")
                break
            except Exception as e:
                self.logger.error(f"System error: {e}")
                time.sleep(60)

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("🤖 PROFESSIONAL ALGORITHMIC TRADING SYSTEM")
    print("="*70 + "\n")
    
    # Check API keys
    if "YOUR_ALPACA_API_KEY" in CONFIG.ALPACA_API_KEY:
        print("❌ ERROR: Please add your Alpaca API keys in CONFIG class at top of file")
        return
    
    # Get symbols from user
    print("Enter the stock symbols you want to trade (comma-separated)")
    print("Example: SPY,QQQ,DIA,IWM")
    print("\nOr press Enter for default (SPY,QQQ,DIA,IWM): ")
    
    user_input = input().strip()
    
    if user_input:
        symbols = [s.strip().upper() for s in user_input.split(',')]
    else:
        symbols = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    print(f"\n✅ Trading symbols: {', '.join(symbols)}")
    print("\nStarting system in 3 seconds...")
    time.sleep(3)
    
    # Create and run system
    system = TradingSystem(symbols)
    system.run()

if __name__ == "__main__":
    main()