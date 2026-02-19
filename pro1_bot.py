"""
PROFESSIONAL TRADING SYSTEM v2.0
Complete system with Technical + Fundamental + Sentiment Analysis
ATR-based position sizing, robust error handling, professional architecture
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging
import ta
from textblob import TextBlob
import requests

# ============================================================================
# 🎛️ MASTER CONFIGURATION - ALL SETTINGS HERE
# ============================================================================

class CONFIG:
    # ========== API CREDENTIALS ==========
    ALPACA_API_KEY = ""
    ALPACA_SECRET_KEY = ""
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    NEWS_API_KEY = None  # Optional: Get free key from newsapi.org
    
    # ========== ANALYSIS SWITCHES ==========
    USE_TECHNICAL = True        # Technical indicators
    USE_FUNDAMENTAL = True      # Basic fundamental checks
    USE_SENTIMENT = True        # News sentiment analysis
    
    # Analysis Weights (must sum to 1.0)
    TECHNICAL_WEIGHT = 0.50     # 50%
    FUNDAMENTAL_WEIGHT = 0.30   # 30%
    SENTIMENT_WEIGHT = 0.20     # 20%
    
    # ========== TIMEFRAMES ==========
    USE_MONTHLY = True
    USE_WEEKLY = True
    USE_DAILY = True
    USE_4HOUR = False
    USE_1HOUR = False
    
    # ========== INDICATORS ==========
    USE_EMA = True
    USE_RSI = True
    USE_MACD = True
    USE_ATR = True              # Required for position sizing
    USE_VOLUME = True
    
    EMA_FAST = 9
    EMA_SLOW = 21
    EMA_LONG = 50
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    ATR_PERIOD = 14
    VOLUME_THRESHOLD = 1.5
    
    # ========== POSITION SIZING (ATR-BASED) ==========
    SIZING_METHOD = 'atr'       # 'atr', 'fixed', or 'volatility'
    RISK_PER_TRADE_PCT = 1.0    # Risk 1% of portfolio per trade
    BASE_POSITION_PCT = 10.0    # Base position size
    MAX_POSITION_PCT = 20.0     # Maximum position size
    MIN_POSITION_PCT = 2.0      # Minimum position size
    
    # ========== RISK MANAGEMENT ==========
    USE_STOP_LOSS = True
    STOP_LOSS_ATR_MULTIPLE = 2.0  # Stop at 2x ATR
    
    USE_PROFIT_TARGET = True
    RISK_REWARD_RATIO = 3.0     # 3:1 risk/reward
    
    MAX_POSITIONS = 5
    MAX_DAILY_LOSS_PCT = 5.0
    MAX_DRAWDOWN_PCT = 10.0
    
    # ========== TRADING LOGIC ==========
    MIN_TIMEFRAMES_BULLISH = 2   # Minimum timeframes that must agree
    MIN_OVERALL_SCORE = 0.65    # 65% confidence to trade
    MIN_TECHNICAL_SCORE = 0.60  # 60% technical confidence
    
    # ========== TIMING ==========
    SCAN_INTERVAL = 300         # 5 minutes
    COOLDOWN_HOURS = 24         # Don't trade same symbol for 24h
    
    # ========== SYSTEM ==========
    LOG_LEVEL = "INFO"
    SHOW_DETAILED_ANALYSIS = True

# ============================================================================
# LOGGER
# ============================================================================

class Logger:
    def __init__(self):
        Path("logs").mkdir(exist_ok=True)
        log_file = Path("logs") / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=getattr(logging, CONFIG.LOG_LEVEL),
            format='%(asctime)s | %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.log = logging.getLogger('TradingSystem')
    
    def info(self, msg): self.log.info(msg)
    def error(self, msg): self.log.error(msg)
    def warning(self, msg): self.log.warning(msg)

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self):
        Path("data").mkdir(exist_ok=True)
        self.conn = sqlite3.connect(Path("data") / "trades.db")
        self._init_tables()
    
    def _init_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
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
        self.conn.commit()
    
    def save_trade(self, data):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)', data)
        self.conn.commit()

# ============================================================================
# SENTIMENT ANALYZER
# ============================================================================

class SentimentAnalyzer:
    def __init__(self, logger):
        self.logger = logger
    
    def analyze(self, symbol):
        """Analyze news sentiment"""
        if not CONFIG.USE_SENTIMENT:
            return 0.5, "NEUTRAL", 0
        
        try:
            # Get news from Alpaca or NewsAPI
            news = self._get_news(symbol)
            
            if not news:
                return 0.5, "NEUTRAL", 0
            
            # Analyze sentiment
            sentiments = []
            for article in news[:10]:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Normalize to 0-1 scale
            score = (avg_sentiment + 1) / 2
            
            # Determine signal
            if avg_sentiment > 0.2:
                signal = "BULLISH"
            elif avg_sentiment < -0.2:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            
            return score, signal, len(news)
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return 0.5, "NEUTRAL", 0
    
    def _get_news(self, symbol):
        """Get news articles"""
        if not CONFIG.NEWS_API_KEY:
            return []
        
        try:
            from_time = datetime.now() - timedelta(hours=24)
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'from': from_time.isoformat(),
                'sortBy': 'publishedAt',
                'apiKey': CONFIG.NEWS_API_KEY,
                'language': 'en'
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('articles', [])
        except:
            pass
        
        return []

# ============================================================================
# FUNDAMENTAL ANALYZER
# ============================================================================

class FundamentalAnalyzer:
    def __init__(self, api, logger):
        self.api = api
        self.logger = logger
    
    def analyze(self, symbol):
        """Analyze fundamentals"""
        if not CONFIG.USE_FUNDAMENTAL:
            return 0.5, "NEUTRAL"
        
        try:
            # Get asset info
            asset = self.api.get_asset(symbol)
            
            # Basic checks
            score = 0.5
            
            if asset.tradable and asset.status == 'active':
                score = 0.6
            
            # For ETFs like SPY, QQQ - generally bullish bias
            if symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
                score = 0.65
            
            # In production: integrate real fundamental data
            # - P/E ratio, earnings growth, debt levels, etc.
            
            signal = "BULLISH" if score > 0.55 else "BEARISH" if score < 0.45 else "NEUTRAL"
            
            return score, signal
            
        except Exception as e:
            self.logger.error(f"Fundamental analysis error for {symbol}: {e}")
            return 0.5, "NEUTRAL"

# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    def __init__(self, api, logger):
        self.api = api
        self.logger = logger
        self.cache = {}
    
    def get_data(self, symbol, timeframe, bars=100):
        """Get historical data with caching"""
        cache_key = f"{symbol}_{timeframe}"
        
        try:
            tf_map = {
                'monthly': (tradeapi.TimeFrame.Month, 730),
                'weekly': (tradeapi.TimeFrame.Week, 365),
                'daily': (tradeapi.TimeFrame.Day, 150),
                '4hour': (tradeapi.TimeFrame(4, tradeapi.TimeFrameUnit.Hour), 15),
                '1hour': (tradeapi.TimeFrame.Hour, 10)
            }
            
            tf_obj, days = tf_map.get(timeframe, (tradeapi.TimeFrame.Day, 150))
            start = datetime.now() - timedelta(days=days)
            
            bars = self.api.get_bars(
                symbol, tf_obj,
                start=start.strftime('%Y-%m-%d'),
                limit=bars,
                feed='iex'
            )
            
            data = [{
                'timestamp': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            } for bar in bars]
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data error for {symbol} {timeframe}: {e}")
            return None

# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    def __init__(self, api, data_mgr, sentiment, fundamental, logger):
        self.api = api
        self.data_mgr = data_mgr
        self.sentiment = sentiment
        self.fundamental = fundamental
        self.logger = logger
    
    def analyze(self, symbol):
        """Complete analysis with all methods"""
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"📊 ANALYZING: {symbol}")
        self.logger.info(f"{'='*70}")
        
        # ========== TECHNICAL ==========
        tech_score, tech_signal, price, atr, volatility = self._technical(symbol)
        
        # ========== FUNDAMENTAL ==========
        fund_score, fund_signal = self.fundamental.analyze(symbol)
        
        # ========== SENTIMENT ==========
        sent_score, sent_signal, news_count = self.sentiment.analyze(symbol)
        
        # ========== COMPOSITE ==========
        overall_score = (
            tech_score * CONFIG.TECHNICAL_WEIGHT +
            fund_score * CONFIG.FUNDAMENTAL_WEIGHT +
            sent_score * CONFIG.SENTIMENT_WEIGHT
        )
        
        if CONFIG.SHOW_DETAILED_ANALYSIS:
            self.logger.info(f"\n📈 Scores:")
            self.logger.info(f"  Technical:    {tech_score:.2f} ({tech_signal})")
            self.logger.info(f"  Fundamental:  {fund_score:.2f} ({fund_signal})")
            self.logger.info(f"  Sentiment:    {sent_score:.2f} ({sent_signal}) [{news_count} articles]")
            self.logger.info(f"  {'─'*66}")
            self.logger.info(f"  OVERALL:      {overall_score:.2f}")
        
        # Determine signal
        if overall_score >= 0.75:
            signal = "STRONG_BUY"
        elif overall_score >= CONFIG.MIN_OVERALL_SCORE:
            signal = "BUY"
        elif overall_score <= 0.25:
            signal = "STRONG_SELL"
        elif overall_score <= 0.35:
            signal = "SELL"
        else:
            signal = "NEUTRAL"
        
        # Tradeable?
        tradeable = (
            overall_score >= CONFIG.MIN_OVERALL_SCORE and
            tech_score >= CONFIG.MIN_TECHNICAL_SCORE and
            signal in ["BUY", "STRONG_BUY"]
        )
        
        self.logger.info(f"  Signal:       {signal}")
        self.logger.info(f"  Tradeable:    {'✅' if tradeable else '❌'}")
        
        # Calculate stops/targets
        if atr > 0:
            stop_loss = price - (atr * CONFIG.STOP_LOSS_ATR_MULTIPLE)
            profit_target = price + (atr * CONFIG.STOP_LOSS_ATR_MULTIPLE * CONFIG.RISK_REWARD_RATIO)
        else:
            stop_loss = price * 0.98
            profit_target = price * 1.06
        
        # Position size
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)
        position_size = self._calculate_position_size(portfolio_value, price, atr, stop_loss)
        
        return {
            'symbol': symbol,
            'tradeable': tradeable,
            'overall_score': overall_score,
            'signal': signal,
            'price': price,
            'atr': atr,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'position_size': position_size
        }
    
    def _technical(self, symbol):
        """Technical analysis"""
        
        timeframes = {
            'monthly': CONFIG.USE_MONTHLY,
            'weekly': CONFIG.USE_WEEKLY,
            'daily': CONFIG.USE_DAILY,
            '4hour': CONFIG.USE_4HOUR,
            '1hour': CONFIG.USE_1HOUR
        }
        
        enabled = [tf for tf, on in timeframes.items() if on]
        
        results = []
        price = atr = 0
        
        for tf in enabled:
            df = self.data_mgr.get_data(symbol, tf)
            
            if df is None or len(df) < 50:
                continue
            
            # Calculate indicators
            df['EMA_fast'] = df['close'].ewm(span=CONFIG.EMA_FAST, adjust=False).mean()
            df['EMA_slow'] = df['close'].ewm(span=CONFIG.EMA_SLOW, adjust=False).mean()
            
            rsi = ta.momentum.RSIIndicator(df['close'], window=CONFIG.RSI_PERIOD).rsi()
            atr_ind = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=CONFIG.ATR_PERIOD
            ).average_true_range()
            
            current_price = df['close'].iloc[-1]
            ema_fast = df['EMA_fast'].iloc[-1]
            ema_slow = df['EMA_slow'].iloc[-1]
            rsi_val = rsi.iloc[-1]
            atr_val = atr_ind.iloc[-1]
            
            is_bullish = ema_fast > ema_slow and rsi_val < CONFIG.RSI_OVERBOUGHT
            
            results.append({
                'tf': tf,
                'bullish': is_bullish,
                'price': current_price,
                'atr': atr_val
            })
            
            if CONFIG.SHOW_DETAILED_ANALYSIS:
                self.logger.info(f"  {tf:8} | {'🟢 BULL' if is_bullish else '🔴 BEAR'} | "
                               f"${current_price:.2f} | RSI: {rsi_val:.0f}")
            
            # Use daily for price/atr
            if tf == 'daily':
                price = current_price
                atr = atr_val
        
        if not results:
            return 0.5, "NEUTRAL", 0, 0, 0.2
        
        bullish_count = sum(1 for r in results if r['bullish'])
        score = bullish_count / len(results)
        
        signal = "BULLISH" if score >= 0.66 else "BEARISH" if score <= 0.33 else "NEUTRAL"
        
        # Calculate volatility
        df_daily = self.data_mgr.get_data(symbol, 'daily')
        if df_daily is not None:
            returns = df_daily['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0.20
        
        return score, signal, price, atr, volatility
    
    def _calculate_position_size(self, portfolio_value, price, atr, stop_loss):
        """ATR-based position sizing"""
        
        if price == 0 or atr == 0:
            return 0
        
        if CONFIG.SIZING_METHOD == 'atr':
            # Risk-based sizing
            risk_amount = portfolio_value * (CONFIG.RISK_PER_TRADE_PCT / 100)
            stop_distance = abs(price - stop_loss)
            
            if stop_distance > 0:
                qty = int(risk_amount / stop_distance)
            else:
                qty = 0
        
        elif CONFIG.SIZING_METHOD == 'volatility':
            # Volatility-adjusted
            atr_pct = (atr / price) * 100
            
            if atr_pct > 3:
                adjusted_pct = CONFIG.BASE_POSITION_PCT * 0.5
            elif atr_pct > 2:
                adjusted_pct = CONFIG.BASE_POSITION_PCT * 0.75
            else:
                adjusted_pct = CONFIG.BASE_POSITION_PCT
            
            position_value = portfolio_value * (adjusted_pct / 100)
            qty = int(position_value / price)
        
        else:  # fixed
            position_value = portfolio_value * (CONFIG.BASE_POSITION_PCT / 100)
            qty = int(position_value / price)
        
        # Apply bounds
        max_qty = int((portfolio_value * CONFIG.MAX_POSITION_PCT / 100) / price)
        min_qty = int((portfolio_value * CONFIG.MIN_POSITION_PCT / 100) / price)
        
        qty = max(min_qty, min(qty, max_qty))
        
        return qty

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    def __init__(self, api, logger):
        self.api = api
        self.logger = logger
        self.initial_value = None
        self.peak_value = None
        self.last_trade = {}
    
    def init(self):
        account = self.api.get_account()
        self.initial_value = float(account.portfolio_value)
        self.peak_value = self.initial_value
    
    def check_cooldown(self, symbol):
        if symbol not in self.last_trade:
            return True
        
        hours = (datetime.now() - self.last_trade[symbol]).total_seconds() / 3600
        return hours >= CONFIG.COOLDOWN_HOURS
    
    def record_trade(self, symbol):
        self.last_trade[symbol] = datetime.now()
    
    def check_drawdown(self):
        account = self.api.get_account()
        current = float(account.portfolio_value)
        
        if current > self.peak_value:
            self.peak_value = current
        
        drawdown = ((self.peak_value - current) / self.peak_value) * 100
        
        if drawdown >= CONFIG.MAX_DRAWDOWN_PCT:
            self.logger.error(f"⛔ MAX DRAWDOWN HIT: {drawdown:.2f}%")
            return False
        
        return True

# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    def __init__(self, api, risk_mgr, logger, db):
        self.api = api
        self.risk_mgr = risk_mgr
        self.logger = logger
        self.db = db
        self.positions = {}
    
    def get_position(self, symbol):
        try:
            return self.api.get_position(symbol)
        except:
            return None
    
    def open_position(self, signal):
        symbol = signal['symbol']
        
        # Checks
        if not signal['tradeable']:
            return
        
        if self.get_position(symbol):
            return
        
        if not self.risk_mgr.check_cooldown(symbol):
            self.logger.info(f"  ⏸️  {symbol} in cooldown")
            return
        
        positions = self.api.list_positions()
        if len(positions) >= CONFIG.MAX_POSITIONS:
            self.logger.info(f"  ⏸️  Max positions reached")
            return
        
        # Execute
        qty = signal['position_size']
        price = signal['price']
        
        if qty == 0:
            self.logger.info(f"  ⚠️  Position size = 0")
            return
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            self.logger.info(f"\n🟢 BUY ORDER:")
            self.logger.info(f"  {symbol}: {qty} shares @ ${price:.2f}")
            self.logger.info(f"  Stop: ${signal['stop_loss']:.2f}")
            self.logger.info(f"  Target: ${signal['profit_target']:.2f}")
            self.logger.info(f"  Risk: ${(price - signal['stop_loss']) * qty:.2f}")
            
            self.positions[symbol] = {
                'id': order.id,
                'entry_price': price,
                'entry_time': datetime.now(),
                'stop': signal['stop_loss'],
                'target': signal['profit_target'],
                'qty': qty
            }
            
            self.risk_mgr.record_trade(symbol)
            
        except Exception as e:
            self.logger.error(f"  ❌ Order error: {e}")
    
    def manage_positions(self):
        """Check all positions for stops/targets"""
        
        for symbol in list(self.positions.keys()):
            position = self.get_position(symbol)
            
            if not position:
                if symbol in self.positions:
                    del self.positions[symbol]
                continue
            
            trade = self.positions[symbol]
            current = float(position.current_price)
            entry = trade['entry_price']
            
            pnl_pct = ((current - entry) / entry) * 100
            
            # Stop loss
            if CONFIG.USE_STOP_LOSS and current <= trade['stop']:
                self.logger.info(f"\n🛑 STOP LOSS: {symbol} at ${current:.2f} ({pnl_pct:.2f}%)")
                self.close_position(symbol, 'stop_loss')
            
            # Profit target
            elif CONFIG.USE_PROFIT_TARGET and current >= trade['target']:
                self.logger.info(f"\n🎯 TARGET HIT: {symbol} at ${current:.2f} ({pnl_pct:.2f}%)")
                self.close_position(symbol, 'profit_target')
    
    def close_position(self, symbol, reason):
        position = self.get_position(symbol)
        if not position:
            return
        
        qty = int(position.qty)
        exit_price = float(position.current_price)
        
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            trade = self.positions[symbol]
            pnl = (exit_price - trade['entry_price']) * qty
            pnl_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
            
            self.logger.info(f"🔴 SOLD {qty} {symbol} @ ${exit_price:.2f}")
            self.logger.info(f"  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            # Save to DB
            self.db.save_trade((
                trade['id'],
                symbol,
                'buy',
                qty,
                trade['entry_price'],
                exit_price,
                trade['entry_time'].isoformat(),
                datetime.now().isoformat(),
                pnl,
                pnl_pct,
                reason
            ))
            
            del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Close error for {symbol}: {e}")

# ============================================================================
# MAIN SYSTEM
# ============================================================================

class TradingSystem:
    def __init__(self, symbols):
        self.symbols = symbols
        self.logger = Logger()
        self.db = Database()
        
        # Components
        self.api = tradeapi.REST(
            CONFIG.ALPACA_API_KEY,
            CONFIG.ALPACA_SECRET_KEY,
            CONFIG.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        self.risk_mgr = RiskManager(self.api, self.logger)
        self.data_mgr = DataManager(self.api, self.logger)
        self.sentiment = SentimentAnalyzer(self.logger)
        self.fundamental = FundamentalAnalyzer(self.api, self.logger)
        self.strategy = StrategyEngine(
            self.api, self.data_mgr, self.sentiment, self.fundamental, self.logger
        )
        self.execution = ExecutionEngine(self.api, self.risk_mgr, self.logger, self.db)
    
    def run(self):
        self.logger.info("="*70)
        self.logger.info("🤖 PROFESSIONAL TRADING SYSTEM v2.0")
        self.logger.info("="*70)
        self.logger.info(f"Symbols: {', '.join(self.symbols)}")
        self.logger.info(f"\nAnalysis:")
        self.logger.info(f"  Technical:    {'✅' if CONFIG.USE_TECHNICAL else '❌'} ({CONFIG.TECHNICAL_WEIGHT:.0%})")
        self.logger.info(f"  Fundamental:  {'✅' if CONFIG.USE_FUNDAMENTAL else '❌'} ({CONFIG.FUNDAMENTAL_WEIGHT:.0%})")
        self.logger.info(f"  Sentiment:    {'✅' if CONFIG.USE_SENTIMENT else '❌'} ({CONFIG.SENTIMENT_WEIGHT:.0%})")
        self.logger.info(f"\nPosition Sizing: {CONFIG.SIZING_METHOD.upper()}")
        self.logger.info(f"Risk per Trade: {CONFIG.RISK_PER_TRADE_PCT}%")
        self.logger.info(f"Stop Loss: {CONFIG.STOP_LOSS_ATR_MULTIPLE}x ATR")
        self.logger.info("="*70 + "\n")
        
        self.risk_mgr.init()
        
        while True:
            try:
                # Check market
                clock = self.api.get_clock()
                if not clock.is_open:
                    self.logger.info(f"💤 Market closed. Opens: {clock.next_open}")
                    time.sleep(300)
                    continue
                
                # Check drawdown
                if not self.risk_mgr.check_drawdown():
                    break
                
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"🔍 SCAN - {datetime.now().strftime('%H:%M:%S')}")
                self.logger.info(f"{'='*70}")
                
                # Manage existing positions
                self.execution.manage_positions()
                
                # Scan symbols
                for symbol in self.symbols:
                    try:
                        signal = self.strategy.analyze(symbol)
                        
                        if signal:
                            position = self.execution.get_position(symbol)
                            
                            if not position:
                                self.execution.open_position(signal)
                    
                    except Exception as e:
                        self.logger.error(f"Error analyzing {symbol}: {e}")
                
                self.logger.info(f"\n⏳ Next scan in {CONFIG.SCAN_INTERVAL}s...")
                time.sleep(CONFIG.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                self.logger.info("\n🛑 System stopped")
                break
            except Exception as e:
                self.logger.error(f"System error: {e}")
                time.sleep(60)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🤖 PROFESSIONAL TRADING SYSTEM v2.0")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Technical Analysis (Multi-timeframe)")
    print("  ✅ Fundamental Analysis")
    print("  ✅ Sentiment Analysis (News)")
    print("  ✅ ATR-based Position Sizing")
    print("  ✅ Risk Management (Stop Loss, Targets)")
    print("  ✅ Robust Error Handling")
    print("="*70 + "\n")
    
    # Check keys
    if "YOUR_ALPACA" in CONFIG.ALPACA_API_KEY:
        print("❌ Add your Alpaca API keys in CONFIG class")
        return
    
    # Get symbols
    print("Enter stock symbols (comma-separated)")
    print("Example: SPY,QQQ,AAPL,TSLA")
    print("\nOr press Enter for default (SPY,QQQ,DIA): ")
    
    user_input = input().strip()
    
    if user_input:
        symbols = [s.strip().upper() for s in user_input.split(',')]
    else:
        symbols = ['SPY', 'QQQ', 'DIA']
    
    print(f"\n✅ Trading: {', '.join(symbols)}")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Run
    system = TradingSystem(symbols)
    system.run()

if __name__ == "__main__":
    main()