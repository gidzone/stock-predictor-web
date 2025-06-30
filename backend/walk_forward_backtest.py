#!/usr/bin/env python3
"""
Walk-Forward Backtesting Module
Add this to your existing backend to test if you actually have an edge
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    """
    Realistic backtesting with proper validation and transaction costs
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.transaction_cost = 0.001  # 0.1% round-trip (realistic for retail)
        self.min_signal_threshold = 0.002  # Need 0.2% predicted move to trade
        
    def fetch_intraday_data(self, symbol: str, days_back: int = 60) -> pd.DataFrame:
        """
        Fetch higher frequency data for better backtesting
        Using 1-hour bars for more signals than daily but less noise than 15-min
        """
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            logger.info(f"Fetching {symbol} hourly data from {start_date}")
            
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start_date
            )
            
            data = pd.DataFrame([{
                'timestamp': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            } for bar in bars])
            
            if data.empty:
                return None
            
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            # Add trading features
            return self.create_trading_features(data)
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def create_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features optimized for short-term mean reversion
        """
        # Returns and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std()  # 24-hour volatility
        
        # Mean reversion signals
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # Momentum indicators
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['momentum_12h'] = df['close'] / df['close'].shift(12) - 1
        df['momentum_24h'] = df['close'] / df['close'].shift(24) - 1
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_spike'] = (df['volume'] > df['volume_sma'] * 1.5).astype(int)
        
        # Volatility features
        df['bb_position'] = self.bollinger_position(df['close'])
        df['volatility_rank'] = df['volatility'].rolling(100).rank(pct=True)
        
        # Time-based features (important for intraday)
        df['hour'] = df.index.hour
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        df['is_power_hour'] = ((df['hour'] >= 15) & (df['hour'] <= 16)).astype(int)
        
        # Gap and range features
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['intraday_range'] = (df['high'] - df['low']) / df['open']
        
        # Target: next hour return
        df['target'] = df['returns'].shift(-1)
        
        return df.dropna()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands (0-1 scale)"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (prices - lower) / (upper - lower)
    
    def walk_forward_backtest(self, symbol: str, training_window: int = 500, 
                             retrain_frequency: int = 24) -> Dict:
        """
        Walk-forward backtesting with realistic constraints
        """
        logger.info(f"Starting walk-forward backtest for {symbol}")
        
        # Get data
        data = self.fetch_intraday_data(symbol, days_back=90)
        if data is None or len(data) < training_window + 100:
            return None
        
        # Feature columns
        feature_cols = [
            'price_vs_sma20', 'price_vs_sma50', 'rsi', 'momentum_12h', 'momentum_24h',
            'volume_ratio', 'volume_spike', 'bb_position', 'volatility_rank',
            'hour', 'is_market_open', 'is_power_hour', 'intraday_range'
        ]
        
        results = []
        model = None
        last_retrain = 0
        
        # Walk forward through time
        for i in range(training_window, len(data) - 1):
            
            # Retrain model periodically
            if i - last_retrain >= retrain_frequency or model is None:
                train_data = data.iloc[i-training_window:i]
                
                # Prepare training data
                X_train = train_data[feature_cols].fillna(0)
                y_train = train_data['target'].fillna(0)
                
                # Remove extreme outliers
                y_train = y_train.clip(-0.1, 0.1)  # Cap at ±10% hourly moves
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                last_retrain = i
                
                logger.info(f"Retrained model at position {i}")
            
            # Make prediction
            current_row = data.iloc[i]
            features = current_row[feature_cols].fillna(0).values.reshape(1, -1)
            predicted_return = model.predict(features)[0]
            
            # Actual return
            actual_return = data.iloc[i + 1]['returns']
            
            # Trading decision logic
            signal_strength = abs(predicted_return)
            
            # Only trade if:
            # 1. Signal > minimum threshold
            # 2. During market hours
            # 3. Not extreme volatility
            trade_conditions = (
                signal_strength > self.min_signal_threshold and
                current_row['is_market_open'] == 1 and
                current_row['volatility_rank'] < 0.95  # Not extreme volatility
            )
            
            if trade_conditions:
                # Position sizing based on signal strength
                base_position = 1.0
                confidence_multiplier = min(2.0, signal_strength / self.min_signal_threshold)
                position_size = base_position * confidence_multiplier
                
                # Direction
                position = position_size if predicted_return > 0 else -position_size
                
                # Calculate P&L with transaction costs
                gross_return = actual_return * position
                net_return = gross_return - self.transaction_cost
                
            else:
                position = 0
                net_return = 0
                gross_return = 0
            
            # Store results
            results.append({
                'timestamp': data.index[i + 1],
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'position': position,
                'gross_return': gross_return,
                'net_return': net_return,
                'signal_strength': signal_strength,
                'traded': trade_conditions
            })
        
        # Convert to DataFrame and calculate metrics
        df_results = pd.DataFrame(results)
        return self.calculate_backtest_metrics(df_results, symbol)
    
    def calculate_backtest_metrics(self, df_results: pd.DataFrame, symbol: str) -> Dict:
        """
        Calculate comprehensive backtest metrics
        """
        # Basic stats
        total_trades = len(df_results[df_results['traded']])
        winning_trades = len(df_results[
            (df_results['traded']) & (df_results['net_return'] > 0)
        ])
        
        if total_trades == 0:
            return {
                'symbol': symbol,
                'error': 'No trades generated',
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        win_rate = winning_trades / total_trades
        
        # Returns analysis
        returns = df_results['net_return']
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Risk metrics
        annual_return = (1 + total_return) ** (365 * 24 / len(df_results)) - 1
        return_volatility = returns.std() * np.sqrt(365 * 24)  # Annualized
        sharpe_ratio = annual_return / return_volatility if return_volatility > 0 else 0
        
        # Drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        trade_returns = df_results[df_results['traded']]['net_return']
        avg_win = trade_returns[trade_returns > 0].mean() if len(trade_returns[trade_returns > 0]) > 0 else 0
        avg_loss = trade_returns[trade_returns < 0].mean() if len(trade_returns[trade_returns < 0]) > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else 0
        
        # Kelly criterion for optimal position sizing
        if avg_loss != 0:
            kelly_fraction = (avg_win * win_rate + avg_loss * (1 - win_rate)) / avg_loss
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
        else:
            kelly_fraction = 0
        
        # Time in market
        time_in_market = len(df_results[df_results['position'] != 0]) / len(df_results)
        
        # Calculate what $5K would become
        final_value = 5000 * (1 + total_return)
        
        metrics = {
            'symbol': symbol,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'kelly_fraction': kelly_fraction,
            'time_in_market': time_in_market,
            'final_value': final_value,
            'results_df': df_results
        }
        
        return metrics
    
    def test_multiple_symbols(self, symbols: List[str]) -> Dict:
        """
        Test multiple symbols and rank by performance
        """
        logger.info(f"Testing {len(symbols)} symbols...")
        
        results = {}
        
        for symbol in symbols:
            try:
                metrics = self.walk_forward_backtest(symbol)
                if metrics and 'error' not in metrics:
                    results[symbol] = metrics
                    
                    logger.info(f"{symbol}: {metrics['total_return']:.1%} return, "
                               f"{metrics['sharpe_ratio']:.2f} Sharpe, "
                               f"{metrics['win_rate']:.1%} win rate")
                else:
                    logger.warning(f"Failed to backtest {symbol}")
                    
            except Exception as e:
                logger.error(f"Error testing {symbol}: {e}")
                continue
        
        return results
    
    def print_summary_report(self, results: Dict):
        """
        Print comprehensive summary of all backtests
        """
        if not results:
            print("❌ No successful backtests")
            return
        
        print("\n" + "="*80)
        print("WALK-FORWARD BACKTEST RESULTS")
        print("="*80)
        
        # Sort by Sharpe ratio
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['sharpe_ratio'], 
                               reverse=True)
        
        print(f"{'Symbol':<8} {'Return':<8} {'Annual':<8} {'Sharpe':<7} {'Win%':<6} {'Trades':<7} {'$5K→':<8}")
        print("-" * 80)
        
        profitable_count = 0
        
        for symbol, metrics in sorted_results:
            if metrics['total_return'] > 0:
                profitable_count += 1
            
            print(f"{symbol:<8} {metrics['total_return']:>7.1%} "
                  f"{metrics['annual_return']:>7.1%} "
                  f"{metrics['sharpe_ratio']:>6.2f} "
                  f"{metrics['win_rate']:>5.1%} "
                  f"{metrics['total_trades']:>6} "
                  f"${metrics['final_value']:>7,.0f}")
        
        print("-" * 80)
        print(f"Profitable strategies: {profitable_count}/{len(results)}")
        
        # Best performer analysis
        if sorted_results:
            best_symbol, best_metrics = sorted_results[0]
            print(f"\n🏆 BEST PERFORMER: {best_symbol}")
            print(f"   Total Return: {best_metrics['total_return']:.1%}")
            print(f"   Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {best_metrics['max_drawdown']:.1%}")
            print(f"   Win Rate: {best_metrics['win_rate']:.1%}")
            print(f"   Profit Factor: {best_metrics['profit_factor']:.2f}")
            print(f"   Kelly Position: {best_metrics['kelly_fraction']:.1%}")
            print(f"   $5,000 → ${best_metrics['final_value']:,.0f}")
            
            if best_metrics['final_value'] >= 50000:
                print("🎉 TARGET ACHIEVED! You have a profitable edge!")
            else:
                multiplier_needed = 50000 / best_metrics['final_value']
                print(f"📈 Need {multiplier_needed:.1f}x improvement to reach $50K target")

# Integration with your existing Flask app
def add_backtesting_endpoint(app, predictor_instance):
    """
    Add backtesting endpoint to your existing Flask app
    """
    
    @app.route('/api/backtest', methods=['POST'])
    def backtest():
        """Run walk-forward backtest on symbols"""
        try:
            data = request.get_json()
            symbols = data.get('symbols', ['AAPL', 'SPY', 'QQQ'])
            
            # Initialize backtester with same credentials
            backtester = WalkForwardBacktester(
                api_key=predictor_instance.api.api_key,
                secret_key=predictor_instance.api.secret_key,
                base_url=predictor_instance.api.base_url
            )
            
            # Run backtests
            results = backtester.test_multiple_symbols(symbols)
            
            # Format for frontend
            formatted_results = []
            for symbol, metrics in results.items():
                formatted_results.append({
                    'symbol': symbol,
                    'total_return': metrics['total_return'],
                    'annual_return': metrics['annual_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'win_rate': metrics['win_rate'],
                    'max_drawdown': metrics['max_drawdown'],
                    'total_trades': metrics['total_trades'],
                    'final_value': metrics['final_value'],
                    'has_edge': metrics['sharpe_ratio'] > 1.0 and metrics['total_return'] > 0
                })
            
            # Sort by Sharpe ratio
            formatted_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            
            return jsonify({
                'results': formatted_results,
                'summary': {
                    'total_tested': len(symbols),
                    'profitable': len([r for r in formatted_results if r['total_return'] > 0]),
                    'with_edge': len([r for r in formatted_results if r['has_edge']])
                }
            })
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test with your credentials
    backtester = WalkForwardBacktester(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    )
    
    # Test popular symbols
    test_symbols = ['AAPL', 'SPY', 'QQQ', 'NVDL', 'TQQQ', 'TSLA', 'MSFT']
    results = backtester.test_multiple_symbols(test_symbols)
    backtester.print_summary_report(results)