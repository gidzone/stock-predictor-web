#!/usr/bin/env python3
"""
Flask Backend for Stock Predictor
Wraps your existing stock predictor and provides REST API endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import alpaca_trade_api as tradeapi
from walk_forward_backtest import WalkForwardBacktester
from datetime import datetime, timedelta
import json
import traceback
import logging

from walk_forward_backtest import WalkForwardBacktester
from portfolio_guardian import PortfolioGuardian, AlertSeverity
from volatility_predictor import VolatilityPredictor
from sentiment_analyzer import SocialSentimentAnalyzer


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS with environment variable
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=cors_origins)

# Load API credentials from environment variables
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Enhanced type conversion function
def safe_json_convert(obj):
    """
    Convert any object to JSON-serializable format
    Handles all numpy types, pandas types, and nested structures
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        # Handle NaN and infinite values
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (str, np.str_)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): safe_json_convert(value) for key, value in obj.items()}
    elif hasattr(obj, 'item'):  # numpy scalars
        return safe_json_convert(obj.item())
    elif hasattr(obj, 'tolist'):  # other numpy-like objects
        return obj.tolist()
    else:
        # Last resort: convert to string
        return str(obj)

def test_json_serialization(data, name="data"):
    """Test if data can be JSON serialized and show problematic fields"""
    try:
        json.dumps(data)
        print(f"✅ {name} is JSON serializable")
        return True
    except Exception as e:
        print(f"❌ {name} JSON error: {e}")
        
        # Debug each field
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    json.dumps({key: value})
                except Exception as field_error:
                    print(f"  Problem field '{key}': {type(value)} = {value}")
                    print(f"  Error: {field_error}")
        return False

class StockPredictorAPI:
    """Stock predictor optimized for web API"""
    
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
    def fetch_data(self, symbol, days_back=365):
        """Fetch stock data from Alpaca"""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            logger.info(f"Fetching {symbol} data from {start_date}")
            
            # Your fix: Don't specify end_date to avoid SIP restrictions
            bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Day, start=start_date)
            
            data = pd.DataFrame([{
                'date': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            } for bar in bars])
            
            if data.empty:
                return None
                
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            logger.info(f"Got {len(data)} days of {symbol} data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def create_features(self, data):
        """Create ML features"""
        df = data.copy()
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_vs_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target
        df['target'] = df['close'].shift(-1)
        
        return df
    
    def train_and_predict(self, symbol, days_back=365):
        """Complete prediction workflow - optimized for web"""
        try:
            # Fetch data
            data = self.fetch_data(symbol, days_back)
            if data is None:
                return None
            
            # Create features
            features_df = self.create_features(data)
            
            # Prepare data
            feature_cols = [col for col in features_df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
            
            df_clean = features_df.dropna()
            
            if len(df_clean) < 50:  # Need minimum data
                return None
            
            X = df_clean[feature_cols]
            y = df_clean['target']
            
            # Split: 80% train, 20% test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model (smaller for web speed)
            model = RandomForestRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Metrics
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            
            # Prepare data for frontend
            result = {
                'symbol': symbol,
                'current_price': float(data['close'].iloc[-1]),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'accuracy_pct': float(max(0, (1 - test_mae / data['close'].iloc[-1]) * 100)),
                'data_points': len(data),
                'train_size': len(X_train),
                'test_size': len(X_test),
                
                # Chart data
                'chart_data': {
                    'dates': [d.isoformat() for d in data.index[-len(test_pred):].tolist()],
                    'actual_prices': y_test.tolist(),
                    'predicted_prices': test_pred.tolist(),
                    'historical_prices': data['close'].tail(100).tolist(),
                    'historical_dates': [d.isoformat() for d in data.index[-100:].tolist()]
                },
                
                # Feature importance (top 5)
                'feature_importance': [
                    {
                        'feature': feature_cols[i],
                        'importance': float(model.feature_importances_[i])
                    }
                    for i in np.argsort(model.feature_importances_)[-5:][::-1]
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {traceback.format_exc()}")
            return None

# Initialize predictor
predictor = None

# Validate environment variables
if not API_KEY or not SECRET_KEY:
    logger.error("Missing required environment variables: ALPACA_API_KEY and/or ALPACA_SECRET_KEY")
    logger.error("Please check your .env file")
else:
    try:
        predictor = StockPredictorAPI(API_KEY, SECRET_KEY, BASE_URL)
        logger.info("✅ Stock predictor initialized successfully")
        logger.info(f"🔗 Using Alpaca endpoint: {BASE_URL}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        predictor = None

guardian = None
if predictor:
    try:
        # Initialize Portfolio Guardian
        # Note: Reddit credentials are optional - will use mock data without them
        reddit_creds = {
            'client_id': os.getenv('REDDIT_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET')
        }
        
        guardian = PortfolioGuardian(
            API_KEY, SECRET_KEY, BASE_URL,
            reddit_credentials=reddit_creds if reddit_creds['client_id'] else None
        )
        logger.info("✅ Portfolio Guardian initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Portfolio Guardian: {e}")
        guardian = None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_ready': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict stock price"""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized. Check API credentials.'}), 500
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        logger.info(f"Predicting {symbol}")
        
        result = predictor.train_and_predict(symbol)
        
        if result is None:
            return jsonify({'error': f'Could not fetch or process data for {symbol}'}), 404
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare multiple stocks"""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized. Check API credentials.'}), 500
    
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols or len(symbols) > 5:  # Limit to 5 for performance
            return jsonify({'error': 'Provide 1-5 symbols'}), 400
        
        results = []
        for symbol in symbols:
            symbol = symbol.upper().strip()
            logger.info(f"Comparing {symbol}")
            
            result = predictor.train_and_predict(symbol)
            if result:
                # Simplified result for comparison
                results.append({
                    'symbol': result['symbol'],
                    'current_price': result['current_price'],
                    'test_rmse': result['test_rmse'],
                    'test_mae': result['test_mae'],
                    'accuracy_pct': result['accuracy_pct']
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Comparison error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/popular-symbols', methods=['GET'])
def popular_symbols():
    """Get popular stock symbols"""
    symbols = {
        'Tech': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META'],
        'ETFs': ['SPY', 'QQQ', 'VTI', 'IWM', 'NVDL', 'TQQQ'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS'],
        'Popular': ['AAPL', 'TSLA', 'GOOGL', 'NVDL', 'SPY']
    }
    
    return jsonify(symbols)
# Updated quick validation endpoint
@app.route('/api/quick-validation', methods=['POST'])
def quick_validation():
    """Quick validation on a single symbol with robust JSON handling"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'SPY').upper()
        
        from walk_forward_backtest import WalkForwardBacktester
        backtester = WalkForwardBacktester(API_KEY, SECRET_KEY, BASE_URL)
        
        logger.info(f"Quick validation for {symbol}")
        metrics = backtester.walk_forward_backtest(symbol)
        
        if not metrics or 'error' in metrics:
            return jsonify({'error': f'Could not validate {symbol}'}), 400
        
        print(f"🔍 Raw metrics types:")
        for key, value in metrics.items():
            if key != 'results_df':  # Skip DataFrame
                print(f"  {key}: {type(value)} = {value}")
        
        # Step 1: Extract and convert individual fields with explicit type checking
        symbol_str = str(symbol)
        total_return = float(metrics['total_return']) if not np.isnan(metrics['total_return']) else 0.0
        sharpe_ratio = float(metrics['sharpe_ratio']) if not np.isnan(metrics['sharpe_ratio']) else 0.0
        win_rate = float(metrics['win_rate']) if not np.isnan(metrics['win_rate']) else 0.0
        max_drawdown = float(metrics['max_drawdown']) if not np.isnan(metrics['max_drawdown']) else 0.0
        final_value = float(metrics['final_value']) if not np.isnan(metrics['final_value']) else 5000.0
        kelly_fraction = float(metrics['kelly_fraction']) if not np.isnan(metrics['kelly_fraction']) else 0.0
        
        # Step 2: Calculate has_edge as pure Python boolean
        has_edge_calc = (sharpe_ratio > 1.0 and total_return > 0.0 and win_rate > 0.5)
        has_edge = bool(has_edge_calc)  # Ensure it's a Python bool
        
        # Step 3: Create recommendation as string
        recommendation_str = "✅ TRADEABLE" if has_edge else "❌ AVOID"
        
        # Step 4: Build result with explicit Python types only
        result = {
            'symbol': symbol_str,
            'has_edge': has_edge,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'recommendation': recommendation_str,
            'kelly_position': kelly_fraction
        }
        
        print(f"🔍 Result types:")
        for key, value in result.items():
            print(f"  {key}: {type(value)} = {value}")
        
        # Step 5: Test JSON serialization before returning
        if not test_json_serialization(result, "quick validation result"):
            # Fallback: apply safe conversion
            result = safe_json_convert(result)
        
        # Step 6: Final JSON test
        json_string = json.dumps(result)
        print(f"✅ JSON serialization successful: {len(json_string)} characters")
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Quick validation error: {str(e)}"
        logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

# Updated full backtest endpoint
@app.route('/api/backtest', methods=['POST'])
def backtest():
    """Run walk-forward backtest with robust JSON handling"""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized. Check API credentials.'}), 500
    
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['AAPL', 'SPY', 'QQQ'])
        
        if len(symbols) > 10:
            return jsonify({'error': 'Maximum 10 symbols allowed'}), 400
        
        logger.info(f"Running backtests for {symbols}")
        
        from walk_forward_backtest import WalkForwardBacktester
        backtester = WalkForwardBacktester(API_KEY, SECRET_KEY, BASE_URL)
        
        # Run backtests
        results = backtester.test_multiple_symbols(symbols)
        
        if not results:
            return jsonify({'error': 'No successful backtests completed'}), 500
        
        # Process results with explicit type conversion
        formatted_results = []
        profitable_count = 0
        edge_count = 0
        
        for symbol, metrics in results.items():
            print(f"🔍 Processing {symbol} metrics...")
            
            # Convert each field explicitly
            symbol_str = str(symbol)
            total_return = float(metrics['total_return']) if not np.isnan(metrics['total_return']) else 0.0
            annual_return = float(metrics['annual_return']) if not np.isnan(metrics['annual_return']) else 0.0
            sharpe_ratio = float(metrics['sharpe_ratio']) if not np.isnan(metrics['sharpe_ratio']) else 0.0
            win_rate = float(metrics['win_rate']) if not np.isnan(metrics['win_rate']) else 0.0
            max_drawdown = float(metrics['max_drawdown']) if not np.isnan(metrics['max_drawdown']) else 0.0
            total_trades = int(metrics['total_trades'])
            profit_factor = float(metrics['profit_factor']) if not np.isnan(metrics['profit_factor']) else 0.0
            kelly_fraction = float(metrics['kelly_fraction']) if not np.isnan(metrics['kelly_fraction']) else 0.0
            final_value = float(metrics['final_value']) if not np.isnan(metrics['final_value']) else 5000.0
            time_in_market = float(metrics['time_in_market']) if not np.isnan(metrics['time_in_market']) else 0.0
            
            # Calculate booleans as pure Python types
            is_profitable = bool(total_return > 0.0)
            has_edge = bool(sharpe_ratio > 1.0 and total_return > 0.1 and win_rate > 0.5)
            
            if is_profitable:
                profitable_count += 1
            if has_edge:
                edge_count += 1
            
            # Build result record
            result_record = {
                'symbol': symbol_str,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'profit_factor': profit_factor,
                'kelly_fraction': kelly_fraction,
                'final_value': final_value,
                'is_profitable': is_profitable,
                'has_edge': has_edge,
                'time_in_market': time_in_market
            }
            
            # Test this record
            if not test_json_serialization(result_record, f"{symbol} record"):
                result_record = safe_json_convert(result_record)
            
            formatted_results.append(result_record)
        
        # Sort by Sharpe ratio
        formatted_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        # Build summary
        best_performer = formatted_results[0] if formatted_results else None
        target_achieved = bool(best_performer and best_performer['final_value'] >= 50000.0)
        
        summary = {
            'total_tested': int(len(symbols)),
            'successful_backtests': int(len(formatted_results)),
            'profitable_strategies': int(profitable_count),
            'strategies_with_edge': int(edge_count),
            'target_achieved': target_achieved,
            'best_performer': str(best_performer['symbol']) if best_performer else None,
            'best_return': float(best_performer['total_return']) if best_performer else 0.0,
            'recommendation': str(get_trading_recommendation(formatted_results))
        }
        
        # Build final response
        response_data = {
            'results': formatted_results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Final JSON test
        if not test_json_serialization(response_data, "full response"):
            response_data = safe_json_convert(response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

def get_trading_recommendation(results):
    """Generate trading recommendation - returns pure string"""
    if not results:
        return "No valid backtests completed. Check your symbols and data availability."
    
    best = results[0]
    edge_strategies = [r for r in results if r['has_edge']]
    
    if not edge_strategies:
        return "❌ NO PROFITABLE EDGE DETECTED. Do not trade with real money."
    
    if best['final_value'] >= 50000:
        return f"🎉 STRONG EDGE DETECTED! {best['symbol']} could turn $5K into ${best['final_value']:,.0f}."
    
    if best['sharpe_ratio'] > 1.5 and best['total_return'] > 0.2:
        return f"📈 MODERATE EDGE DETECTED. {best['symbol']} shows {best['total_return']:.1%} returns."
    
    if len(edge_strategies) >= 2:
        return f"⚖️ DIVERSIFICATION OPPORTUNITY. {len(edge_strategies)} symbols show positive edge."
    
    return "⚠️ WEAK EDGE. Results are marginal."

# Debug endpoint to test JSON serialization
@app.route('/api/debug-json', methods=['GET'])
def debug_json():
    """Debug endpoint to test JSON serialization"""
    test_data = {
        'python_bool': True,
        'numpy_bool': np.bool_(True),
        'python_int': 42,
        'numpy_int': np.int64(42),
        'python_float': 3.14,
        'numpy_float': np.float64(3.14),
        'numpy_nan': np.nan,
        'python_str': "hello",
        'numpy_str': np.str_("world")
    }
    
    print("🔍 Testing JSON serialization of different types:")
    for key, value in test_data.items():
        print(f"  {key}: {type(value)} = {value}")
        try:
            json.dumps({key: value})
            print(f"    ✅ Serializable")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Convert and test
    safe_data = safe_json_convert(test_data)
    return jsonify(safe_data)

def get_trading_recommendation(results):
    """Generate trading recommendation - returns pure string"""
    if not results:
        return "No valid backtests completed. Check your symbols and data availability."
    
    best = results[0]
    edge_strategies = [r for r in results if r['has_edge']]
    
    if not edge_strategies:
        return "❌ NO PROFITABLE EDGE DETECTED. Do not trade with real money."
    
    if best['final_value'] >= 50000:
        return f"🎉 STRONG EDGE DETECTED! {best['symbol']} could turn $5K into ${best['final_value']:,.0f}."
    
    if best['sharpe_ratio'] > 1.5 and best['total_return'] > 0.2:
        return f"📈 MODERATE EDGE DETECTED. {best['symbol']} shows {best['total_return']:.1%} returns."
    
    if len(edge_strategies) >= 2:
        return f"⚖️ DIVERSIFICATION OPPORTUNITY. {len(edge_strategies)} symbols show positive edge."
    
    return "⚠️ WEAK EDGE. Results are marginal."

@app.route('/api/analyze-risk', methods=['POST'])
def analyze_risk():
    """Analyze risk for a single position"""
    if not guardian:
        return jsonify({'error': 'Portfolio Guardian not initialized'}), 500
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        position_size = data.get('position_size')
        purchase_price = data.get('purchase_price')
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        logger.info(f"Analyzing risk for {symbol}")
        
        # Perform analysis
        analysis = guardian.analyze_position(symbol, position_size, purchase_price)
        
        if not analysis:
            return jsonify({'error': f'Could not analyze {symbol}'}), 404
        
        # Convert alerts to JSON-serializable format
        if 'alerts' in analysis:
            analysis['alerts'] = [
                {
                    'symbol': alert.symbol,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'action_required': alert.action_required,
                    'data': alert.data
                }
                for alert in analysis['alerts']
            ]
        
        return jsonify(safe_json_convert(analysis))
        
    except Exception as e:
        logger.error(f"Risk analysis error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/monitor-portfolio', methods=['POST'])
def monitor_portfolio():
    """Monitor entire portfolio for risks"""
    if not guardian:
        return jsonify({'error': 'Portfolio Guardian not initialized'}), 500
    
    try:
        data = request.get_json()
        portfolio = data.get('portfolio', [])
        
        if not portfolio:
            return jsonify({'error': 'Portfolio is required'}), 400
        
        logger.info(f"Monitoring portfolio with {len(portfolio)} positions")
        
        # Monitor portfolio
        results = guardian.monitor_portfolio(portfolio)
        
        # Convert alerts to JSON-serializable format
        if 'alerts' in results:
            results['alerts'] = [
                {
                    'symbol': alert.symbol,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'action_required': alert.action_required,
                    'data': alert.data
                }
                for alert in results['alerts']
            ]
        
        return jsonify(safe_json_convert(results))
        
    except Exception as e:
        logger.error(f"Portfolio monitoring error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/sentiment/<symbol>', methods=['GET'])
def get_sentiment(symbol):
    """Get social sentiment for a symbol"""
    if not guardian:
        return jsonify({'error': 'Portfolio Guardian not initialized'}), 500
    
    try:
        symbol = symbol.upper().strip()
        logger.info(f"Getting sentiment for {symbol}")
        
        # Get sentiment analysis
        sentiment = guardian.sentiment_analyzer.analyze_symbol_sentiment(symbol)
        
        if not sentiment:
            return jsonify({'error': f'Could not analyze sentiment for {symbol}'}), 404
        
        return jsonify(safe_json_convert(sentiment))
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/market-pulse', methods=['GET'])
def market_pulse():
    """Get overall market sentiment and trending tickers"""
    if not guardian:
        return jsonify({'error': 'Portfolio Guardian not initialized'}), 500
    
    try:
        logger.info("Getting market pulse")
        
        pulse = guardian.get_market_pulse()
        
        # Also get trending ticker details
        trending_details = []
        for ticker in pulse['trending_tickers'][:5]:  # Top 5
            try:
                # Quick risk check for each trending ticker
                risk_metrics = guardian.volatility_predictor.predict_risk_metrics(ticker)
                if risk_metrics:
                    trending_details.append({
                        'symbol': ticker,
                        'current_price': risk_metrics['current_price'],
                        'volatility': risk_metrics['current_volatility'],
                        'risk_score': risk_metrics['risk_score'],
                        'risk_level': risk_metrics['risk_level']
                    })
            except:
                continue
        
        pulse['trending_details'] = trending_details
        
        return jsonify(safe_json_convert(pulse))
        
    except Exception as e:
        logger.error(f"Market pulse error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/volatility-forecast', methods=['POST'])
def volatility_forecast():
    """Get volatility forecast for a symbol"""
    if not guardian:
        return jsonify({'error': 'Portfolio Guardian not initialized'}), 500
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        logger.info(f"Getting volatility forecast for {symbol}")
        
        # Get volatility predictions
        risk_metrics = guardian.volatility_predictor.predict_risk_metrics(symbol)
        
        if not risk_metrics:
            return jsonify({'error': f'Could not forecast volatility for {symbol}'}), 404
        
        return jsonify(safe_json_convert(risk_metrics))
        
    except Exception as e:
        logger.error(f"Volatility forecast error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Stock Predictor API Server")
    print("=" * 40)
    
    # Check environment variables
    if not API_KEY or not SECRET_KEY:
        print("❌ Missing environment variables!")
        print("📝 Please create a .env file with:")
        print("   ALPACA_API_KEY=your_api_key")
        print("   ALPACA_SECRET_KEY=your_secret_key")
        print("💡 See .env.example for template")
    else:
        print("✅ Environment variables loaded")
        print(f"🔑 API Key: {API_KEY[:8]}...")
        print(f"🔗 Base URL: {BASE_URL}")
    
    # Get port from environment or default to 5000
    port = int(os.getenv('FLASK_PORT', 5040))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"🌐 Starting server on http://localhost:{port}")
    print("📊 Endpoints:")
    print("   GET  /api/health")
    print("   POST /api/predict")
    print("   POST /api/compare")
    print("   GET  /api/popular-symbols")
    print("   POST /api/backtest")
    print("   POST /api/quick-validation")
    print("\n🛡️  Portfolio Guardian Endpoints:")
    print("   POST /api/analyze-risk")
    print("   POST /api/monitor-portfolio")
    print("   GET  /api/sentiment/<symbol>")
    print("   GET  /api/market-pulse")
    print("   POST /api/volatility-forecast")
    
    app.run(debug=debug, host='0.0.0.0', port=port)