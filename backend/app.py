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
from datetime import datetime, timedelta
import json
import traceback
import logging

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
    
    app.run(debug=debug, host='0.0.0.0', port=port)