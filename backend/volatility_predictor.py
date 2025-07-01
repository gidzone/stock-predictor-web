import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class VolatilityPredictor:
    """
    Predicts future volatility and risk metrics for portfolio protection
    """
    
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.models = {}
        self.scalers = {}
        
    def calculate_risk_features(self, df):
        """Calculate features specifically for volatility prediction"""
        
        # Realized volatility at different timeframes
        for window in [5, 10, 20, 60]:
            df[f'realized_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        # GARCH-like features
        df['squared_returns'] = df['returns'] ** 2
        df['abs_returns'] = df['returns'].abs()
        
        # Volatility of volatility
        df['vol_of_vol'] = df['realized_vol_20'].rolling(20).std()
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(252 * df['log_hl_ratio'].rolling(20).var() / (4 * np.log(2)))
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(252 * (
            0.5 * df['log_hl_ratio']**2 - 
            (2*np.log(2) - 1) * df['log_oc_ratio']**2
        ).rolling(20).mean())
        
        # Volume-based features (volume spikes often precede volatility)
        df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_volatility'] = df['volume'].rolling(20).std()
        
        # Market microstructure
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)).abs() / df['close'].shift(1)
        
        # Technical indicators that predict volatility
        df['atr'] = self.calculate_atr(df)
        df['bollinger_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # Regime features
        df['volatility_regime'] = pd.qcut(df['realized_vol_20'], q=5, labels=[1,2,3,4,5])
        df['trend_strength'] = (df['close'] - df['sma_50']).abs() / df['sma_50']
        
        return df
    
    def calculate_atr(self, df, period=14):
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def prepare_volatility_data(self, df):
        """Prepare features and targets for volatility prediction"""
        
        # Add base features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['log_hl_ratio'] = np.log(df['high'] / df['low'])
        df['log_oc_ratio'] = np.log(df['close'] / df['open'])
        
        # Calculate risk features
        df = self.calculate_risk_features(df)
        
        # Target: future realized volatility
        df['target_vol_5d'] = df['returns'].shift(-5).rolling(5).std() * np.sqrt(252)
        df['target_vol_20d'] = df['returns'].shift(-20).rolling(20).std() * np.sqrt(252)
        
        # Target: maximum drawdown in next N days
        df['future_max_drawdown_5d'] = self.calculate_future_drawdown(df, 5)
        df['future_max_drawdown_20d'] = self.calculate_future_drawdown(df, 20)
        
        return df
    
    def calculate_future_drawdown(self, df, days):
        """Calculate maximum drawdown in the next N days"""
        future_drawdowns = []
        
        for i in range(len(df) - days):
            future_prices = df['close'].iloc[i:i+days].values
            peak = np.maximum.accumulate(future_prices)
            drawdown = (future_prices - peak) / peak
            max_dd = drawdown.min()
            future_drawdowns.append(max_dd)
        
        # Pad with NaN for the last 'days' entries
        future_drawdowns.extend([np.nan] * days)
        
        return pd.Series(future_drawdowns, index=df.index)
    
    def predict_risk_metrics(self, symbol, portfolio_value=None):
        """
        Predict various risk metrics for a symbol
        Returns volatility forecast, VaR, and risk scores
        """
        try:
            # Fetch recent data
            data = self.fetch_enhanced_data(symbol)
            if data is None:
                return None
            
            # Prepare features
            risk_data = self.prepare_volatility_data(data)
            
            # Feature columns for model
            feature_cols = [
                'realized_vol_5', 'realized_vol_10', 'realized_vol_20', 'realized_vol_60',
                'vol_of_vol', 'parkinson_vol', 'gk_vol', 'volume_surge',
                'high_low_spread', 'overnight_gap', 'atr', 'bollinger_width',
                'trend_strength'
            ]
            
            # Get latest features
            latest_features = risk_data[feature_cols].iloc[-1:].fillna(method='ffill')
            
            # Train or load model
            if symbol not in self.models:
                self.train_volatility_model(symbol, risk_data, feature_cols)
            
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            # Scale features
            scaled_features = scaler.transform(latest_features)
            
            # Predict
            vol_5d_pred = model['vol_5d'].predict(scaled_features)[0]
            vol_20d_pred = model['vol_20d'].predict(scaled_features)[0]
            dd_5d_pred = model['dd_5d'].predict(scaled_features)[0]
            dd_20d_pred = model['dd_20d'].predict(scaled_features)[0]
            
            # Calculate risk metrics
            current_price = float(data['close'].iloc[-1])
            current_vol = float(risk_data['realized_vol_20'].iloc[-1])
            
            # Value at Risk (95% confidence)
            var_5d = current_price * 1.645 * vol_5d_pred * np.sqrt(5/252)
            var_20d = current_price * 1.645 * vol_20d_pred * np.sqrt(20/252)
            
            # Risk score (0-100)
            vol_percentile = (risk_data['realized_vol_20'].rank(pct=True).iloc[-1] * 100)
            risk_score = self.calculate_risk_score(
                vol_percentile, 
                dd_5d_pred, 
                vol_5d_pred / current_vol
            )
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'current_volatility': current_vol * 100,  # As percentage
                'predicted_volatility_5d': vol_5d_pred * 100,
                'predicted_volatility_20d': vol_20d_pred * 100,
                'predicted_max_drawdown_5d': dd_5d_pred * 100,
                'predicted_max_drawdown_20d': dd_20d_pred * 100,
                'value_at_risk_5d': var_5d,
                'value_at_risk_20d': var_20d,
                'risk_score': risk_score,
                'risk_level': self.get_risk_level(risk_score),
                'volatility_trend': 'increasing' if vol_5d_pred > current_vol else 'decreasing',
                'risk_percentile': vol_percentile
            }
            
            # Add portfolio-specific metrics if portfolio value provided
            if portfolio_value:
                position_value = portfolio_value * 0.1  # Assume 10% position
                result['portfolio_var_5d'] = var_5d * (position_value / current_price)
                result['portfolio_var_20d'] = var_20d * (position_value / current_price)
                result['recommended_position_size'] = self.calculate_kelly_position(
                    expected_return=0.08,  # Could be from your prediction model
                    volatility=vol_5d_pred,
                    risk_free_rate=0.04
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting risk for {symbol}: {e}")
            return None
    
    def calculate_risk_score(self, vol_percentile, predicted_dd, vol_change_ratio):
        """Calculate overall risk score from 0-100"""
        # Weight different risk factors
        vol_score = vol_percentile * 0.4
        dd_score = min(abs(predicted_dd) * 1000, 100) * 0.4  # Scale drawdown to 0-100
        momentum_score = min(abs(vol_change_ratio - 1) * 100, 100) * 0.2
        
        return min(vol_score + dd_score + momentum_score, 100)
    
    def get_risk_level(self, risk_score):
        """Convert risk score to human-readable level"""
        if risk_score < 20:
            return "Very Low"
        elif risk_score < 40:
            return "Low"
        elif risk_score < 60:
            return "Moderate"
        elif risk_score < 80:
            return "High"
        else:
            return "Extreme"
    
    def calculate_kelly_position(self, expected_return, volatility, risk_free_rate=0.04):
        """Calculate optimal position size using Kelly Criterion"""
        # Simplified Kelly for continuous outcomes
        kelly_fraction = (expected_return - risk_free_rate) / (volatility ** 2)
        
        # Cap at 25% for safety
        return min(max(kelly_fraction, 0), 0.25)
    
    def train_volatility_model(self, symbol, data, feature_cols):
        """Train volatility prediction models"""
        # Prepare training data
        train_data = data[feature_cols + ['target_vol_5d', 'target_vol_20d', 
                                          'future_max_drawdown_5d', 'future_max_drawdown_20d']].dropna()
        
        if len(train_data) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Split features and targets
        X = train_data[feature_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train multiple models for different targets
        models = {}
        
        # 5-day volatility model
        models['vol_5d'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        models['vol_5d'].fit(X_scaled, train_data['target_vol_5d'])
        
        # 20-day volatility model
        models['vol_20d'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        models['vol_20d'].fit(X_scaled, train_data['target_vol_20d'])
        
        # Drawdown models
        models['dd_5d'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        models['dd_5d'].fit(X_scaled, train_data['future_max_drawdown_5d'])
        
        models['dd_20d'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        models['dd_20d'].fit(X_scaled, train_data['future_max_drawdown_20d'])
        
        # Store models and scaler
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        logger.info(f"Trained volatility models for {symbol}")
    
    def fetch_enhanced_data(self, symbol, days_back=250):
        """Fetch data with additional OHLC for volatility calculations"""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
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
            
            # Add basic technical indicators needed
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            
            # Bollinger Bands
            std_20 = data['close'].rolling(20).std()
            data['bb_upper'] = data['sma_20'] + (2 * std_20)
            data['bb_lower'] = data['sma_20'] - (2 * std_20)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None