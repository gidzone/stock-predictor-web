import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import warnings
import json
import argparse
import logging
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalStockPredictor:
    """
    Universal Stock Price Predictor using LSTM
    Can predict any stock/ETF/crypto that's available on Alpaca Markets
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = 'https://paper-api.alpaca.markets'):
        """
        Initialize the Universal Stock Predictor
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca base URL
        """
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.models = {}  # Store models for different symbols
        self.scalers = {}  # Store scalers for different symbols
        self.data_cache = {}  # Cache data for different symbols
        
    def get_available_symbols(self, search_term: str = None) -> List[str]:
        """
        Get list of available symbols from Alpaca
        
        Args:
            search_term: Optional search term to filter symbols
            
        Returns:
            List of available symbols
        """
        try:
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            symbols = [asset.symbol for asset in assets if asset.tradable]
            
            if search_term:
                symbols = [s for s in symbols if search_term.upper() in s.upper()]
                
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is available for trading
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid and tradable
        """
        try:
            asset = self.api.get_asset(symbol)
            return asset.tradable and asset.status == 'active'
        except:
            return False
    
    def fetch_data(self, 
                   symbol: str, 
                   start_date: str = None, 
                   end_date: str = None,
                   timeframe: str = 'Day') -> Optional[pd.DataFrame]:
        """
        Fetch historical data for any symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'NVDL', 'SPY')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe ('Day', 'Hour', 'Minute')
            
        Returns:
            DataFrame with historical data
        """
        # Set default dates
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
       # if end_date is None:
        #    end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Validate symbol first
        if not self.validate_symbol(symbol):
            logger.error(f"Symbol {symbol} is not valid or not tradable")
            return None
            
        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}...")
        
        try:
            # Map timeframe string to Alpaca TimeFrame
            timeframe_map = {
                'Day': tradeapi.TimeFrame.Day,
                'Hour': tradeapi.TimeFrame.Hour,
                'Minute': tradeapi.TimeFrame.Minute
            }
            
            bars = self.api.get_bars(
                symbol,
                timeframe_map[timeframe],
                start=start_date,
                end=end_date,
                adjustment='raw'
            )
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            } for bar in bars])
            
            if df.empty:
                logger.error(f"No data returned for {symbol}")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Cache the data
            self.data_cache[symbol] = df
            
            logger.info(f"Successfully fetched {len(df)} periods of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving average
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['price_change'] = df['close'].pct_change()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def prepare_data(self, 
                     symbol: str, 
                     features: List[str] = None,
                     sequence_length: int = 60,
                     train_split: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            symbol: Stock symbol
            features: List of features to use
            sequence_length: Number of time steps to look back
            train_split: Fraction of data for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if symbol not in self.data_cache:
            raise ValueError(f"No data available for {symbol}. Please fetch data first.")
        
        data = self.data_cache[symbol]
        
        # Default features if none provided
        if features is None:
            features = ['close', 'volume', 'sma_20', 'rsi', 'macd', 'volatility']
        
        # Ensure all features exist in data
        available_features = [f for f in features if f in data.columns]
        if len(available_features) != len(features):
            missing = set(features) - set(available_features)
            logger.warning(f"Missing features for {symbol}: {missing}")
            features = available_features
        
        # Select and scale features
        feature_data = data[features].values
        
        # Initialize scaler for this symbol
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(feature_data)
        self.scalers[symbol] = scaler
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict close price (first feature)
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_index = int(len(X) * train_split)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        logger.info(f"Data prepared for {symbol}: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, 
                    symbol: str,
                    X_train: np.ndarray,
                    lstm_units: List[int] = [64, 32],
                    dropout_rate: float = 0.2,
                    learning_rate: float = 0.001) -> Sequential:
        """
        Build LSTM model for a specific symbol
        
        Args:
            symbol: Stock symbol
            X_train: Training data to determine input shape
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=lstm_units[0], 
                      return_sequences=len(lstm_units) > 1, 
                      input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(units=lstm_units[i], return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        # Store model for this symbol
        self.models[symbol] = model
        
        logger.info(f"Model built for {symbol}")
        return model
    
    def train_model(self, 
                    symbol: str,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 32,
                    validation_split: float = 0.2,
                    verbose: int = 1):
        """
        Train the model for a specific symbol
        """
        if symbol not in self.models:
            raise ValueError(f"No model built for {symbol}")
        
        model = self.models[symbol]
        
        logger.info(f"Training model for {symbol}...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict_prices(self, 
                       symbol: str,
                       X_train: np.ndarray,
                       X_test: np.ndarray,
                       y_train: np.ndarray,
                       y_test: np.ndarray) -> Dict:
        """
        Make predictions for a specific symbol
        """
        if symbol not in self.models or symbol not in self.scalers:
            raise ValueError(f"Model or scaler not available for {symbol}")
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Inverse transform predictions
        n_features = scaler.scale_.shape[0]
        
        # Create dummy arrays for inverse scaling
        train_pred_scaled = np.zeros((len(train_pred), n_features))
        test_pred_scaled = np.zeros((len(test_pred), n_features))
        train_actual_scaled = np.zeros((len(y_train), n_features))
        test_actual_scaled = np.zeros((len(y_test), n_features))
        
        train_pred_scaled[:, 0] = train_pred.flatten()
        test_pred_scaled[:, 0] = test_pred.flatten()
        train_actual_scaled[:, 0] = y_train
        test_actual_scaled[:, 0] = y_test
        
        train_pred_original = scaler.inverse_transform(train_pred_scaled)[:, 0]
        test_pred_original = scaler.inverse_transform(test_pred_scaled)[:, 0]
        train_actual_original = scaler.inverse_transform(train_actual_scaled)[:, 0]
        test_actual_original = scaler.inverse_transform(test_actual_scaled)[:, 0]
        
        return {
            'symbol': symbol,
            'train_pred': train_pred_original,
            'train_actual': train_actual_original,
            'test_pred': test_pred_original,
            'test_actual': test_actual_original
        }
    
    def evaluate_model(self, predictions: Dict) -> Dict:
        """
        Evaluate model performance
        """
        symbol = predictions['symbol']
        
        train_rmse = np.sqrt(mean_squared_error(predictions['train_actual'], predictions['train_pred']))
        test_rmse = np.sqrt(mean_squared_error(predictions['test_actual'], predictions['test_pred']))
        train_mae = mean_absolute_error(predictions['train_actual'], predictions['train_pred'])
        test_mae = mean_absolute_error(predictions['test_actual'], predictions['test_pred'])
        
        metrics = {
            'symbol': symbol,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        logger.info(f"Performance for {symbol}:")
        logger.info(f"  Train RMSE: ${train_rmse:.4f}")
        logger.info(f"  Test RMSE: ${test_rmse:.4f}")
        logger.info(f"  Train MAE: ${train_mae:.4f}")
        logger.info(f"  Test MAE: ${test_mae:.4f}")
        
        return metrics
    
    def plot_predictions(self, predictions: Dict, training_history=None):
        """
        Plot predictions for a symbol
        """
        symbol = predictions['symbol']
        data = self.data_cache[symbol]
        
        # Create dates for plotting
        sequence_length = 60  # Default sequence length
        train_dates = data.index[sequence_length:sequence_length + len(predictions['train_actual'])]
        test_dates = data.index[sequence_length + len(predictions['train_actual']):]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{symbol} Price Predictions vs Actual', 'Training History'),
            row_heights=[0.7, 0.3]
        )
        
        # Predictions plot
        fig.add_trace(
            go.Scatter(x=train_dates, y=predictions['train_actual'],
                      name='Actual (Train)', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=train_dates, y=predictions['train_pred'],
                      name='Predicted (Train)', line=dict(color='lightblue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=test_dates, y=predictions['test_actual'],
                      name='Actual (Test)', line=dict(color='red', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=test_dates, y=predictions['test_pred'],
                      name='Predicted (Test)', line=dict(color='pink', width=1)),
            row=1, col=1
        )
        
        # Training history
        if training_history:
            fig.add_trace(
                go.Scatter(y=training_history.history['loss'], name='Training Loss'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=training_history.history['val_loss'], name='Validation Loss'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text=f"{symbol} LSTM Price Prediction Results")
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        
        fig.show()
    
    def predict_multiple_symbols(self, 
                                 symbols: List[str],
                                 config: Dict = None) -> Dict:
        """
        Predict prices for multiple symbols
        
        Args:
            symbols: List of stock symbols
            config: Configuration dictionary with training parameters
            
        Returns:
            Dictionary with results for each symbol
        """
        if config is None:
            config = {
                'sequence_length': 60,
                'lstm_units': [64, 32],
                'epochs': 50,
                'batch_size': 16
            }
        
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")
                
                # Fetch data
                data = self.fetch_data(symbol)
                if data is None:
                    continue
                
                # Prepare data
                X_train, X_test, y_train, y_test = self.prepare_data(
                    symbol, 
                    sequence_length=config['sequence_length']
                )
                
                # Build and train model
                self.build_model(symbol, X_train, lstm_units=config['lstm_units'])
                history = self.train_model(
                    symbol, X_train, y_train, 
                    epochs=config['epochs'], 
                    batch_size=config['batch_size'],
                    verbose=0
                )
                
                # Make predictions
                predictions = self.predict_prices(symbol, X_train, X_test, y_train, y_test)
                
                # Evaluate
                metrics = self.evaluate_model(predictions)
                
                results[symbol] = {
                    'predictions': predictions,
                    'metrics': metrics,
                    'history': history
                }
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        return results


def create_config_file(filename: str = 'stock_predictor_config.json'):
    """Create a sample configuration file"""
    config = {
        "alpaca_credentials": {
            "api_key": "YOUR_ALPACA_API_KEY",
            "secret_key": "YOUR_ALPACA_SECRET_KEY",
            "base_url": "https://paper-api.alpaca.markets"
        },
        "model_config": {
            "sequence_length": 60,
            "lstm_units": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32
        },
        "data_config": {
            "timeframe": "Day",
            "features": ["close", "volume", "sma_20", "rsi", "macd", "volatility"],
            "train_split": 0.8
        },
        "symbols": ["AAPL", "GOOGL", "MSFT", "NVDL", "SPY"]
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration file created: {filename}")
    print("Please update with your Alpaca API credentials!")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Universal Stock Price Predictor')
    parser.add_argument('--symbol', type=str, help='Stock symbol to predict')
    parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols')
    parser.add_argument('--config', type=str, default='stock_predictor_config.json', 
                       help='Configuration file path')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--list-symbols', type=str, nargs='?', const='', 
                       help='List available symbols (optional search term)')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_config_file(args.config)
        return
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found!")
        print("Run with --create-config to create a sample configuration file")
        return
    
    # Initialize predictor
    predictor = UniversalStockPredictor(
        api_key=config['alpaca_credentials']['api_key'],
        secret_key=config['alpaca_credentials']['secret_key'],
        base_url=config['alpaca_credentials']['base_url']
    )
    
    # List symbols
    if args.list_symbols is not None:
        symbols = predictor.get_available_symbols(args.list_symbols)
        print(f"Found {len(symbols)} symbols:")
        for i, symbol in enumerate(symbols[:50]):  # Show first 50
            print(f"{symbol}", end="  ")
            if (i + 1) % 10 == 0:
                print()  # New line every 10 symbols
        print("\n...")
        return
    
    # Determine symbols to process
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = config.get('symbols', ['AAPL'])
    
    # Process symbols
    if len(symbols) == 1:
        # Single symbol processing with detailed output
        symbol = symbols[0]
        print(f"🚀 Predicting prices for {symbol}")
        
        # Fetch and prepare data
        data = predictor.fetch_data(symbol)
        if data is None:
            return
        
        X_train, X_test, y_train, y_test = predictor.prepare_data(
            symbol, 
            features=config['data_config']['features'],
            sequence_length=config['model_config']['sequence_length']
        )
        
        # Build and train model
        predictor.build_model(symbol, X_train, **config['model_config'])
        history = predictor.train_model(symbol, X_train, y_train, **config['model_config'])
        
        # Make predictions and evaluate
        predictions = predictor.predict_prices(symbol, X_train, X_test, y_train, y_test)
        metrics = predictor.evaluate_model(predictions)
        
        # Plot results
        predictor.plot_predictions(predictions, history)
        
    else:
        # Multiple symbols processing
        print(f"🚀 Predicting prices for {len(symbols)} symbols: {symbols}")
        results = predictor.predict_multiple_symbols(symbols, config['model_config'])
        
        # Summary report
        print("\n📊 Summary Report:")
        print("-" * 60)
        for symbol, result in results.items():
            metrics = result['metrics']
            print(f"{symbol:6} | Test RMSE: ${metrics['test_rmse']:8.2f} | Test MAE: ${metrics['test_mae']:8.2f}")


if __name__ == "__main__":
    main()