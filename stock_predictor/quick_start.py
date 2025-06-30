#!/usr/bin/env python3
"""
Quick Start Script for Universal Stock Predictor
Just add your API credentials and run!
"""

import os
import json
from universal_stock_predictor import UniversalStockPredictor

# TODO: Add your Alpaca API credentials here
API_KEY = ""
SECRET_KEY = ""

def main():
    print("🚀 Universal Stock Predictor - Quick Start")
    print("=" * 50)
    
    # Check if credentials are set
    if API_KEY == "YOUR_ALPACA_API_KEY" or SECRET_KEY == "YOUR_ALPACA_SECRET_KEY":
        print("❌ Please update your API credentials in this file!")
        print("   1. Get credentials from alpaca.markets")
        print("   2. Replace API_KEY and SECRET_KEY variables")
        print("   3. Run this script again")
        return
    
    # Initialize predictor
    try:
        predictor = UniversalStockPredictor(
            api_key=API_KEY,
            secret_key=SECRET_KEY
        )
        print("✅ Connected to Alpaca Markets")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Interactive menu
    while True:
        print("\n" + "=" * 50)
        print("What would you like to do?")
        print("1. Predict single stock (e.g., AAPL)")
        print("2. Compare multiple stocks")
        print("3. Find available symbols")
        print("4. Quick NVDL prediction (original request)")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
            predict_single_stock(predictor, symbol)
            
        elif choice == "2":
            symbols_input = input("Enter symbols separated by spaces (e.g., AAPL TSLA GOOGL): ").strip().upper()
            symbols = symbols_input.split()
            if symbols:
                predict_multiple_stocks(predictor, symbols)
            
        elif choice == "3":
            search_term = input("Enter search term (or press Enter for popular stocks): ").strip()
            find_symbols(predictor, search_term)
            
        elif choice == "4":
            print("🎯 Running NVDL prediction (your original request)...")
            predict_single_stock(predictor, "NVDL")
            
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice, please try again")

def predict_single_stock(predictor, symbol):
    """Predict prices for a single stock"""
    try:
        print(f"\n📊 Analyzing {symbol}...")
        
        # Validate symbol
        if not predictor.validate_symbol(symbol):
            print(f"❌ {symbol} is not available for trading")
            return
        
        # Fetch data
        print("📥 Fetching historical data...")
        data = predictor.fetch_data(symbol)
        if data is None:
            return
        
        print(f"✅ Got {len(data)} days of data")
        
        # Prepare data
        print("🔧 Preparing data...")
        X_train, X_test, y_train, y_test = predictor.prepare_data(
            symbol, 
            features=['close', 'volume', 'sma_20', 'rsi', 'volatility'],
            sequence_length=30  # Faster training
        )
        
        # Build and train model
        print("🧠 Building model...")
        predictor.build_model(symbol, X_train, lstm_units=[32, 16])  # Smaller model for speed
        
        print("🏋️ Training model (this may take 2-3 minutes)...")
        history = predictor.train_model(
            symbol, X_train, y_train, 
            epochs=50,  # Fewer epochs for speed
            batch_size=16,
            verbose=0
        )
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = predictor.predict_prices(symbol, X_train, X_test, y_train, y_test)
        metrics = predictor.evaluate_model(predictions)
        
        # Show results
        latest_price = data['close'].iloc[-1]
        print(f"\n💰 Current {symbol} Price: ${latest_price:.2f}")
        print(f"📈 Model Accuracy (RMSE): ${metrics['test_rmse']:.2f}")
        print(f"📊 Mean Error (MAE): ${metrics['test_mae']:.2f}")
        
        # Plot results
        print("📊 Generating interactive chart...")
        predictor.plot_predictions(predictions, history)
        
    except Exception as e:
        print(f"❌ Error predicting {symbol}: {e}")

def predict_multiple_stocks(predictor, symbols):
    """Predict prices for multiple stocks"""
    try:
        print(f"\n📊 Comparing {len(symbols)} stocks: {symbols}")
        
        # Filter valid symbols
        valid_symbols = []
        for symbol in symbols:
            if predictor.validate_symbol(symbol):
                valid_symbols.append(symbol)
                print(f"✅ {symbol} is valid")
            else:
                print(f"❌ {symbol} is not available")
        
        if not valid_symbols:
            print("❌ No valid symbols found")
            return
        
        print(f"\n🏋️ Training models for {len(valid_symbols)} stocks...")
        print("This will take a few minutes...")
        
        # Use faster config for multiple stocks
        config = {
            'sequence_length': 30,
            'lstm_units': [32, 16],
            'epochs': 30,
            'batch_size': 16
        }
        
        results = predictor.predict_multiple_symbols(valid_symbols, config)
        
        # Show comparison
        print(f"\n📊 Results Summary:")
        print("-" * 60)
        print(f"{'Symbol':<8} {'Current Price':<12} {'RMSE':<10} {'MAE':<10}")
        print("-" * 60)
        
        for symbol, result in results.items():
            metrics = result['metrics']
            current_price = predictor.data_cache[symbol]['close'].iloc[-1]
            print(f"{symbol:<8} ${current_price:<11.2f} ${metrics['test_rmse']:<9.2f} ${metrics['test_mae']:<9.2f}")
        
        # Find best performer
        best_symbol = min(results.keys(), 
                         key=lambda x: results[x]['metrics']['test_rmse'])
        print(f"\n🏆 Most predictable stock: {best_symbol}")
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")

def find_symbols(predictor, search_term=""):
    """Find available symbols"""
    try:
        if search_term:
            print(f"🔍 Searching for symbols containing '{search_term}'...")
            symbols = predictor.get_available_symbols(search_term)
        else:
            print("📋 Popular stocks and ETFs:")
            # Show some popular ones instead of all symbols
            popular = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
                      'SPY', 'QQQ', 'VTI', 'NVDL', 'TQQQ', 'ARKK', 'BTC', 'ETH']
            symbols = [s for s in popular if predictor.validate_symbol(s)]
        
        if symbols:
            print(f"Found {len(symbols)} symbols:")
            # Show in columns
            for i, symbol in enumerate(symbols[:50]):  # Limit to 50
                print(f"{symbol:<8}", end="")
                if (i + 1) % 8 == 0:  # 8 per row
                    print()
            print("\n")
        else:
            print("❌ No symbols found")
            
    except Exception as e:
        print(f"❌ Error searching symbols: {e}")

if __name__ == "__main__":
    main()