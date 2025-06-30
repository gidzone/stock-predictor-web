'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { 
  TrendingUp, TrendingDown, DollarSign, Target, Brain, Search, Loader2, 
  AlertCircle, CheckCircle, XCircle, BarChart3, TestTube 
} from 'lucide-react'

const API_BASE_URL = 'http://localhost:5040/api'

export default function StockPredictor() {
  const [activeTab, setActiveTab] = useState('predict') // 'predict' or 'backtest'
  
  // Prediction state
  const [symbol, setSymbol] = useState('')
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState('')
  const [popularSymbols, setPopularSymbols] = useState({})
  const [compareMode, setCompareMode] = useState(false)
  const [compareSymbols, setCompareSymbols] = useState('')
  const [compareResults, setCompareResults] = useState([])

  // Backtest state
  const [backtestSymbols, setBacktestSymbols] = useState('AAPL,SPY,QQQ,NVDL,TQQQ')
  const [backtestLoading, setBacktestLoading] = useState(false)
  const [backtestResults, setBacktestResults] = useState(null)
  const [quickTest, setQuickTest] = useState('')
  const [quickResult, setQuickResult] = useState(null)

  // Fetch popular symbols on component mount
  useEffect(() => {
    fetchPopularSymbols()
  }, [])

  const fetchPopularSymbols = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/popular-symbols`)
      const data = await response.json()
      setPopularSymbols(data)
    } catch (err) {
      console.error('Failed to fetch popular symbols:', err)
    }
  }

  const predictStock = async (stockSymbol) => {
    if (!stockSymbol.trim()) {
      setError('Please enter a stock symbol')
      return
    }

    setLoading(true)
    setError('')
    setPrediction(null)

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbol: stockSymbol.toUpperCase() })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to predict stock price')
      }
      
      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message || 'Failed to predict stock price')
    } finally {
      setLoading(false)
    }
  }

  const compareStocks = async () => {
    const symbols = compareSymbols.split(',').map(s => s.trim()).filter(s => s)
    
    if (symbols.length === 0) {
      setError('Please enter stock symbols separated by commas')
      return
    }

    if (symbols.length > 5) {
      setError('Maximum 5 symbols allowed for comparison')
      return
    }

    setLoading(true)
    setError('')
    setCompareResults([])

    try {
      const response = await fetch(`${API_BASE_URL}/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbols: symbols })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to compare stocks')
      }
      
      const data = await response.json()
      setCompareResults(data.results)
    } catch (err) {
      setError(err.message || 'Failed to compare stocks')
    } finally {
      setLoading(false)
    }
  }

  const runBacktest = async () => {
    if (!backtestSymbols.trim()) {
      setError('Please enter symbols to test')
      return
    }

    setBacktestLoading(true)
    setError('')
    setBacktestResults(null)

    try {
      const symbolList = backtestSymbols.split(',').map(s => s.trim().toUpperCase()).filter(s => s)
      
      const response = await fetch(`${API_BASE_URL}/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbols: symbolList })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Backtest failed')
      }
      
      const data = await response.json()
      setBacktestResults(data)
    } catch (err) {
      setError(err.message || 'Backtest failed')
    } finally {
      setBacktestLoading(false)
    }
  }

  const quickValidation = async () => {
    if (!quickTest.trim()) return

    try {
      const response = await fetch(`${API_BASE_URL}/quick-validation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbol: quickTest.toUpperCase() })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Quick validation failed')
      }

      const data = await response.json()
      setQuickResult(data)
    } catch (err) {
      setError(err.message || 'Quick validation failed')
    }
  }

  const handleSubmit = () => {
    if (compareMode) {
      compareStocks()
    } else {
      predictStock(symbol)
    }
  }

  const formatChartData = (prediction) => {
    if (!prediction?.chart_data) return []
    
    const { dates, actual_prices, predicted_prices } = prediction.chart_data
    
    return dates.map((date, index) => ({
      date: new Date(date).toLocaleDateString(),
      actual: actual_prices[index],
      predicted: predicted_prices[index]
    }))
  }

  const formatHistoricalData = (prediction) => {
    if (!prediction?.chart_data) return []
    
    const { historical_dates, historical_prices } = prediction.chart_data
    
    return historical_dates.map((date, index) => ({
      date: new Date(date).toLocaleDateString(),
      price: historical_prices[index]
    }))
  }

  const formatFeatureData = (prediction) => {
    if (!prediction?.feature_importance) return []
    
    return prediction.feature_importance.map(item => ({
      feature: item.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      importance: (item.importance * 100).toFixed(1)
    }))
  }

  const formatCurrency = (value) => `$${value.toLocaleString()}`
  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`

  const getStatusIcon = (hasEdge) => {
    return hasEdge ? 
      <CheckCircle className="text-green-500" size={20} /> : 
      <XCircle className="text-red-500" size={20} />
  }

  const getStatusColor = (hasEdge) => {
    return hasEdge ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-3">
            <Brain className="text-blue-600" size={40} />
            AI Stock Predictor
          </h1>
          <p className="text-gray-600 text-lg">Machine Learning-powered stock price predictions and backtesting</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-lg p-1 shadow-md">
            <button
              onClick={() => setActiveTab('predict')}
              className={`px-6 py-3 rounded-md transition-all flex items-center gap-2 ${
                activeTab === 'predict' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'text-gray-600 hover:text-blue-600'
              }`}
            >
              <TrendingUp size={20} />
              Predictions
            </button>
            <button
              onClick={() => setActiveTab('backtest')}
              className={`px-6 py-3 rounded-md transition-all flex items-center gap-2 ${
                activeTab === 'backtest' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'text-gray-600 hover:text-blue-600'
              }`}
            >
              <TestTube size={20} />
              Edge Validation
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="max-w-4xl mx-auto mb-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="text-red-500" size={20} />
              <span className="text-red-700">{error}</span>
            </div>
          </div>
        )}

        {/* Prediction Tab */}
        {activeTab === 'predict' && (
          <>
            {/* Mode Toggle */}
            <div className="flex justify-center mb-6">
              <div className="bg-white rounded-lg p-1 shadow-md">
                <button
                  onClick={() => setCompareMode(false)}
                  className={`px-4 py-2 rounded-md transition-all ${
                    !compareMode 
                      ? 'bg-blue-600 text-white shadow-md' 
                      : 'text-gray-600 hover:text-blue-600'
                  }`}
                >
                  Single Prediction
                </button>
                <button
                  onClick={() => setCompareMode(true)}
                  className={`px-4 py-2 rounded-md transition-all ${
                    compareMode 
                      ? 'bg-blue-600 text-white shadow-md' 
                      : 'text-gray-600 hover:text-blue-600'
                  }`}
                >
                  Compare Stocks
                </button>
              </div>
            </div>

            {/* Input Form */}
            <div className="max-w-2xl mx-auto mb-8">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="mb-4">
                  {compareMode ? (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Stock Symbols (comma-separated, max 5)
                      </label>
                      <input
                        type="text"
                        value={compareSymbols}
                        onChange={(e) => setCompareSymbols(e.target.value)}
                        placeholder="AAPL, TSLA, GOOGL"
                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  ) : (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Stock Symbol
                      </label>
                      <input
                        type="text"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value)}
                        placeholder="Enter symbol (e.g., AAPL, NVDL)"
                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  )}
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="animate-spin" size={20} />
                      {compareMode ? 'Comparing...' : 'Predicting...'}
                    </>
                  ) : (
                    <>
                      <Search size={20} />
                      {compareMode ? 'Compare Stocks' : 'Predict Price'}
                    </>
                  )}
                </button>
              </div>

              {/* Popular Symbols */}
              {Object.keys(popularSymbols).length > 0 && (
                <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Popular Symbols</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {Object.entries(popularSymbols).map(([category, symbols]) => (
                      <div key={category}>
                        <h4 className="font-medium text-gray-700 mb-2">{category}</h4>
                        <div className="flex flex-wrap gap-1">
                          {symbols.map((sym) => (
                            <button
                              key={sym}
                              onClick={() => compareMode ? setCompareSymbols(sym) : setSymbol(sym)}
                              className="px-2 py-1 bg-gray-100 hover:bg-blue-100 text-gray-700 hover:text-blue-700 rounded text-sm transition-colors"
                            >
                              {sym}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Prediction Results */}
            {prediction && !compareMode && (
              <div className="max-w-6xl mx-auto space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <div className="flex items-center gap-3">
                      <DollarSign className="text-green-600" size={24} />
                      <div>
                        <p className="text-sm text-gray-600">Current Price</p>
                        <p className="text-xl font-bold text-gray-800">
                          ${prediction.current_price.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <div className="flex items-center gap-3">
                      <Target className="text-blue-600" size={24} />
                      <div>
                        <p className="text-sm text-gray-600">Accuracy</p>
                        <p className="text-xl font-bold text-gray-800">
                          {prediction.accuracy_pct.toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <div className="flex items-center gap-3">
                      <TrendingUp className="text-orange-600" size={24} />
                      <div>
                        <p className="text-sm text-gray-600">RMSE</p>
                        <p className="text-xl font-bold text-gray-800">
                          ${prediction.test_rmse.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <div className="flex items-center gap-3">
                      <CheckCircle className="text-purple-600" size={24} />
                      <div>
                        <p className="text-sm text-gray-600">Data Points</p>
                        <p className="text-xl font-bold text-gray-800">
                          {prediction.data_points}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                      Prediction vs Actual (Test Period)
                    </h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={formatChartData(prediction)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="actual" 
                          stroke="#ef4444" 
                          strokeWidth={2}
                          name="Actual Price"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="predicted" 
                          stroke="#3b82f6" 
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          name="Predicted Price"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                      Feature Importance
                    </h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={formatFeatureData(prediction)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="feature" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="importance" fill="#8b5cf6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {/* Comparison Results */}
            {compareResults.length > 0 && compareMode && (
              <div className="max-w-4xl mx-auto">
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">
                    Stock Comparison Results
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="py-3 px-4 font-medium text-gray-700">Symbol</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Current Price</th>
                          <th className="py-3 px-4 font-medium text-gray-700">RMSE</th>
                          <th className="py-3 px-4 font-medium text-gray-700">MAE</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Accuracy</th>
                        </tr>
                      </thead>
                      <tbody>
                        {compareResults.map((result, index) => (
                          <tr key={result.symbol} className="border-b border-gray-100">
                            <td className="py-3 px-4">
                              <span className="font-medium text-blue-600">{result.symbol}</span>
                            </td>
                            <td className="py-3 px-4">${result.current_price.toFixed(2)}</td>
                            <td className="py-3 px-4">${result.test_rmse.toFixed(2)}</td>
                            <td className="py-3 px-4">${result.test_mae.toFixed(2)}</td>
                            <td className="py-3 px-4">
                              <span className={`px-2 py-1 rounded text-sm ${
                                result.accuracy_pct > 80 
                                  ? 'bg-green-100 text-green-800' 
                                  : result.accuracy_pct > 60 
                                  ? 'bg-yellow-100 text-yellow-800' 
                                  : 'bg-red-100 text-red-800'
                              }`}>
                                {result.accuracy_pct.toFixed(1)}%
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Backtest Tab */}
        {activeTab === 'backtest' && (
          <>
            {/* Controls */}
            <div className="max-w-4xl mx-auto mb-8 space-y-4">
              
              {/* Full Backtest */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">
                  Walk-Forward Backtest - Find Your Edge
                </h2>
                <div className="flex gap-4">
                  <input
                    type="text"
                    value={backtestSymbols}
                    onChange={(e) => setBacktestSymbols(e.target.value)}
                    placeholder="AAPL,SPY,QQQ,NVDL,TQQQ"
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <button
                    onClick={runBacktest}
                    disabled={backtestLoading}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                  >
                    {backtestLoading ? (
                      <>
                        <Loader2 className="animate-spin" size={20} />
                        Testing...
                      </>
                    ) : (
                      <>
                        <BarChart3 size={20} />
                        Run Backtest
                      </>
                    )}
                  </button>
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  This will test if your model can actually make money with realistic transaction costs
                </p>
              </div>

              {/* Quick Test */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">
                  Quick Symbol Validation
                </h2>
                <div className="flex gap-4">
                  <input
                    type="text"
                    value={quickTest}
                    onChange={(e) => setQuickTest(e.target.value)}
                    placeholder="Enter single symbol (e.g., AAPL)"
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <button
                    onClick={quickValidation}
                    className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
                  >
                    <Target size={20} />
                    Quick Test
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Result */}
            {quickResult && (
              <div className="max-w-4xl mx-auto mb-6">
                <div className={`rounded-lg p-6 border-2 ${quickResult.has_edge ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      {getStatusIcon(quickResult.has_edge)}
                      {quickResult.symbol} Quick Validation
                    </h3>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(quickResult.has_edge)}`}>
                      {quickResult.recommendation}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-600">Total Return</p>
                      <p className="text-xl font-bold">{formatPercent(quickResult.total_return)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Sharpe Ratio</p>
                      <p className="text-xl font-bold">{quickResult.sharpe_ratio.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Win Rate</p>
                      <p className="text-xl font-bold">{formatPercent(quickResult.win_rate)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">$5K Becomes</p>
                      <p className="text-xl font-bold">{formatCurrency(quickResult.final_value)}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Backtest Results */}
            {backtestResults && (
              <div className="max-w-7xl mx-auto space-y-6">
                
                {/* Summary */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h2 className="text-2xl font-bold text-gray-800 mb-4">
                    Backtest Summary
                  </h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    <div className="bg-blue-50 rounded-lg p-4">
                      <p className="text-sm text-blue-600">Symbols Tested</p>
                      <p className="text-2xl font-bold text-blue-800">{backtestResults.summary.total_tested}</p>
                    </div>
                    <div className="bg-green-50 rounded-lg p-4">
                      <p className="text-sm text-green-600">Profitable</p>
                      <p className="text-2xl font-bold text-green-800">{backtestResults.summary.profitable_strategies}</p>
                    </div>
                    <div className="bg-purple-50 rounded-lg p-4">
                      <p className="text-sm text-purple-600">With Edge</p>
                      <p className="text-2xl font-bold text-purple-800">{backtestResults.summary.strategies_with_edge}</p>
                    </div>
                    <div className={`rounded-lg p-4 ${backtestResults.summary.target_achieved ? 'bg-yellow-50' : 'bg-gray-50'}`}>
                      <p className={`text-sm ${backtestResults.summary.target_achieved ? 'text-yellow-600' : 'text-gray-600'}`}>
                        10x Target
                      </p>
                      <p className={`text-2xl font-bold ${backtestResults.summary.target_achieved ? 'text-yellow-800' : 'text-gray-800'}`}>
                        {backtestResults.summary.target_achieved ? '✅ HIT' : '❌ MISS'}
                      </p>
                    </div>
                  </div>

                  {/* Recommendation */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-800 mb-2">Recommendation:</h3>
                    <p className="text-gray-700">{backtestResults.summary.recommendation}</p>
                  </div>
                </div>

                {/* Detailed Results Table */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h2 className="text-2xl font-bold text-gray-800 mb-4">
                    Detailed Results
                  </h2>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="py-3 px-4 font-medium text-gray-700">Symbol</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Status</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Total Return</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Sharpe</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Win Rate</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Max DD</th>
                          <th className="py-3 px-4 font-medium text-gray-700">Trades</th>
                          <th className="py-3 px-4 font-medium text-gray-700">$5K → </th>
                        </tr>
                      </thead>
                      <tbody>
                        {backtestResults.results.map((result, index) => (
                          <tr key={result.symbol} className="border-b border-gray-100">
                            <td className="py-3 px-4">
                              <span className="font-medium text-blue-600">{result.symbol}</span>
                            </td>
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                {getStatusIcon(result.has_edge)}
                                <span className={`text-sm ${result.has_edge ? 'text-green-600' : 'text-red-600'}`}>
                                  {result.has_edge ? 'EDGE' : 'NO EDGE'}
                                </span>
                              </div>
                            </td>
                            <td className="py-3 px-4">
                              <span className={result.total_return > 0 ? 'text-green-600' : 'text-red-600'}>
                                {formatPercent(result.total_return)}
                              </span>
                            </td>
                            <td className="py-3 px-4">{result.sharpe_ratio.toFixed(2)}</td>
                            <td className="py-3 px-4">{formatPercent(result.win_rate)}</td>
                            <td className="py-3 px-4 text-red-600">{formatPercent(result.max_drawdown)}</td>
                            <td className="py-3 px-4">{result.total_trades}</td>
                            <td className="py-3 px-4">
                              <span className={result.final_value >= 50000 ? 'text-green-600 font-bold' : 'text-gray-600'}>
                                {formatCurrency(result.final_value)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Performance Chart */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h2 className="text-2xl font-bold text-gray-800 mb-4">
                    Performance Comparison
                  </h2>
                  
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart 
                      data={backtestResults.results.map(r => ({
                        symbol: r.symbol,
                        return: r.total_return * 100,
                        sharpe: r.sharpe_ratio * 10, // Scale for visibility
                        hasEdge: r.has_edge
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="symbol" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value, name) => [
                          name === 'return' ? `${value.toFixed(1)}%` : (value/10).toFixed(2),
                          name === 'return' ? 'Return' : 'Sharpe Ratio'
                        ]}
                      />
                      <Legend />
                      <Bar 
                        dataKey="return" 
                        fill="#3b82f6" 
                        name="Total Return (%)"
                      />
                      <Bar 
                        dataKey="sharpe" 
                        fill="#ef4444" 
                        name="Sharpe Ratio (×10)"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}