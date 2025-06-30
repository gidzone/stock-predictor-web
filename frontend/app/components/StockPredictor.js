'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { TrendingUp, TrendingDown, DollarSign, Target, Brain, Search, Loader2, AlertCircle, CheckCircle } from 'lucide-react'

const API_BASE_URL = 'http://localhost:5040/api'

export default function StockPredictor() {
  const [symbol, setSymbol] = useState('')
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState('')
  const [popularSymbols, setPopularSymbols] = useState({})
  const [compareMode, setCompareMode] = useState(false)
  const [compareSymbols, setCompareSymbols] = useState('')
  const [compareResults, setCompareResults] = useState([])

  // Fetch popular symbols on component mount
  useEffect(() => {
    fetchPopularSymbols()
  }, [])

  const fetchPopularSymbols = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/popular-symbols`)
      setPopularSymbols(response.data)
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
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        symbol: stockSymbol.toUpperCase()
      })
      
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to predict stock price')
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
      const response = await axios.post(`${API_BASE_URL}/compare`, {
        symbols: symbols
      })
      
      setCompareResults(response.data.results)
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to compare stocks')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-3">
            <Brain className="text-blue-600" size={40} />
            AI Stock Predictor
          </h1>
          <p className="text-gray-600 text-lg">Machine Learning-powered stock price predictions</p>
        </div>

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
          <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-lg p-6">
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
              type="submit"
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
          </form>

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

        {/* Error Message */}
        {error && (
          <div className="max-w-2xl mx-auto mb-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="text-red-500" size={20} />
              <span className="text-red-700">{error}</span>
            </div>
          </div>
        )}

        {/* Single Prediction Results */}
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
              {/* Price Prediction Chart */}
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

              {/* Historical Price Chart */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                  Historical Price Trend
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={formatHistoricalData(prediction)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#10b981" 
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Feature Importance */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                Most Important Features
              </h3>
              <ResponsiveContainer width="100%" height={250}>
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
      </div>
    </div>
  )
}