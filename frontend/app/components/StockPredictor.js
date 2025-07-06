'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { 
  TrendingUp, TrendingDown, DollarSign, Target, Brain, Search, Loader2, 
  AlertCircle, CheckCircle, XCircle, BarChart3, TestTube 
} from 'lucide-react'

const API_BASE_URL = 'http://localhost:5040/api'

export default function StockPredictor() {
  const [activeTab, setActiveTab] = useState('predict')
  
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
    setLoading(true)
    setError('')
    setPrediction(null)
    
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: stockSymbol.toUpperCase() })
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Prediction failed')
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
    if (symbols.length === 0) return
    
    setLoading(true)
    setError('')
    setCompareResults([])
    
    try {
      const response = await fetch(`${API_BASE_URL}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols })
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Comparison failed')
      }
      
      const data = await response.json()
      setCompareResults(data.comparisons)
    } catch (err) {
      setError(err.message || 'Failed to compare stocks')
    } finally {
      setLoading(false)
    }
  }

  const backtestStrategy = async () => {
    const symbols = backtestSymbols.split(',').map(s => s.trim()).filter(s => s)
    if (symbols.length === 0) return
    
    setBacktestLoading(true)
    setError('')
    
    try {
      const response = await fetch(`${API_BASE_URL}/backtest-edge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols })
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Backtest failed')
      }
      
      const data = await response.json()
      setBacktestResults(data)
    } catch (err) {
      setError(err.message || 'Failed to run backtest')
    } finally {
      setBacktestLoading(false)
    }
  }

  const quickValidate = async () => {
    if (!quickTest) return
    
    setLoading(true)
    setError('')
    
    try {
      const response = await fetch(`${API_BASE_URL}/quick-validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      <CheckCircle className="text-emerald-400" size={20} /> : 
      <XCircle className="text-red-400" size={20} />
  }

  const getStatusColor = (hasEdge) => {
    return hasEdge ? 'text-emerald-500 bg-emerald-900/20' : 'text-red-500 bg-red-900/20'
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Brain className="text-emerald-400" size={40} />
            AI Stock Predictor
          </h1>
          <p className="text-slate-300 text-lg">Machine Learning-powered stock price predictions and backtesting</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-slate-800 rounded-lg p-1 shadow-md border border-slate-700">
            <button
              onClick={() => setActiveTab('predict')}
              className={`px-6 py-3 rounded-md transition-all flex items-center gap-2 ${
                activeTab === 'predict' 
                  ? 'bg-emerald-500 text-white shadow-md' 
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              <Brain size={20} />
              Predict
            </button>
            <button
              onClick={() => setActiveTab('backtest')}
              className={`px-6 py-3 rounded-md transition-all flex items-center gap-2 ${
                activeTab === 'backtest' 
                  ? 'bg-emerald-500 text-white shadow-md' 
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              <TestTube size={20} />
              Backtest Edge
            </button>
          </div>
        </div>

        {/* Predict Tab */}
        {activeTab === 'predict' && (
          <>
            {/* Search Section */}
            <div className="max-w-4xl mx-auto mb-8">
              <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-white">Stock Price Prediction</h2>
                  <button
                    onClick={() => setCompareMode(!compareMode)}
                    className={`px-4 py-2 rounded-lg transition-all ${
                      compareMode 
                        ? 'bg-emerald-500 text-white' 
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {compareMode ? 'Single Mode' : 'Compare Mode'}
                  </button>
                </div>

                <div className="flex gap-4">
                  <input
                    type="text"
                    placeholder={compareMode ? "Enter symbols (AAPL,MSFT,GOOGL)" : "Enter stock symbol (e.g., AAPL)"}
                    value={compareMode ? compareSymbols : symbol}
                    onChange={(e) => compareMode ? setCompareSymbols(e.target.value) : setSymbol(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
                    className="flex-1 px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/20"
                  />
                  <button
                    onClick={handleSubmit}
                    disabled={loading || (compareMode ? !compareSymbols : !symbol)}
                    className="px-8 py-3 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="animate-spin" size={20} />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Search size={20} />
                        {compareMode ? 'Compare' : 'Predict'}
                      </>
                    )}
                  </button>
                </div>

                {/* Popular Symbols */}
                <div className="mt-4">
                  <p className="text-sm text-slate-400 mb-2">Popular symbols:</p>
                  <div className="space-y-2">
                    {Object.entries(popularSymbols).map(([category, symbols]) => (
                      <div key={category} className="flex flex-wrap gap-2">
                        <span className="text-xs text-slate-500 font-medium min-w-[80px]">{category}:</span>
                        <div className="flex gap-2">
                          {symbols.map(sym => (
                            <button
                              key={sym}
                              onClick={() => compareMode ? setCompareSymbols(sym) : setSymbol(sym)}
                              className="px-2 py-1 bg-slate-700 hover:bg-emerald-700 text-slate-300 hover:text-white rounded text-sm transition-colors"
                            >
                              {sym}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Error Display */}
              {error && (
                <div className="mt-4 bg-red-900/20 border border-red-500 rounded-lg p-4 flex items-center gap-3">
                  <AlertCircle className="text-red-400" size={24} />
                  <p className="text-red-300">{error}</p>
                </div>
              )}
            </div>

            {/* Prediction Results */}
            {prediction && !compareMode && (
              <div className="max-w-6xl mx-auto space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <div className="flex items-center gap-3">
                      <DollarSign className="text-emerald-400" size={24} />
                      <div>
                        <p className="text-sm text-slate-400">Current Price</p>
                        <p className="text-xl font-bold text-white">
                          ${prediction.current_price.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <div className="flex items-center gap-3">
                      <Target className="text-emerald-400" size={24} />
                      <div>
                        <p className="text-sm text-slate-400">Accuracy</p>
                        <p className="text-xl font-bold text-white">
                          {prediction.accuracy_pct.toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <div className="flex items-center gap-3">
                      {prediction.predicted_price > prediction.current_price ? (
                        <TrendingUp className="text-emerald-400" size={24} />
                      ) : (
                        <TrendingDown className="text-red-400" size={24} />
                      )}
                      <div>
                        <p className="text-sm text-slate-400">Predicted Price</p>
                        <p className="text-xl font-bold text-white">
                          ${prediction.predicted_price.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <div className="flex items-center gap-3">
                      <BarChart3 className="text-emerald-400" size={24} />
                      <div>
                        <p className="text-sm text-slate-400">RMSE</p>
                        <p className="text-xl font-bold text-white">
                          ${prediction.rmse.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Prediction Chart */}
                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <h3 className="text-lg font-semibold text-white mb-4">Prediction vs Actual</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={formatChartData(prediction)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="date" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                          labelStyle={{ color: '#94a3b8' }}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="actual" stroke="#10b981" name="Actual Price" strokeWidth={2} />
                        <Line type="monotone" dataKey="predicted" stroke="#ef4444" name="Predicted Price" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Historical Chart */}
                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <h3 className="text-lg font-semibold text-white mb-4">Historical Trend</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={formatHistoricalData(prediction)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="date" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                          labelStyle={{ color: '#94a3b8' }}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="price" stroke="#10b981" name="Price" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Feature Importance */}
                <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                  <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={formatFeatureData(prediction)} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#94a3b8" />
                      <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={120} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                        labelStyle={{ color: '#94a3b8' }}
                      />
                      <Bar dataKey="importance" fill="#a855f7" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Comparison Results */}
            {compareResults.length > 0 && compareMode && (
              <div className="max-w-6xl mx-auto">
                <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Stock Comparison Results</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-3 px-4 text-slate-300">Symbol</th>
                          <th className="text-right py-3 px-4 text-slate-300">Current Price</th>
                          <th className="text-right py-3 px-4 text-slate-300">Predicted Price</th>
                          <th className="text-right py-3 px-4 text-slate-300">Change</th>
                          <th className="text-right py-3 px-4 text-slate-300">Accuracy</th>
                          <th className="text-right py-3 px-4 text-slate-300">RMSE</th>
                          <th className="text-center py-3 px-4 text-slate-300">Trend</th>
                        </tr>
                      </thead>
                      <tbody>
                        {compareResults.map((result, idx) => {
                          const changePercent = ((result.predicted_price - result.current_price) / result.current_price) * 100
                          const isPositive = changePercent > 0
                          
                          return (
                            <tr key={idx} className="border-b border-slate-700">
                              <td className="py-3 px-4 font-medium text-white">{result.symbol}</td>
                              <td className="text-right py-3 px-4 text-white">${result.current_price.toFixed(2)}</td>
                              <td className="text-right py-3 px-4 text-white">${result.predicted_price.toFixed(2)}</td>
                              <td className={`text-right py-3 px-4 font-medium ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                                {isPositive ? '+' : ''}{changePercent.toFixed(2)}%
                              </td>
                              <td className="text-right py-3 px-4 text-white">{result.accuracy_pct.toFixed(1)}%</td>
                              <td className="text-right py-3 px-4 text-slate-300">${result.rmse.toFixed(2)}</td>
                              <td className="text-center py-3 px-4">
                                {isPositive ? (
                                  <TrendingUp className="text-emerald-400 mx-auto" size={20} />
                                ) : (
                                  <TrendingDown className="text-red-400 mx-auto" size={20} />
                                )}
                              </td>
                            </tr>
                          )
                        })}
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
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Backtest Input */}
            <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
              <h2 className="text-xl font-semibold text-white mb-4">Test Predictive Edge</h2>
              <p className="text-slate-300 mb-4">
                Test if our AI predictions would have given you an edge over buy-and-hold strategy.
              </p>
              
              <div className="flex gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Enter symbols to test (comma-separated)"
                  value={backtestSymbols}
                  onChange={(e) => setBacktestSymbols(e.target.value)}
                  className="flex-1 px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/20"
                />
                <button
                  onClick={backtestStrategy}
                  disabled={backtestLoading || !backtestSymbols}
                  className="px-8 py-3 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
                >
                  {backtestLoading ? (
                    <>
                      <Loader2 className="animate-spin" size={20} />
                      Testing...
                    </>
                  ) : (
                    <>
                      <TestTube size={20} />
                      Run Backtest
                    </>
                  )}
                </button>
              </div>

              {/* Quick Test */}
              <div className="border-t border-slate-700 pt-4">
                <p className="text-sm text-slate-400 mb-2">Quick test a single symbol:</p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="Symbol"
                    value={quickTest}
                    onChange={(e) => setQuickTest(e.target.value)}
                    className="w-32 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-400"
                  />
                  <button
                    onClick={quickValidate}
                    className="px-4 py-2 bg-slate-600 text-white rounded hover:bg-slate-700 transition-all"
                  >
                    Quick Test
                  </button>
                  {quickResult && (
                    <div className={`flex items-center gap-2 px-4 py-2 rounded ${getStatusColor(quickResult.has_edge)}`}>
                      {getStatusIcon(quickResult.has_edge)}
                      <span>{quickResult.has_edge ? 'Has Edge' : 'No Edge'}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Backtest Results */}
            {backtestResults && (
              <>
                {/* Summary Stats */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <h3 className="text-sm text-slate-400 mb-2">Stocks with Edge</h3>
                    <p className="text-2xl font-bold text-emerald-400">
                      {backtestResults.summary.stocks_with_edge} / {backtestResults.summary.total_stocks}
                    </p>
                  </div>
                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <h3 className="text-sm text-slate-400 mb-2">Average Accuracy</h3>
                    <p className="text-2xl font-bold text-white">
                      {backtestResults.summary.avg_accuracy.toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <h3 className="text-sm text-slate-400 mb-2">Win Rate</h3>
                    <p className="text-2xl font-bold text-emerald-400">
                      {formatPercent(backtestResults.summary.win_rate)}
                    </p>
                  </div>
                  <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                    <h3 className="text-sm text-slate-400 mb-2">Best Performer</h3>
                    <p className="text-2xl font-bold text-white">
                      {backtestResults.summary.best_performer}
                    </p>
                  </div>
                </div>

                {/* Detailed Results */}
                <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Backtest Results by Symbol</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-3 px-4 text-slate-300">Symbol</th>
                          <th className="text-center py-3 px-4 text-slate-300">Has Edge?</th>
                          <th className="text-right py-3 px-4 text-slate-300">Accuracy</th>
                          <th className="text-right py-3 px-4 text-slate-300">ML Return</th>
                          <th className="text-right py-3 px-4 text-slate-300">Buy & Hold</th>
                          <th className="text-right py-3 px-4 text-slate-300">Outperformance</th>
                          <th className="text-right py-3 px-4 text-slate-300">Total Profit</th>
                          <th className="text-center py-3 px-4 text-slate-300">Recommendation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {backtestResults.results.map((result, idx) => (
                          <tr key={idx} className={`border-b border-slate-700 ${
                            result.has_predictive_edge ? 'bg-emerald-900/10' : ''
                          }`}>
                            <td className="py-3 px-4 font-medium text-white">{result.symbol}</td>
                            <td className="text-center py-3 px-4">
                              {getStatusIcon(result.has_predictive_edge)}
                            </td>
                            <td className="text-right py-3 px-4 text-white">
                              {result.accuracy_pct.toFixed(1)}%
                            </td>
                            <td className="text-right py-3 px-4 text-emerald-400">
                              {formatPercent(result.ml_return)}
                            </td>
                            <td className="text-right py-3 px-4 text-slate-300">
                              {formatPercent(result.buy_hold_return)}
                            </td>
                            <td className={`text-right py-3 px-4 font-medium ${
                              result.outperformance > 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}>
                              {result.outperformance > 0 ? '+' : ''}{formatPercent(result.outperformance)}
                            </td>
                            <td className="text-right py-3 px-4 text-white">
                              {formatCurrency(result.total_profit)}
                            </td>
                            <td className="text-center py-3 px-4">
                              <span className={`px-3 py-1 rounded text-sm font-medium ${
                                result.recommendation === 'USE_ML' 
                                  ? 'bg-emerald-500 text-white' 
                                  : 'bg-slate-600 text-slate-300'
                              }`}>
                                {result.recommendation === 'USE_ML' ? 'Use AI' : 'Buy & Hold'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Strategy Summary */}
                <div className="bg-emerald-900/20 border border-emerald-500 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-emerald-400 mb-3">Strategy Recommendation</h3>
                  <p className="text-white">
                    Based on backtest results, use AI predictions for: 
                    <span className="font-bold text-emerald-400 ml-2">
                      {backtestResults.results
                        .filter(r => r.has_predictive_edge)
                        .map(r => r.symbol)
                        .join(', ') || 'None'}
                    </span>
                  </p>
                  <p className="text-slate-300 mt-2">
                    These stocks showed consistent predictive accuracy above 85% and outperformed buy-and-hold strategy.
                  </p>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}