import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar } from 'recharts'
import { 
  Shield, AlertTriangle, TrendingUp, TrendingDown, Activity, 
  MessageSquare, Bell, ChevronRight, X, Plus, Trash2,
  Eye, Brain, Zap, Target, AlertCircle, CheckCircle
} from 'lucide-react'

const API_BASE_URL = 'http://localhost:5040/api'

// Alert severity colors
const severityColors = {
  info: 'blue',
  warning: 'yellow',
  critical: 'red'
}

// Risk level colors
const riskLevelColors = {
  'VERY_LOW': 'green',
  'LOW': 'lime',
  'MODERATE': 'yellow',
  'HIGH': 'orange',
  'EXTREME': 'red'
}

export default function PortfolioGuardian() {
  // Portfolio state
  const [portfolio, setPortfolio] = useState([
    { symbol: 'AAPL', shares: 100, purchase_price: 150 },
    { symbol: 'SPY', shares: 50, purchase_price: 400 },
    { symbol: 'TSLA', shares: 20, purchase_price: 200 }
  ])
  const [newPosition, setNewPosition] = useState({ symbol: '', shares: '', purchase_price: '' })
  
  // Analysis state
  const [portfolioAnalysis, setPortfolioAnalysis] = useState(null)
  const [selectedPosition, setSelectedPosition] = useState(null)
  const [marketPulse, setMarketPulse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  // UI state
  const [activeView, setActiveView] = useState('overview') // overview, position, alerts, market

  // Load portfolio analysis on mount and when portfolio changes
  useEffect(() => {
    if (portfolio.length > 0) {
      monitorPortfolio()
    }
  }, [portfolio])

  // Load market pulse periodically
  useEffect(() => {
    loadMarketPulse()
    const interval = setInterval(loadMarketPulse, 60000) // Every minute
    return () => clearInterval(interval)
  }, [])

  const monitorPortfolio = async () => {
    setLoading(true)
    setError('')
    
    try {
      const response = await fetch(`${API_BASE_URL}/monitor-portfolio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ portfolio })
      })
      
      if (!response.ok) throw new Error('Failed to analyze portfolio')
      
      const data = await response.json()
      setPortfolioAnalysis(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const loadMarketPulse = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/market-pulse`)
      if (!response.ok) throw new Error('Failed to load market pulse')
      
      const data = await response.json()
      setMarketPulse(data)
    } catch (err) {
      console.error('Market pulse error:', err)
    }
  }

  const addPosition = () => {
    if (newPosition.symbol && newPosition.shares && newPosition.purchase_price) {
      setPortfolio([...portfolio, {
        symbol: newPosition.symbol.toUpperCase(),
        shares: parseFloat(newPosition.shares),
        purchase_price: parseFloat(newPosition.purchase_price)
      }])
      setNewPosition({ symbol: '', shares: '', purchase_price: '' })
    }
  }

  const removePosition = (index) => {
    setPortfolio(portfolio.filter((_, i) => i !== index))
  }

  const formatCurrency = (value) => `$${value.toLocaleString()}`
  const formatPercent = (value) => `${value.toFixed(1)}%`

  // Components for different views
  const OverviewView = () => {
    if (!portfolioAnalysis) return null
    
    const metrics = portfolioAnalysis.portfolio_metrics
    const alerts = portfolioAnalysis.alerts || []
    const criticalAlerts = alerts.filter(a => a.severity === 'critical')
    const warningAlerts = alerts.filter(a => a.severity === 'warning')
    
    return (
      <div className="space-y-6">
        {/* Portfolio Metrics */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <Shield className="text-blue-600" />
            Portfolio Protection Status
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <p className="text-sm text-blue-600">Total Value</p>
              <p className="text-2xl font-bold text-blue-800">
                {formatCurrency(metrics.total_value)}
              </p>
            </div>
            
            <div className="bg-orange-50 rounded-lg p-4">
              <p className="text-sm text-orange-600">5-Day Risk (VaR)</p>
              <p className="text-2xl font-bold text-orange-800">
                {formatCurrency(metrics.total_var_5d)}
              </p>
              <p className="text-xs text-orange-600">
                {formatPercent(metrics.var_percentage_5d)} of portfolio
              </p>
            </div>
            
            <div className="bg-purple-50 rounded-lg p-4">
              <p className="text-sm text-purple-600">Diversification Score</p>
              <p className="text-2xl font-bold text-purple-800">
                {metrics.diversification_score.toFixed(0)}/100
              </p>
            </div>
            
            <div className={`rounded-lg p-4 ${
              criticalAlerts.length > 0 ? 'bg-red-50' : 
              warningAlerts.length > 0 ? 'bg-yellow-50' : 'bg-green-50'
            }`}>
              <p className={`text-sm ${
                criticalAlerts.length > 0 ? 'text-red-600' : 
                warningAlerts.length > 0 ? 'text-yellow-600' : 'text-green-600'
              }`}>Active Alerts</p>
              <p className={`text-2xl font-bold ${
                criticalAlerts.length > 0 ? 'text-red-800' : 
                warningAlerts.length > 0 ? 'text-yellow-800' : 'text-green-800'
              }`}>
                {alerts.length}
              </p>
              {criticalAlerts.length > 0 && (
                <p className="text-xs text-red-600">{criticalAlerts.length} critical</p>
              )}
            </div>
          </div>
        </div>

        {/* Action Items */}
        {portfolioAnalysis.action_items?.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <Target className="text-red-600" />
              Action Required
            </h3>
            <div className="space-y-3">
              {portfolioAnalysis.action_items.map((item, index) => (
                <div key={index} className="flex items-start gap-3 p-3 bg-red-50 rounded-lg">
                  <AlertTriangle className="text-red-600 mt-1" size={20} />
                  <div className="flex-1">
                    <p className="font-medium text-gray-800">{item.action}</p>
                    <p className="text-sm text-gray-600">
                      {item.symbol} • {item.deadline}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Position Summary */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Position Analysis</h3>
          <div className="space-y-3">
            {portfolioAnalysis.portfolio_analysis?.map((analysis, index) => {
              if (!analysis) return null
              
              const position = portfolio[index]
              const riskScore = analysis.combined_risk_score || { level: 'UNKNOWN', score: 0 }
              const alerts = analysis.alerts || []
              
              return (
                <div 
                  key={position.symbol}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
                  onClick={() => {
                    setSelectedPosition(analysis)
                    setActiveView('position')
                  }}
                >
                  <div className="flex items-center gap-4">
                    <div>
                      <p className="font-semibold text-gray-800">{position.symbol}</p>
                      <p className="text-sm text-gray-600">
                        {position.shares} shares @ {formatCurrency(position.purchase_price)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    {alerts.length > 0 && (
                      <div className="flex items-center gap-1">
                        <Bell className={`${
                          alerts.some(a => a.severity === 'critical') ? 'text-red-500' :
                          alerts.some(a => a.severity === 'warning') ? 'text-yellow-500' :
                          'text-blue-500'
                        }`} size={20} />
                        <span className="text-sm font-medium">{alerts.length}</span>
                      </div>
                    )}
                    
                    <div className={`px-3 py-1 rounded-full text-sm font-medium text-white ${
                      riskScore.level === 'EXTREME' ? 'bg-red-500' :
                      riskScore.level === 'HIGH' ? 'bg-orange-500' :
                      riskScore.level === 'MODERATE' ? 'bg-yellow-500' :
                      riskScore.level === 'LOW' ? 'bg-lime-500' :
                      'bg-green-500'
                    }`}>
                      {riskScore.level}
                    </div>
                    
                    <ChevronRight className="text-gray-400" size={20} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    )
  }

  const PositionDetailView = () => {
    if (!selectedPosition) return null
    
    const risk = selectedPosition.risk_analysis || {}
    const sentiment = selectedPosition.sentiment_analysis || {}
    const alerts = selectedPosition.alerts || []
    const recommendations = selectedPosition.recommendations || []
    
    // Check if we have valid risk data
    const hasRiskData = risk && risk.current_volatility !== undefined
    
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800">
            {selectedPosition.symbol} Analysis
          </h2>
          <button
            onClick={() => setActiveView('overview')}
            className="text-gray-600 hover:text-gray-800"
          >
            <X size={24} />
          </button>
        </div>

        {/* Risk Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Volatility Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Volatility Forecast
            </h3>
            {hasRiskData ? (
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-600">Current Volatility</p>
                  <p className="text-xl font-bold">{formatPercent(risk.current_volatility || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">5-Day Forecast</p>
                  <p className={`text-xl font-bold ${
                    (risk.predicted_volatility_5d || 0) > (risk.current_volatility || 0) ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {formatPercent(risk.predicted_volatility_5d || 0)}
                    {(risk.predicted_volatility_5d || 0) > (risk.current_volatility || 0) ? ' ↑' : ' ↓'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Max Drawdown Risk</p>
                  <p className="text-xl font-bold text-orange-600">
                    {formatPercent(Math.abs(risk.predicted_max_drawdown_5d || 0))}
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-gray-500">Loading risk analysis...</p>
            )}
          </div>

          {/* Sentiment Analysis */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Social Sentiment
            </h3>
            {sentiment && sentiment.overall_sentiment ? (
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-600">Overall Sentiment</p>
                  <p className={`text-xl font-bold ${
                    sentiment.overall_sentiment.score > 50 ? 'text-green-600' :
                    sentiment.overall_sentiment.score < -50 ? 'text-red-600' :
                    'text-gray-600'
                  }`}>
                    {sentiment.overall_sentiment.interpretation || 'Unknown'}
                  </p>
                  <p className="text-sm text-gray-500">
                    Score: {(sentiment.overall_sentiment.score || 0).toFixed(0)}/100
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Social Activity</p>
                  <p className="text-xl font-bold">
                    {(sentiment.reddit_metrics?.mention_velocity || 0).toFixed(0)} mentions/hour
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Confidence</p>
                  <p className="text-lg font-medium">{sentiment.overall_sentiment.confidence || 'Low'}</p>
                </div>
              </div>
            ) : (
              <p className="text-gray-500">Loading sentiment analysis...</p>
            )}
          </div>
        </div>

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Active Alerts</h3>
            <div className="space-y-3">
              {alerts.map((alert, index) => (
                <div key={index} className={`p-4 rounded-lg border-l-4 ${
                  alert.severity === 'critical' ? 'bg-red-50 border-red-500' :
                  alert.severity === 'warning' ? 'bg-yellow-50 border-yellow-500' :
                  'bg-blue-50 border-blue-500'
                }`}>
                  <div className="flex items-start gap-3">
                    <AlertTriangle className={`mt-0.5 ${
                      alert.severity === 'critical' ? 'text-red-600' :
                      alert.severity === 'warning' ? 'text-yellow-600' :
                      'text-blue-600'
                    }`} size={20} />
                    <div>
                      <p className="font-medium text-gray-800">{alert.message}</p>
                      <p className="text-sm text-gray-600 mt-1">
                        {new Date(alert.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Recommendations</h3>
            <div className="space-y-3">
              {recommendations.map((rec, index) => (
                <div key={index} className="p-4 bg-blue-50 rounded-lg">
                  <p className="font-medium text-gray-800">{rec.message}</p>
                  <p className="text-sm text-gray-600 mt-1">{rec.rationale}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  const MarketPulseView = () => {
    if (!marketPulse) return null
    
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800">Market Pulse</h2>
        
        {/* Trending Tickers */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Trending on Social Media
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {marketPulse.trending_details?.map((ticker) => (
              <div key={ticker.symbol} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-semibold text-lg">{ticker.symbol}</p>
                  <span className={`px-2 py-1 rounded text-xs font-medium text-white ${
                    ticker.risk_level === 'Extreme' ? 'bg-red-500' :
                    ticker.risk_level === 'High' ? 'bg-orange-500' :
                    ticker.risk_level === 'Moderate' ? 'bg-yellow-500' :
                    'bg-green-500'
                  }`}>
                    {ticker.risk_level} Risk
                  </span>
                </div>
                <p className="text-gray-600">
                  Price: {formatCurrency(ticker.current_price)}
                </p>
                <p className="text-gray-600">
                  Volatility: {formatPercent(ticker.volatility)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-3">
            <Shield className="text-blue-600" size={40} />
            Portfolio Guardian
          </h1>
          <p className="text-gray-600 text-lg">
            AI-powered portfolio protection and risk alerts
          </p>
        </div>

        {/* Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-lg p-1 shadow-md">
            <button
              onClick={() => setActiveView('overview')}
              className={`px-6 py-3 rounded-md transition-all ${
                activeView === 'overview' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'text-gray-600 hover:text-blue-600'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveView('market')}
              className={`px-6 py-3 rounded-md transition-all ${
                activeView === 'market' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'text-gray-600 hover:text-blue-600'
              }`}
            >
              Market Pulse
            </button>
          </div>
        </div>

        {/* Portfolio Management */}
        {activeView === 'overview' && (
          <div className="max-w-4xl mx-auto mb-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Your Portfolio</h3>
              
              {/* Current Positions */}
              <div className="space-y-2 mb-4">
                {portfolio.map((position, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                    <div>
                      <span className="font-medium">{position.symbol}</span>
                      <span className="text-sm text-gray-600 ml-2">
                        {position.shares} shares @ {formatCurrency(position.purchase_price)}
                      </span>
                    </div>
                    <button
                      onClick={() => removePosition(index)}
                      className="text-red-500 hover:text-red-700"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                ))}
              </div>
              
              {/* Add Position */}
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Symbol"
                  value={newPosition.symbol}
                  onChange={(e) => setNewPosition({...newPosition, symbol: e.target.value})}
                  className="flex-1 px-3 py-2 border rounded"
                />
                <input
                  type="number"
                  placeholder="Shares"
                  value={newPosition.shares}
                  onChange={(e) => setNewPosition({...newPosition, shares: e.target.value})}
                  className="w-24 px-3 py-2 border rounded"
                />
                <input
                  type="number"
                  placeholder="Price"
                  value={newPosition.purchase_price}
                  onChange={(e) => setNewPosition({...newPosition, purchase_price: e.target.value})}
                  className="w-24 px-3 py-2 border rounded"
                />
                <button
                  onClick={addPosition}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  <Plus size={20} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="max-w-4xl mx-auto mb-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="text-red-500" size={20} />
              <span className="text-red-700">{error}</span>
            </div>
          </div>
        )}

        {/* Main Content */}
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : (
          <>
            {activeView === 'overview' && <OverviewView />}
            {activeView === 'position' && <PositionDetailView />}
            {activeView === 'market' && <MarketPulseView />}
          </>
        )}
      </div>
    </div>
  )
}