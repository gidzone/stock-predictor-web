'use client'

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
  'VERY_LOW': 'emerald',
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
        {/* Portfolio Summary */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <div className="flex items-center gap-3">
              <Shield className="text-emerald-400" size={24} />
              <div>
                <p className="text-sm text-slate-400">Total Value</p>
                <p className="text-xl font-bold text-white">
                  {formatCurrency(metrics.total_value)}
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <div className="flex items-center gap-3">
              <AlertTriangle className="text-yellow-400" size={24} />
              <div>
                <p className="text-sm text-slate-400">Risk Level</p>
                <p className="text-xl font-bold text-white">
                  {metrics.var_percentage_20d.toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <div className="flex items-center gap-3">
              <Bell className="text-red-400" size={24} />
              <div>
                <p className="text-sm text-slate-400">Active Alerts</p>
                <p className="text-xl font-bold text-white">
                  {criticalAlerts.length} Critical
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <div className="flex items-center gap-3">
              <Activity className="text-emerald-400" size={24} />
              <div>
                <p className="text-sm text-slate-400">Diversification</p>
                <p className="text-xl font-bold text-white">
                  {metrics.diversification_score.toFixed(1)}/10
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="bg-red-900/20 rounded-lg p-6 border border-red-500">
            <h3 className="text-lg font-semibold text-red-400 mb-4 flex items-center gap-2">
              <AlertTriangle size={20} />
              Active Alerts ({alerts.length})
            </h3>
            <div className="space-y-3">
              {alerts.map((alert, idx) => (
                <div key={idx} className={`p-4 rounded-lg border ${
                  alert.severity === 'critical' 
                    ? 'bg-red-900/20 border-red-700' 
                    : alert.severity === 'warning'
                    ? 'bg-yellow-900/20 border-yellow-700'
                    : 'bg-blue-900/20 border-blue-700'
                }`}>
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-medium text-white">{alert.symbol}: {alert.type}</h4>
                      <p className="text-slate-300">{alert.message}</p>
                      {alert.recommendation && (
                        <p className="text-sm text-emerald-400 mt-2">
                          Action: {alert.recommendation}
                        </p>
                      )}
                    </div>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      alert.severity === 'critical' 
                        ? 'bg-red-500 text-white' 
                        : alert.severity === 'warning'
                        ? 'bg-yellow-500 text-black'
                        : 'bg-blue-500 text-white'
                    }`}>
                      {alert.severity.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Position Analysis */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Position Analysis</h3>
          <div className="space-y-4">
            {portfolioAnalysis.portfolio_analysis.map((position, idx) => (
              <div key={idx} className="border border-slate-700 rounded-lg p-4 hover:bg-slate-700/50 transition-colors">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <h4 className="font-bold text-lg text-white">{position.symbol}</h4>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      position.combined_risk.level === 'EXTREME' 
                        ? 'bg-red-500 text-white' 
                        : position.combined_risk.level === 'HIGH'
                        ? 'bg-orange-500 text-white'
                        : position.combined_risk.level === 'MODERATE'
                        ? 'bg-yellow-500 text-black'
                        : 'bg-emerald-500 text-white'
                    }`}>
                      {position.combined_risk.level} RISK
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedPosition(position)}
                    className="text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                  >
                    View Details <ChevronRight size={16} />
                  </button>
                </div>
                
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-slate-400">Current Price</p>
                    <p className="font-medium text-white">{formatCurrency(position.current_price)}</p>
                  </div>
                  <div>
                    <p className="text-slate-400">Volatility</p>
                    <p className="font-medium text-white">{formatPercent(position.risk_metrics.volatility)}</p>
                  </div>
                  <div>
                    <p className="text-slate-400">Risk Score</p>
                    <p className="font-medium text-white">{position.combined_risk.score}/100</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Action Items */}
        {portfolioAnalysis.action_items?.length > 0 && (
          <div className="bg-emerald-900/20 rounded-lg p-6 border border-emerald-500">
            <h3 className="text-lg font-semibold text-emerald-400 mb-4">Recommended Actions</h3>
            <ul className="space-y-2">
              {portfolioAnalysis.action_items.map((item, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <CheckCircle className="text-emerald-400 mt-0.5" size={16} />
                  <span className="text-white">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    )
  }

  const MarketPulseView = () => {
    if (!marketPulse) return null
    
    return (
      <div className="space-y-6">
        {/* Market Overview */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Market Overview</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <p className="text-sm text-slate-400 mb-2">Market Sentiment</p>
              <div className="flex items-center gap-3">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                  marketPulse.market_sentiment.overall_score > 60 
                    ? 'bg-emerald-500' 
                    : marketPulse.market_sentiment.overall_score < 40
                    ? 'bg-red-500'
                    : 'bg-yellow-500'
                }`}>
                  <span className="text-xl font-bold text-white">
                    {marketPulse.market_sentiment.overall_score}
                  </span>
                </div>
                <div>
                  <p className="font-medium text-white">
                    {marketPulse.market_sentiment.sentiment}
                  </p>
                  <p className="text-sm text-slate-400">
                    {marketPulse.market_sentiment.trend}
                  </p>
                </div>
              </div>
            </div>
            
            <div>
              <p className="text-sm text-slate-400 mb-2">Volatility Index</p>
              <p className="text-2xl font-bold text-white">
                {marketPulse.volatility_index.vix}
              </p>
              <p className={`text-sm ${
                marketPulse.volatility_index.level === 'High' 
                  ? 'text-red-400' 
                  : marketPulse.volatility_index.level === 'Low'
                  ? 'text-emerald-400'
                  : 'text-yellow-400'
              }`}>
                {marketPulse.volatility_index.level} - {marketPulse.volatility_index.interpretation}
              </p>
            </div>
            
            <div>
              <p className="text-sm text-slate-400 mb-2">Market Trend</p>
              <div className="flex items-center gap-2">
                {marketPulse.market_sentiment.overall_score > 50 ? (
                  <TrendingUp className="text-emerald-400" size={24} />
                ) : (
                  <TrendingDown className="text-red-400" size={24} />
                )}
                <span className="text-lg font-medium text-white">
                  {marketPulse.market_sentiment.overall_score > 60 
                    ? 'Bullish' 
                    : marketPulse.market_sentiment.overall_score < 40
                    ? 'Bearish'
                    : 'Neutral'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Trending Risks */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Trending Market Risks</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {marketPulse.trending_risks.map((risk, idx) => (
              <div key={idx} className="flex items-start gap-3 p-4 bg-slate-700 rounded-lg">
                <AlertTriangle className={`mt-1 ${
                  risk.severity === 'High' 
                    ? 'text-red-400' 
                    : risk.severity === 'Medium'
                    ? 'text-yellow-400'
                    : 'text-blue-400'
                }`} size={20} />
                <div>
                  <h4 className="font-medium text-white">{risk.risk}</h4>
                  <p className="text-sm text-slate-300 mt-1">{risk.description}</p>
                  <p className="text-xs text-slate-400 mt-2">Impact: {risk.impact}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hot Tickers */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Market Movers</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {marketPulse.hot_tickers.map((ticker, idx) => (
              <div key={idx} className="p-4 bg-slate-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-bold text-lg text-white">{ticker.symbol}</h4>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    ticker.risk_level === 'Extreme' ? 'bg-red-500 text-white' :
                    ticker.risk_level === 'High' ? 'bg-orange-500 text-white' :
                    ticker.risk_level === 'Moderate' ? 'bg-yellow-500 text-black' :
                    'bg-emerald-500 text-white'
                  }`}>
                    {ticker.risk_level} Risk
                  </span>
                </div>
                <p className="text-slate-300">
                  Price: {formatCurrency(ticker.current_price)}
                </p>
                <p className="text-slate-300">
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
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Shield className="text-emerald-400" size={40} />
            Portfolio Guardian
          </h1>
          <p className="text-slate-300 text-lg">
            AI-powered portfolio protection and risk alerts
          </p>
        </div>

        {/* Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-slate-800 rounded-lg p-1 shadow-md border border-slate-700">
            <button
              onClick={() => setActiveView('overview')}
              className={`px-6 py-3 rounded-md transition-all ${
                activeView === 'overview' 
                  ? 'bg-emerald-500 text-white shadow-md' 
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveView('market')}
              className={`px-6 py-3 rounded-md transition-all ${
                activeView === 'market' 
                  ? 'bg-emerald-500 text-white shadow-md' 
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              Market Pulse
            </button>
          </div>
        </div>

        {/* Portfolio Management */}
        {activeView === 'overview' && (
          <div className="max-w-4xl mx-auto mb-6">
            <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
              <h3 className="text-lg font-semibold text-white mb-4">Your Portfolio</h3>
              
              {/* Current Positions */}
              <div className="space-y-2 mb-4">
                {portfolio.map((position, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-slate-700 rounded">
                    <div>
                      <span className="font-medium text-white">{position.symbol}</span>
                      <span className="text-sm text-slate-300 ml-2">
                        {position.shares} shares @ {formatCurrency(position.purchase_price)}
                      </span>
                    </div>
                    <button
                      onClick={() => removePosition(index)}
                      className="text-red-400 hover:text-red-300"
                    >
                      <Trash2 size={16} />
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
                  className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-400"
                />
                <input
                  type="number"
                  placeholder="Shares"
                  value={newPosition.shares}
                  onChange={(e) => setNewPosition({...newPosition, shares: e.target.value})}
                  className="w-24 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-400"
                />
                <input
                  type="number"
                  placeholder="Price"
                  value={newPosition.purchase_price}
                  onChange={(e) => setNewPosition({...newPosition, purchase_price: e.target.value})}
                  className="w-24 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-400"
                />
                <button
                  onClick={addPosition}
                  className="px-4 py-2 bg-emerald-500 text-white rounded hover:bg-emerald-600 transition-all"
                >
                  <Plus size={16} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="max-w-4xl mx-auto mb-6">
            <div className="bg-red-900/20 border border-red-500 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="text-red-400" size={24} />
              <p className="text-red-300">{error}</p>
            </div>
          </div>
        )}

        {/* Main Content Area */}
        <div className="max-w-6xl mx-auto">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400 mx-auto mb-4"></div>
                <p className="text-slate-300">Analyzing portfolio...</p>
              </div>
            </div>
          ) : activeView === 'overview' ? (
            <OverviewView />
          ) : (
            <MarketPulseView />
          )}
        </div>

        {/* Position Detail Modal */}
        {selectedPosition && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-slate-900 rounded-2xl p-8 max-w-4xl w-full border border-slate-700 relative max-h-[90vh] overflow-y-auto">
              <button 
                onClick={() => setSelectedPosition(null)}
                className="absolute top-6 right-6 text-slate-400 hover:text-white"
              >
                <X size={24} />
              </button>
              
              <h3 className="text-2xl font-bold text-white mb-6">
                {selectedPosition.symbol} Analysis
              </h3>
              
              {/* Risk Overview */}
              <div className="grid grid-cols-2 gap-6 mb-6">
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <h4 className="text-lg font-medium text-white mb-4">Risk Metrics</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Volatility</span>
                      <span className="text-white font-medium">
                        {formatPercent(selectedPosition.risk_metrics.volatility)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Beta</span>
                      <span className="text-white font-medium">
                        {selectedPosition.risk_metrics.beta.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sharpe Ratio</span>
                      <span className="text-white font-medium">
                        {selectedPosition.risk_metrics.sharpe_ratio.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Max Drawdown</span>
                      <span className="text-red-400 font-medium">
                        {formatPercent(selectedPosition.risk_metrics.max_drawdown)}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <h4 className="text-lg font-medium text-white mb-4">Sentiment Analysis</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">News Sentiment</span>
                      <span className={`font-medium ${
                        selectedPosition.sentiment_data.news_sentiment.score > 0
                          ? 'text-emerald-400'
                          : 'text-red-400'
                      }`}>
                        {selectedPosition.sentiment_data.news_sentiment.score > 0 ? 'Positive' : 'Negative'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Social Sentiment</span>
                      <span className={`font-medium ${
                        selectedPosition.sentiment_data.social_sentiment.score > 0
                          ? 'text-emerald-400'
                          : 'text-red-400'
                      }`}>
                        {selectedPosition.sentiment_data.social_sentiment.score > 0 ? 'Positive' : 'Negative'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Overall Score</span>
                      <span className="text-white font-medium">
                        {selectedPosition.sentiment_data.overall_sentiment.score}/100
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Alerts for this position */}
              {selectedPosition.alerts?.length > 0 && (
                <div className="bg-red-900/20 rounded-lg p-6 border border-red-500 mb-6">
                  <h4 className="text-lg font-medium text-red-400 mb-3">Active Alerts</h4>
                  <ul className="space-y-2">
                    {selectedPosition.alerts.map((alert, idx) => (
                      <li key={idx} className="text-white">
                        • {alert.message}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}