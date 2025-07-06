'use client'

import React, { useState, useEffect } from 'react'
import { useUser } from '@clerk/nextjs'
import { Upload, Download, AlertTriangle, TrendingUp, RefreshCw, Bell, Activity, ChevronRight } from 'lucide-react'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5040/api'

export default function PortfolioUpload() {
  const [file, setFile] = useState(null)
  const [portfolio, setPortfolio] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [allPositions, setAllPositions] = useState([])
  const [loading, setLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [phone, setPhone] = useState('')
  const [lastChecked, setLastChecked] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [refreshInterval, setRefreshInterval] = useState(null)
  const [expandedRows, setExpandedRows] = useState(new Set())

  const toggleRowExpansion = (symbol) => {
    const newExpanded = new Set(expandedRows)
    if (newExpanded.has(symbol)) {
      newExpanded.delete(symbol)
    } else {
      newExpanded.add(symbol)
    }
    setExpandedRows(newExpanded)
  }

  useEffect(() => {
    if (autoRefresh && portfolio) {
      const interval = setInterval(() => {
        checkAlerts()
      }, 60000) // Check every minute
      setRefreshInterval(interval)
    } else if (refreshInterval) {
      clearInterval(refreshInterval)
      setRefreshInterval(null)
    }
    
    return () => {
      if (refreshInterval) clearInterval(refreshInterval)
    }
  }, [autoRefresh, portfolio])

  const handleUpload = async () => {
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/upload-portfolio`, {
        method: 'POST',
        body: formData
      })
      
      const data = await response.json()
      if (data.success) {
        setPortfolio(data.portfolio)
        // Check for alerts immediately after upload
        setTimeout(checkAlerts, 1000)
      } else {
        alert(data.error || 'Upload failed')
      }
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Upload failed. Check console for details.')
    } finally {
      setLoading(false)
    }
  }

  const checkAlerts = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/check-volatility-alerts`)
      const data = await response.json()
      
      if (data.error) {
        alert(data.error)
        return
      }
      
      setAlerts(data.alerts || [])
      setAllPositions(data.all_positions || [])
      setLastChecked(new Date())
      
      // Show notification if critical alerts
      const criticalAlerts = data.alerts.filter(a => a.severity === 'CRITICAL')
      if (criticalAlerts.length > 0) {
        alert(`🚨 ${criticalAlerts.length} CRITICAL alerts require immediate attention!`)
      }
    } catch (error) {
      console.error('Alert check failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const savePreferences = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/save-alert-preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, phone })
      })
      
      if (response.ok) {
        alert('Preferences saved!')
      }
    } catch (error) {
      console.error('Failed to save preferences:', error)
    }
  }

  const downloadTemplate = () => {
    const csv = 'symbol,shares,purchase_price\nAAPL,100,150.50\nTSLA,50,250.00\nNVDA,30,450.00'
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'portfolio_template.csv'
    a.click()
  }

  const formatCurrency = (value) => {
    if (value === undefined || value === null) return '$0.00'
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const formatPercent = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`

  const exportVolatilityData = () => {
    const csvData = allPositions.map(pos => ({
      Date: new Date().toLocaleString(),
      Symbol: pos.symbol,
      'Current Vol %': pos.current_volatility?.toFixed(2) || '',
      'Historical Vol %': pos.historical_volatility?.toFixed(2) || '',
      'Spike Ratio': pos.volatility_ratio?.toFixed(2) || '',
      'Position Value': pos.position_value?.toFixed(2) || '',
      'Should Alert': pos.should_alert ? 'YES' : 'NO',
      'Distance to Alert': pos.volatility_ratio < 2 
        ? `${((2 - pos.volatility_ratio) / 2 * 100).toFixed(0)}% away`
        : 'TRIGGERED'
    }))
    
    const headers = Object.keys(csvData[0]).join(',')
    const rows = csvData.map(row => Object.values(row).join(','))
    const csv = [headers, ...rows].join('\n')
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `volatility_analysis_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="max-w-6xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-8 flex items-center gap-3 text-white">
          <AlertTriangle className="text-yellow-400" />
          Volatility Alert System
        </h1>

        {/* Upload Section */}
        <div className="bg-slate-800 rounded-lg shadow-lg p-6 mb-6 border border-slate-700">
          <h2 className="text-xl font-semibold mb-4 text-white">Upload Your Portfolio</h2>
          
          <div className="mb-4">
            <p className="text-slate-300 mb-4">
              Upload a CSV file with your positions. Format: symbol, shares, purchase_price
            </p>
            
            <div className="flex gap-4 items-center mb-4">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files[0])}
                className="flex-1 p-2 border border-slate-600 rounded bg-slate-700 text-white file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-emerald-500 file:text-white hover:file:bg-emerald-600"
              />
              <button
                onClick={downloadTemplate}
                className="flex items-center gap-2 text-emerald-400 hover:text-emerald-300"
              >
                <Download size={18} />
                Download Template
              </button>
            </div>
            
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="flex items-center gap-2 px-6 py-3 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 transition-all"
            >
              <Upload size={18} />
              {loading ? 'Uploading...' : 'Upload Portfolio'}
            </button>
          </div>

          {/* Notification Preferences */}
          <div className="mt-6 pt-6 border-t border-slate-700">
            <h3 className="font-medium mb-3 text-white">Notification Preferences (Optional)</h3>
            <div className="flex gap-4">
              <input
                type="email"
                placeholder="Email for alerts"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="flex-1 px-3 py-2 border border-slate-600 rounded bg-slate-700 text-white placeholder-slate-400"
              />
              <input
                type="tel"
                placeholder="Phone for SMS (US only)"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                className="flex-1 px-3 py-2 border border-slate-600 rounded bg-slate-700 text-white placeholder-slate-400"
              />
              <button
                onClick={savePreferences}
                className="px-4 py-2 bg-slate-600 text-white rounded hover:bg-slate-700 transition-all"
              >
                Save
              </button>
            </div>
          </div>
        </div>

        {/* Portfolio Display */}
        {portfolio && (
          <div className="bg-slate-800 rounded-lg shadow-lg p-6 mb-6 border border-slate-700">
            <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
              <h2 className="text-xl font-semibold text-white">Your Portfolio</h2>
              <div className="flex items-center gap-4">
                <button
                  onClick={checkAlerts}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded hover:bg-emerald-600 disabled:opacity-50 transition-all"
                >
                  <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
                  Check for Alerts
                </button>
                <label className="flex items-center gap-2 text-slate-300">
                  <input
                    type="checkbox"
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                    className="rounded border-slate-600 bg-slate-700"
                  />
                  <span className="text-sm">Auto-refresh (1 min)</span>
                </label>
                <button
                  onClick={exportVolatilityData}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded hover:bg-slate-700 transition-all"
                >
                  <Download size={18} />
                  Export Analysis
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-slate-700 bg-slate-900/50">
                    <th className="py-3 px-4 text-slate-300">Symbol</th>
                    <th className="py-3 px-4 text-slate-300">Shares</th>
                    <th className="py-3 px-4 text-slate-300">Cost Basis</th>
                    <th className="py-3 px-4 text-slate-300">Current Price</th>
                    <th className="py-3 px-4 text-slate-300">Current Value</th>
                    <th className="py-3 px-4 text-slate-300">Gain/Loss</th>
                    <th className="py-3 px-4 text-slate-300">Volatility</th>
                    <th className="py-3 px-4 w-10"></th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.map((position, idx) => {
                    const checkData = allPositions.find(p => p.symbol === position.symbol)
                    const isExpanded = expandedRows.has(position.symbol)
                    const isElevated = checkData?.volatility_ratio > 1.5
                    const isNearAlert = checkData?.volatility_ratio > 1.8
                    
                    return (
                      <React.Fragment key={idx}>
                        <tr 
                          className={`border-b border-slate-700 hover:bg-slate-700/30 cursor-pointer transition-colors ${
                            checkData?.should_alert ? 'bg-red-900/20' : ''
                          }`}
                          onClick={() => toggleRowExpansion(position.symbol)}
                        >
                          <td className="py-3 px-4 font-medium text-white">{position.symbol}</td>
                          <td className="py-3 px-4 text-white">{position.shares}</td>
                          <td className="py-3 px-4 text-white">{formatCurrency(position.purchase_price)}</td>
                          <td className="py-3 px-4 text-white">{formatCurrency(position.current_price)}</td>
                          <td className="py-3 px-4 font-medium text-white">{formatCurrency(position.current_value)}</td>
                          <td className={`py-3 px-4 font-medium ${position.gain_loss >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {formatCurrency(position.gain_loss)}
                            <span className="text-sm ml-1">({formatPercent(position.gain_loss_pct)})</span>
                          </td>
                          <td className="py-3 px-4">
                            {checkData && (
                              <span className={`inline-flex items-center px-2 py-1 rounded text-sm ${
                                checkData.should_alert ? 'bg-red-900/50 text-red-400 border border-red-700' : 
                                isNearAlert ? 'bg-yellow-900/50 text-yellow-400 border border-yellow-700' :
                                isElevated ? 'bg-orange-900/50 text-orange-400 border border-orange-700' :
                                'bg-emerald-900/50 text-emerald-400 border border-emerald-700'
                              }`}>
                                {checkData.current_volatility?.toFixed(0)}%
                                {checkData.volatility_ratio > 1.5 && (
                                  <span className="ml-1">({checkData.volatility_ratio.toFixed(1)}x)</span>
                                )}
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-4">
                            <ChevronRight 
                              size={20} 
                              className={`transform transition-transform text-slate-400 ${isExpanded ? 'rotate-90' : ''}`}
                            />
                          </td>
                        </tr>
                        
                        {/* Expandable Volatility Analysis Row */}
                        {isExpanded && checkData && (
                          <tr>
                            <td colSpan="8" className="p-0">
                              <div className="bg-slate-900/50 p-6 border-b border-slate-700">
                                <h4 className="font-semibold text-lg mb-4 flex items-center gap-2 text-white">
                                  <Activity className="text-emerald-400" size={20} />
                                  Volatility Analysis for {position.symbol}
                                </h4>
                                
                                {/* Volatility Metrics Grid */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                                  <div className="bg-slate-800 p-3 rounded border border-slate-700">
                                    <p className="text-xs text-slate-400">Current Vol (5-day)</p>
                                    <p className="text-xl font-semibold text-white">
                                      {checkData.current_volatility?.toFixed(1) || 'N/A'}%
                                    </p>
                                  </div>
                                  
                                  <div className="bg-slate-800 p-3 rounded border border-slate-700">
                                    <p className="text-xs text-slate-400">Historical Vol (20-day)</p>
                                    <p className="text-xl font-semibold text-white">
                                      {checkData.historical_volatility?.toFixed(1) || 'N/A'}%
                                    </p>
                                  </div>
                                  
                                  <div className="bg-slate-800 p-3 rounded border border-slate-700">
                                    <p className="text-xs text-slate-400">Spike Ratio</p>
                                    <p className={`text-xl font-semibold ${
                                      checkData.volatility_ratio > 2 ? 'text-red-400' :
                                      checkData.volatility_ratio > 1.5 ? 'text-orange-400' :
                                      'text-emerald-400'
                                    }`}>
                                      {checkData.volatility_ratio?.toFixed(2) || 'N/A'}x
                                    </p>
                                  </div>
                                  
                                  <div className="bg-slate-800 p-3 rounded border border-slate-700">
                                    <p className="text-xs text-slate-400">Alert Status</p>
                                    <p className={`text-xl font-semibold ${
                                      checkData.should_alert ? 'text-red-400' :
                                      isNearAlert ? 'text-yellow-400' :
                                      'text-emerald-400'
                                    }`}>
                                      {checkData.should_alert ? '🚨 ALERT' :
                                       isNearAlert ? '⚠️ WATCH' :
                                       '✅ STABLE'}
                                    </p>
                                  </div>
                                </div>
                                
                                {/* Progress Bar to Alert */}
                                <div className="mb-4">
                                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                                    <span>Normal</span>
                                    <span>Alert Level (2.0x)</span>
                                  </div>
                                  <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                                    <div 
                                      className={`h-full transition-all ${
                                        checkData.volatility_ratio > 2 ? 'bg-red-500' :
                                        checkData.volatility_ratio > 1.5 ? 'bg-orange-500' :
                                        'bg-emerald-500'
                                      }`}
                                      style={{ width: `${Math.min((checkData.volatility_ratio / 2) * 100, 100)}%` }}
                                    />
                                  </div>
                                  <p className="text-sm text-slate-400 mt-1">
                                    {checkData.volatility_ratio < 2 
                                      ? `${((2 - checkData.volatility_ratio) / 2 * 100).toFixed(0)}% away from alert threshold`
                                      : `${((checkData.volatility_ratio - 2) / 2 * 100).toFixed(0)}% above alert threshold`}
                                  </p>
                                </div>
                                
                                {/* Historical Context */}
                                {checkData.historical_accuracy && checkData.historical_accuracy.instances > 0 && (
                                  <div className="bg-blue-900/20 p-4 rounded-lg border border-blue-700">
                                    <p className="font-medium text-blue-400 mb-1">Historical Pattern Analysis</p>
                                    <p className="text-sm text-blue-300">
                                      When volatility spiked to similar levels in the past ({checkData.historical_accuracy.instances} occurrences):
                                    </p>
                                    <ul className="text-sm text-blue-300 mt-2 ml-4 list-disc">
                                      <li>{checkData.historical_accuracy.drop_rate?.toFixed(0)}% of the time resulted in &gt;5% drops</li>
                                      <li>Average move: {checkData.historical_accuracy.avg_move?.toFixed(1)}%</li>
                                      <li>Worst case: {checkData.historical_accuracy.worst_move?.toFixed(1)}%</li>
                                    </ul>
                                  </div>
                                )}
                                
                                {/* Additional Metrics */}
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4 text-sm">
                                  <div>
                                    <span className="text-slate-400">Baseline Vol (60d):</span>
                                    <span className="ml-2 font-medium text-white">{checkData.baseline_volatility?.toFixed(1)}%</span>
                                  </div>
                                  <div>
                                    <span className="text-slate-400">Position at Risk:</span>
                                    <span className="ml-2 font-medium text-white">{formatCurrency(checkData.position_value)}</span>
                                  </div>
                                  <div>
                                    <span className="text-slate-400">Severity Level:</span>
                                    <span className={`ml-2 font-medium ${
                                      checkData.severity === 'CRITICAL' ? 'text-red-400' :
                                      checkData.severity === 'HIGH' ? 'text-orange-400' :
                                      'text-yellow-400'
                                    }`}>{checkData.severity}</span>
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    )
                  })}
                </tbody>
                <tfoot>
                  <tr className="border-t-2 border-slate-700 font-medium">
                    <td colSpan="4" className="py-3 px-4 text-white">Total</td>
                    <td className="py-3 px-4 text-white">{formatCurrency(portfolio.reduce((sum, p) => sum + p.current_value, 0))}</td>
                    <td className={`py-3 px-4 ${
                      portfolio.reduce((sum, p) => sum + p.gain_loss, 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {formatCurrency(portfolio.reduce((sum, p) => sum + p.gain_loss, 0))}
                    </td>
                    <td colSpan="2"></td>
                  </tr>
                </tfoot>
              </table>
            </div>
            
            {lastChecked && (
              <p className="text-sm text-slate-400 mt-2">
                Last checked: {lastChecked.toLocaleTimeString()}
              </p>
            )}
          </div>
        )}

        {/* Understanding Volatility Metrics */}
        <div className="bg-blue-900/20 rounded-lg p-4 mb-6 border border-blue-700">
          <h3 className="font-semibold text-blue-400 mb-2">Understanding Volatility Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-300">
            <div>
              <p><strong>Current Volatility:</strong> How much the stock moved in the last 5 days (annualized %)</p>
              <p><strong>Historical Volatility:</strong> Average movement over 20 days</p>
            </div>
            <div>
              <p><strong>Spike Ratio:</strong> Current ÷ Historical (2.0x = alert threshold)</p>
              <p><strong>Alert Levels:</strong> 🟢 Normal (&lt;1.5x) 🟡 Elevated (1.5x-2.0x) 🔴 Alert (&gt;2x)</p>
            </div>
          </div>
        </div>

        {/* Alerts Display */}
        {alerts.length > 0 && (
          <div className="bg-slate-800 rounded-lg shadow-lg p-6 border border-slate-700">
            <h2 className="text-xl font-semibold mb-4 text-red-400 flex items-center gap-2">
              <Bell className="animate-pulse" />
              Active Alerts ({alerts.length})
            </h2>
            
            {alerts.map((alert, idx) => (
              <div key={idx} className={`mb-6 p-4 rounded-lg border-2 ${
                alert.severity === 'CRITICAL' ? 'bg-red-900/20 border-red-500' : 
                alert.severity === 'HIGH' ? 'bg-orange-900/20 border-orange-500' : 
                'bg-yellow-900/20 border-yellow-500'
              }`}>
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-lg flex items-center gap-2 text-white">
                    <TrendingUp className="text-red-400" />
                    {alert.symbol} - {alert.severity} Volatility Alert
                  </h3>
                  <span className="text-sm text-slate-300">
                    {alert.volatility_ratio.toFixed(1)}x normal
                  </span>
                </div>
                
                <pre className="whitespace-pre-wrap font-sans text-sm text-slate-200">
                  {alert.message}
                </pre>
                
                {alert.historical_accuracy && alert.historical_accuracy.instances > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-700 text-sm text-slate-400">
                    Based on {alert.historical_accuracy.instances} similar historical patterns
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* No Alerts Message */}
        {portfolio && alerts.length === 0 && allPositions.length > 0 && (
          <div className="bg-emerald-900/20 rounded-lg p-6 text-center border border-emerald-700">
            <p className="text-lg text-emerald-400">
              ✅ No volatility alerts at this time. Your portfolio appears stable.
            </p>
            <p className="text-sm text-emerald-300 mt-2">
              We'll monitor for any volatility spikes above 2x normal levels.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}