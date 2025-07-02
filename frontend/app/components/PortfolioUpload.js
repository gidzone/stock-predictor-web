'use client'

import { useState, useEffect } from 'react'
import { Upload, Download, AlertTriangle, TrendingUp, RefreshCw, Bell } from 'lucide-react'

const API_BASE_URL = 'http://localhost:5040/api'

export default function PortfolioUpload() {
  const [file, setFile] = useState(null)
  const [portfolio, setPortfolio] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [allPositions, setAllPositions] = useState([])
  const [loading, setLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [phone, setPhone] = useState('')
  const [lastChecked, setLastChecked] = useState(null)

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

  const formatCurrency = (value) => `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  const formatPercent = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8 flex items-center gap-3">
        <AlertTriangle className="text-orange-600" />
        Volatility Alert System - Test on Your Portfolio
      </h1>

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Step 1: Upload Your Portfolio</h2>
        
        <div className="mb-4">
          <p className="text-gray-600 mb-4">
            Upload a CSV file with your positions. Format: symbol, shares, purchase_price
          </p>
          
          <div className="flex gap-4 items-center mb-4">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
              className="flex-1 p-2 border rounded"
            />
            <button
              onClick={downloadTemplate}
              className="flex items-center gap-2 text-blue-600 hover:text-blue-800"
            >
              <Download size={18} />
              Download Template
            </button>
          </div>
          
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <Upload size={18} />
            {loading ? 'Uploading...' : 'Upload Portfolio'}
          </button>
        </div>

        {/* Notification Preferences */}
        <div className="mt-6 pt-6 border-t">
          <h3 className="font-medium mb-3">Notification Preferences (Optional)</h3>
          <div className="flex gap-4">
            <input
              type="email"
              placeholder="Email for alerts"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="flex-1 px-3 py-2 border rounded"
            />
            <input
              type="tel"
              placeholder="Phone for SMS (US only)"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="flex-1 px-3 py-2 border rounded"
            />
            <button
              onClick={savePreferences}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Save
            </button>
          </div>
        </div>
      </div>

      {/* Portfolio Display */}
      {portfolio && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Your Portfolio</h2>
            <button
              onClick={checkAlerts}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
            >
              <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
              Check for Alerts
            </button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b bg-gray-50">
                  <th className="py-3 px-4">Symbol</th>
                  <th className="py-3 px-4">Shares</th>
                  <th className="py-3 px-4">Cost Basis</th>
                  <th className="py-3 px-4">Current Price</th>
                  <th className="py-3 px-4">Current Value</th>
                  <th className="py-3 px-4">Gain/Loss</th>
                  <th className="py-3 px-4">Volatility</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.map((position, idx) => {
                  const checkData = allPositions.find(p => p.symbol === position.symbol)
                  return (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium">{position.symbol}</td>
                      <td className="py-3 px-4">{position.shares}</td>
                      <td className="py-3 px-4">{formatCurrency(position.purchase_price)}</td>
                      <td className="py-3 px-4">{formatCurrency(position.current_price)}</td>
                      <td className="py-3 px-4 font-medium">{formatCurrency(position.current_value)}</td>
                      <td className={`py-3 px-4 font-medium ${position.gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatCurrency(position.gain_loss)}
                        <span className="text-sm ml-1">({formatPercent(position.gain_loss_pct)})</span>
                      </td>
                      <td className="py-3 px-4">
                        {checkData && (
                          <span className={`inline-flex items-center px-2 py-1 rounded text-sm ${
                            checkData.should_alert ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                          }`}>
                            {checkData.current_volatility?.toFixed(0)}%
                            {checkData.volatility_ratio > 1.5 && (
                              <span className="ml-1">({checkData.volatility_ratio.toFixed(1)}x)</span>
                            )}
                          </span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
              <tfoot>
                <tr className="border-t-2 font-medium">
                  <td colSpan="4" className="py-3 px-4">Total</td>
                  <td className="py-3 px-4">{formatCurrency(portfolio.reduce((sum, p) => sum + p.current_value, 0))}</td>
                  <td className={`py-3 px-4 ${
                    portfolio.reduce((sum, p) => sum + p.gain_loss, 0) >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatCurrency(portfolio.reduce((sum, p) => sum + p.gain_loss, 0))}
                  </td>
                  <td></td>
                </tr>
              </tfoot>
            </table>
          </div>
          
          {lastChecked && (
            <p className="text-sm text-gray-500 mt-2">
              Last checked: {lastChecked.toLocaleTimeString()}
            </p>
          )}
        </div>
      )}

      {/* Alerts Display */}
      {alerts.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-red-600 flex items-center gap-2">
            <Bell className="animate-pulse" />
            Active Alerts ({alerts.length})
          </h2>
          
          {alerts.map((alert, idx) => (
            <div key={idx} className={`mb-6 p-4 rounded-lg border-2 ${
              alert.severity === 'CRITICAL' ? 'bg-red-50 border-red-400' : 
              alert.severity === 'HIGH' ? 'bg-orange-50 border-orange-400' : 
              'bg-yellow-50 border-yellow-400'
            }`}>
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-semibold text-lg flex items-center gap-2">
                  <TrendingUp className="text-red-600" />
                  {alert.symbol} - {alert.severity} Volatility Alert
                </h3>
                <span className="text-sm text-gray-600">
                  {alert.volatility_ratio.toFixed(1)}x normal
                </span>
              </div>
              
              <pre className="whitespace-pre-wrap font-sans text-sm text-gray-800">
                {alert.message}
              </pre>
              
              {alert.historical_accuracy && alert.historical_accuracy.instances > 0 && (
                <div className="mt-3 pt-3 border-t text-sm text-gray-600">
                  Based on {alert.historical_accuracy.instances} similar historical patterns
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* No Alerts Message */}
      {portfolio && alerts.length === 0 && allPositions.length > 0 && (
        <div className="bg-green-50 rounded-lg p-6 text-center">
          <p className="text-lg text-green-800">
            ✅ No volatility alerts at this time. Your portfolio appears stable.
          </p>
          <p className="text-sm text-green-600 mt-2">
            We'll monitor for any volatility spikes above 2x normal levels.
          </p>
        </div>
      )}
    </div>
  )
}