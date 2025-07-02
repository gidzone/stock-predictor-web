'use client'

import { useState } from 'react'
import StockPredictor from './components/StockPredictor'
import PortfolioGuardian from './components/PortfolioGuardian'
import PortfolioUpload from './components/PortfolioUpload'
import { Brain, Shield, AlertTriangle } from 'lucide-react'

export default function Home() {
  const [activeApp, setActiveApp] = useState('alerts') // 'predictor', 'guardian', or 'alerts'

  return (
    <div className="min-h-screen">
      {/* App Selector */}
      <div className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => setActiveApp('alerts')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg transition-all ${
                activeApp === 'alerts'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <AlertTriangle size={20} />
              Volatility Alerts
            </button>
            <button
              onClick={() => setActiveApp('guardian')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg transition-all ${
                activeApp === 'guardian'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Shield size={20} />
              Portfolio Guardian
            </button>
            <button
              onClick={() => setActiveApp('predictor')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg transition-all ${
                activeApp === 'predictor'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Brain size={20} />
              Price Predictor
            </button>
          </div>
        </div>
      </div>

      {/* Active App */}
      {activeApp === 'alerts' ? <PortfolioUpload /> : activeApp === 'guardian' ? <PortfolioGuardian /> : <StockPredictor />}
    </div>
  )
}