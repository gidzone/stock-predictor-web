'use client'
import { useState } from 'react'
import { useUser, SignInButton, UserButton } from '@clerk/nextjs'
import PortfolioUpload from './components/PortfolioUpload'
import LandingPage from './components/LandingPage'
import { Shield } from 'lucide-react'

export default function Home() {
  const { isLoaded, isSignedIn, user } = useUser()
  
  // Show loading state while Clerk loads
  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400 mx-auto mb-4"></div>
          <p className="text-slate-300">Loading...</p>
        </div>
      </div>
    )
  }
  
  // Show landing page if not signed in
  if (!isSignedIn) {
    return <LandingPage />
  }
  
  // Show app if signed in
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      {/* App Header */}
      <div className="bg-slate-900 shadow-sm border-b border-slate-700">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="text-emerald-400" size={32} />
              <h1 className="text-2xl font-bold text-white">Portfolio Watchdog</h1>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-slate-300 text-sm">
                {user.primaryEmailAddress?.emailAddress}
              </span>
              <UserButton 
                afterSignOutUrl="/"
                appearance={{
                  elements: {
                    avatarBox: "w-10 h-10"
                  }
                }}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Volatility Alerts App */}
      <PortfolioUpload />
    </div>
  )
}