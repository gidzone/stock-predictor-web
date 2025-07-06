'use client'
import { useState } from 'react'
import { useUser } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'
import { Shield, Upload, CreditCard, ArrowRight, Check, CheckCircle } from 'lucide-react'

export default function OnboardingPage() {
  const { user } = useUser()
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState(1)
  const [selectedPlan, setSelectedPlan] = useState('essential')
  
  const steps = [
    { number: 1, title: 'Welcome', icon: Shield },
    { number: 2, title: 'Choose Plan', icon: CreditCard },
    { number: 3, title: 'Upload Portfolio', icon: Upload }
  ]
  
  const handlePlanSelection = (plan) => {
    setSelectedPlan(plan)
    // In production, this would create a Stripe checkout session
    console.log(`Selected plan: ${plan}`)
  }
  
  const handleComplete = () => {
    // In production, save onboarding status to user metadata
    router.push('/')
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Progress Bar */}
        <div className="max-w-3xl mx-auto mb-8">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => (
              <div key={step.number} className="flex items-center">
                <div className={`
                  flex items-center justify-center w-10 h-10 rounded-full
                  ${currentStep >= step.number 
                    ? 'bg-emerald-500 text-white' 
                    : 'bg-slate-700 text-slate-400'}
                `}>
                  {currentStep > step.number ? (
                    <Check size={20} />
                  ) : (
                    <span>{step.number}</span>
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-full h-1 mx-2 ${
                    currentStep > step.number ? 'bg-emerald-500' : 'bg-slate-700'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between text-sm">
            {steps.map(step => (
              <span key={step.number} className={
                currentStep >= step.number ? 'text-white' : 'text-slate-400'
              }>
                {step.title}
              </span>
            ))}
          </div>
        </div>
        
        {/* Step Content */}
        <div className="max-w-3xl mx-auto">
          {/* Step 1: Welcome */}
          {currentStep === 1 && (
            <div className="bg-slate-800 rounded-lg p-8 border border-slate-700 text-center">
              <Shield className="text-emerald-400 mx-auto mb-6" size={64} />
              <h2 className="text-3xl font-bold text-white mb-4">
                Welcome to Portfolio Watchdog, {user?.firstName || 'Investor'}!
              </h2>
              <p className="text-xl text-slate-300 mb-8">
                Let's get you set up in less than 2 minutes.
              </p>
              <div className="space-y-4 text-left max-w-md mx-auto mb-8">
                <div className="flex items-start gap-3">
                  <CheckCircle className="text-emerald-400 mt-1" size={20} />
                  <div>
                    <p className="text-white font-medium">Choose your protection level</p>
                    <p className="text-slate-400 text-sm">Essential or Premium monitoring</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="text-emerald-400 mt-1" size={20} />
                  <div>
                    <p className="text-white font-medium">Upload your portfolio</p>
                    <p className="text-slate-400 text-sm">CSV file or manual entry</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="text-emerald-400 mt-1" size={20} />
                  <div>
                    <p className="text-white font-medium">Start monitoring</p>
                    <p className="text-slate-400 text-sm">Get alerts before major drops</p>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setCurrentStep(2)}
                className="bg-emerald-500 hover:bg-emerald-600 px-8 py-3 rounded-lg font-semibold text-white transition-all flex items-center gap-2 mx-auto"
              >
                Get Started <ArrowRight size={20} />
              </button>
            </div>
          )}
          
          {/* Step 2: Choose Plan */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-white mb-4">
                  Choose Your Protection Level
                </h2>
                <p className="text-lg text-slate-300">
                  First month free • No credit card required • Cancel anytime
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                {/* Essential Plan */}
                <div 
                  className={`bg-slate-800 rounded-xl p-6 border-2 cursor-pointer transition-all ${
                    selectedPlan === 'essential' 
                      ? 'border-emerald-500' 
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                  onClick={() => setSelectedPlan('essential')}
                >
                  <h3 className="text-2xl font-bold text-white mb-2">Essential</h3>
                  <p className="text-3xl font-bold text-emerald-400 mb-4">
                    $29<span className="text-lg text-slate-400">/month</span>
                  </p>
                  <ul className="space-y-2 text-slate-300">
                    <li className="flex items-center gap-2">
                      <Check className="text-emerald-400" size={16} />
                      Email alerts for volatility spikes
                    </li>
                    <li className="flex items-center gap-2">
                      <Check className="text-emerald-400" size={16} />
                      Monitor up to 20 stocks
                    </li>
                    <li className="flex items-center gap-2">
                      <Check className="text-emerald-400" size={16} />
                      48-hour advance warnings
                    </li>
                  </ul>
                </div>
                
                {/* Premium Plan */}
                <div 
                  className={`bg-slate-800 rounded-xl p-6 border-2 cursor-pointer transition-all relative ${
                    selectedPlan === 'premium' 
                      ? 'border-emerald-500' 
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                  onClick={() => setSelectedPlan('premium')}
                >
                  <div className="absolute -top-3 right-4 bg-emerald-500 px-3 py-1 rounded-full text-xs font-semibold">
                    RECOMMENDED
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-2">Premium</h3>
                  <p className="text-3xl font-bold text-emerald-400 mb-4">
                    $49<span className="text-lg text-slate-400">/month</span>
                  </p>
                  <ul className="space-y-2 text-slate-300">
                    <li className="flex items-center gap-2">
                      <Check className="text-emerald-400" size={16} />
                      Everything in Essential, plus:
                    </li>
                    <li className="flex items-center gap-2">
                      <Check className="text-emerald-400" size={16} />
                      SMS alerts for urgent warnings
                    </li>
                    <li className="flex items-center gap-2">
                      <Check className="text-emerald-400" size={16} />
                      Unlimited portfolio monitoring
                    </li>
                    <li className="flex items-center gap-2">
                      <Check className="text-emerald-400" size={16} />
                      Personalized exit strategies
                    </li>
                  </ul>
                </div>
              </div>
              
              <div className="flex justify-between">
                <button
                  onClick={() => setCurrentStep(1)}
                  className="text-slate-400 hover:text-white transition-colors"
                >
                  ← Back
                </button>
                <button
                  onClick={() => {
                    handlePlanSelection(selectedPlan)
                    setCurrentStep(3)
                  }}
                  className="bg-emerald-500 hover:bg-emerald-600 px-8 py-3 rounded-lg font-semibold text-white transition-all flex items-center gap-2"
                >
                  Continue with {selectedPlan === 'premium' ? 'Premium' : 'Essential'} <ArrowRight size={20} />
                </button>
              </div>
            </div>
          )}
          
          {/* Step 3: Quick Start */}
          {currentStep === 3 && (
            <div className="bg-slate-800 rounded-lg p-8 border border-slate-700 text-center">
              <Upload className="text-emerald-400 mx-auto mb-6" size={64} />
              <h2 className="text-3xl font-bold text-white mb-4">
                You're All Set!
              </h2>
              <p className="text-xl text-slate-300 mb-8">
                Your {selectedPlan === 'premium' ? 'Premium' : 'Essential'} plan is active.
                <br />Let's upload your portfolio and start monitoring.
              </p>
              <div className="bg-emerald-900/20 border border-emerald-700 rounded-lg p-4 mb-8 max-w-md mx-auto">
                <p className="text-emerald-400 font-medium">🎉 Special Launch Offer</p>
                <p className="text-slate-300 text-sm mt-1">
                  Your first alert within 2 weeks or your money back!
                </p>
              </div>
              <button
                onClick={handleComplete}
                className="bg-emerald-500 hover:bg-emerald-600 px-8 py-3 rounded-lg font-semibold text-white transition-all flex items-center gap-2 mx-auto"
              >
                Upload Portfolio Now <ArrowRight size={20} />
              </button>
              <p className="text-slate-400 text-sm mt-4">
                You can also upload your portfolio later from the dashboard
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}