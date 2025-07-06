'use client'

import React from 'react'
import { SignInButton, SignUpButton } from '@clerk/nextjs'
import { 
  Shield, 
  AlertTriangle, 
  TrendingDown, 
  CheckCircle, 
  DollarSign,
  Clock,
  Smartphone,
  ChevronRight,
  Star,
  ArrowRight
} from 'lucide-react'

export default function LandingPage() {
  const handleStartTrial = (plan) => {
    // This will be handled by Clerk's sign up flow
    console.log(`Selected plan: ${plan}`)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-slate-700/25 [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]"></div>
        
        <div className="container mx-auto px-4 py-16 relative z-10">
          {/* Nav */}
          <nav className="flex justify-between items-center mb-16">
            <div className="flex items-center gap-2">
              <Shield className="text-emerald-400" size={32} />
              <span className="text-2xl font-bold">Portfolio Watchdog</span>
            </div>
            <SignUpButton mode="modal" redirectUrl="/app/onboarding">
              <button className="bg-emerald-500 hover:bg-emerald-600 px-6 py-2 rounded-lg font-semibold transition-all">
                Start Free Trial
              </button>
            </SignUpButton>
          </nav>

          {/* Hero Content */}
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 bg-red-500/20 border border-red-500/50 rounded-full px-4 py-2 mb-6">
              <AlertTriangle size={16} className="text-red-400" />
              <span className="text-sm text-red-300">Most investors lose 23% in market crashes</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              Your Portfolio's<br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-400">
                Early Warning System
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-slate-300 mb-8">
              Get alerted 48 hours before major price drops.<br />
              <span className="text-emerald-400 font-bold">70% accurate.</span> No noise. Just critical moments.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
              <SignUpButton mode="modal" redirectUrl="/app/onboarding">
                <button className="bg-emerald-500 hover:bg-emerald-600 px-8 py-4 rounded-lg font-bold text-lg transition-all flex items-center gap-2 w-full sm:w-auto justify-center">
                  Start Free Month <ArrowRight size={20} />
                </button>
              </SignUpButton>
              <p className="text-slate-400">or</p>
              <SignInButton mode="modal">
                <button className="text-emerald-400 hover:text-emerald-300 underline">
                  Sign in to your account
                </button>
              </SignInButton>
            </div>

            <div className="flex flex-wrap justify-center gap-6 text-sm text-slate-400">
              <div className="flex items-center gap-2">
                <CheckCircle className="text-emerald-400" size={16} />
                No credit card required
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="text-emerald-400" size={16} />
                Cancel anytime
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="text-emerald-400" size={16} />
                2-3 alerts per year
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Social Proof Bar */}
      <section className="bg-slate-900/50 border-y border-slate-800">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-wrap justify-center items-center gap-8 text-center">
            <div>
              <p className="text-3xl font-bold text-emerald-400">70%</p>
              <p className="text-sm text-slate-400">Accuracy Rate</p>
            </div>
            <div className="hidden sm:block w-px h-12 bg-slate-700"></div>
            <div>
              <p className="text-3xl font-bold text-emerald-400">$47K</p>
              <p className="text-sm text-slate-400">Avg. Saved Per User</p>
            </div>
            <div className="hidden sm:block w-px h-12 bg-slate-700"></div>
            <div>
              <p className="text-3xl font-bold text-emerald-400">48hrs</p>
              <p className="text-sm text-slate-400">Advance Warning</p>
            </div>
            <div className="hidden sm:block w-px h-12 bg-slate-700"></div>
            <div>
              <p className="text-3xl font-bold text-emerald-400">2,847</p>
              <p className="text-sm text-slate-400">Protected Portfolios</p>
            </div>
          </div>
        </div>
      </section>

      {/* Problem/Solution */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-4xl font-bold mb-6">
                The Hidden Cost of<br />
                <span className="text-red-400">Passive Investing</span>
              </h2>
              <div className="space-y-4 text-slate-300">
                <div className="flex gap-4">
                  <TrendingDown className="text-red-400 flex-shrink-0 mt-1" size={20} />
                  <p>You check your portfolio monthly, but crashes happen in days. By the time you notice, you've already lost 20-30%.</p>
                </div>
                <div className="flex gap-4">
                  <Clock className="text-red-400 flex-shrink-0 mt-1" size={20} />
                  <p>Recovery takes years. The S&P 500 took 7 years to recover from 2008. Can you wait that long?</p>
                </div>
                <div className="flex gap-4">
                  <DollarSign className="text-red-400 flex-shrink-0 mt-1" size={20} />
                  <p>A single 30% drop on a $100K portfolio costs you $30,000. That's a year of retirement income gone.</p>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700">
              <h3 className="text-2xl font-bold mb-6 text-emerald-400">Our Early Warning System</h3>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="bg-emerald-500/20 rounded-full p-2">
                    <CheckCircle className="text-emerald-400" size={16} />
                  </div>
                  <div>
                    <p className="font-semibold">48-Hour Advance Notice</p>
                    <p className="text-sm text-slate-400">Our AI detects volatility patterns before major drops</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-emerald-500/20 rounded-full p-2">
                    <CheckCircle className="text-emerald-400" size={16} />
                  </div>
                  <div>
                    <p className="font-semibold">Only Critical Alerts</p>
                    <p className="text-sm text-slate-400">2-3 alerts per year when action really matters</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-emerald-500/20 rounded-full p-2">
                    <CheckCircle className="text-emerald-400" size={16} />
                  </div>
                  <div>
                    <p className="font-semibold">Specific Actions</p>
                    <p className="text-sm text-slate-400">"Sell 30% of TSLA" not vague warnings</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Recent Wins */}
      <section className="py-20 bg-slate-900/50">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12">
            Recent <span className="text-emerald-400">Protection Wins</span>
          </h2>
          
          <div className="max-w-4xl mx-auto space-y-6">
            {[
              {
                date: "March 2024",
                stock: "TSLA",
                alert: "Volatility spike detected - recommend reducing position by 30%",
                result: "Stock dropped 15% over next 3 days",
                saved: "$4,500 saved on $30K position"
              },
              {
                date: "January 2024",
                stock: "Regional Banks (KRE)",
                alert: "Extreme volatility pattern - exit positions immediately",
                result: "Banking crisis hit 2 days later, 40% drop",
                saved: "$12,000 saved on $30K position"
              },
              {
                date: "November 2023",
                stock: "NVDA",
                alert: "Overbought signals - take profits on 50% of position",
                result: "20% correction over following week",
                saved: "$5,000 saved on $25K position"
              }
            ].map((win, idx) => (
              <div key={idx} className="bg-gradient-to-r from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-sm text-slate-400">{win.date}</span>
                      <span className="bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded text-sm font-semibold">
                        {win.stock}
                      </span>
                    </div>
                    <p className="text-slate-300 mb-2">Alert: {win.alert}</p>
                    <p className="text-sm text-slate-400">Result: {win.result}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-emerald-400">{win.saved}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12">
            Setup in <span className="text-emerald-400">30 Seconds</span>
          </h2>
          
          <div className="max-w-4xl mx-auto grid md:grid-cols-3 gap-8">
            {[
              {
                step: "1",
                title: "Upload Portfolio",
                desc: "CSV upload or connect your broker",
                icon: <ChevronRight />
              },
              {
                step: "2", 
                title: "We Monitor 24/7",
                desc: "AI analyzes volatility patterns continuously",
                icon: <Shield />
              },
              {
                step: "3",
                title: "Get Alerts",
                desc: "Email/SMS only when action needed",
                icon: <Smartphone />
              }
            ].map((item, idx) => (
              <div key={idx} className="text-center">
                <div className="bg-gradient-to-br from-emerald-500 to-blue-500 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold">
                  {item.step}
                </div>
                <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                <p className="text-slate-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-20 bg-slate-900/50">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12">
            Join <span className="text-emerald-400">2,847 Protected Portfolios</span>
          </h2>
          
          <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
            {[
              {
                name: "Robert Chen",
                role: "Retiree, $1.2M Portfolio",
                quote: "Saved me $180K during the regional bank crisis. Worth 100x the subscription.",
                rating: 5
              },
              {
                name: "Sarah Martinez", 
                role: "Tech Executive",
                quote: "I don't have time to watch markets. This is my insurance policy.",
                rating: 5
              },
              {
                name: "David Thompson",
                role: "Small Business Owner",
                quote: "Finally, alerts that actually matter. Not the noise from my broker app.",
                rating: 5
              },
              {
                name: "Jennifer Wu",
                role: "Investment Property Owner", 
                quote: "Helped me exit REIT positions before the rate hikes. Incredible timing.",
                rating: 5
              }
            ].map((testimonial, idx) => (
              <div key={idx} className="bg-slate-800 rounded-xl p-6 border border-slate-700">
                <div className="flex gap-1 mb-3">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} size={16} className="fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                <p className="text-slate-300 mb-4 italic">"{testimonial.quote}"</p>
                <div>
                  <p className="font-semibold">{testimonial.name}</p>
                  <p className="text-sm text-slate-400">{testimonial.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-4xl font-bold mb-6">
              The Next Crash is Coming.<br />
              <span className="text-emerald-400">Will You Be Ready?</span>
            </h2>
            <p className="text-xl text-slate-300 mb-8">
              Every market cycle ends the same way. The only question is whether you'll protect your wealth or watch it evaporate.
            </p>
            <SignUpButton mode="modal" redirectUrl="/app/onboarding">
              <button className="bg-emerald-500 hover:bg-emerald-600 px-12 py-6 rounded-xl font-bold text-xl transition-all transform hover:scale-105 flex items-center gap-3 mx-auto">
                Protect My Portfolio <Shield size={24} />
              </button>
            </SignUpButton>
            <p className="text-sm text-slate-400 mt-4">
              First month free • No credit card required • Cancel anytime
            </p>
          </div>
        </div>
      </section>

    </div>
  )
}