import { SignIn } from '@clerk/nextjs'
import { Shield } from 'lucide-react'

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield className="text-emerald-400" size={40} />
            <h1 className="text-3xl font-bold text-white">Portfolio Watchdog</h1>
          </div>
          <p className="text-slate-300">Welcome back! Sign in to monitor your portfolio.</p>
        </div>
        
        {/* Clerk SignIn Component */}
        <SignIn 
          routing="path"
          path="/sign-in"
          redirectUrl="/app"
          signUpUrl="/sign-up"
        />
        
        {/* Trust Badges */}
        <div className="mt-8 text-center text-sm text-slate-400">
          <p>🔒 Bank-level security • SOC 2 compliant</p>
          <p className="mt-2">Your financial data is encrypted and secure</p>
        </div>
      </div>
    </div>
  )
}