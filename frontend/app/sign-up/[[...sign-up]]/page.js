import { SignUp } from '@clerk/nextjs'
import { Shield } from 'lucide-react'

export default function SignUpPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield className="text-emerald-400" size={40} />
            <h1 className="text-3xl font-bold text-white">Portfolio Watchdog</h1>
          </div>
          <p className="text-slate-300">Create your account and start protecting your wealth.</p>
        </div>
        
        {/* Clerk SignUp Component */}
        <SignUp 
          routing="path"
          path="/sign-up"
          redirectUrl="/app/onboarding"
          signInUrl="/sign-in"
        />
        
        {/* Value Props */}
        <div className="mt-8 space-y-3">
          <div className="flex items-center gap-3 text-sm text-slate-300">
            <span className="text-emerald-400">✓</span>
            <span>Free for first month, then $29/month</span>
          </div>
          <div className="flex items-center gap-3 text-sm text-slate-300">
            <span className="text-emerald-400">✓</span>
            <span>Cancel anytime, no questions asked</span>
          </div>
          <div className="flex items-center gap-3 text-sm text-slate-300">
            <span className="text-emerald-400">✓</span>
            <span>Average user saves $47K from alerts</span>
          </div>
        </div>
      </div>
    </div>
  )
}