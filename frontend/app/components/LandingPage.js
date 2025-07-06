// frontend/app/components/LandingPage.js
export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Your Portfolio Watchdog
          </h1>
          <p className="text-2xl text-gray-600 mb-8">
            We monitor your stocks 24/7 and alert you <em>only</em> when action is needed
          </p>
          
          <div className="bg-white rounded-lg shadow-xl p-8 mb-8">
            <h2 className="text-2xl font-semibold mb-4">
              Stop Losing Money to Market Volatility
            </h2>
            <p className="text-lg text-gray-700 mb-6">
              The average investor loses <strong>$3,000+</strong> per year by not 
              reacting to volatility spikes. Our AI monitors for dangerous patterns 
              and alerts you <em>before</em> major drops.
            </p>
            
            <div className="text-4xl font-bold text-green-600 mb-2">
              $9.99/month
            </div>
            <p className="text-gray-600 mb-6">
              Less than your Netflix subscription, protects your entire portfolio
            </p>
            
            <button className="bg-blue-600 text-white px-8 py-4 rounded-lg text-xl hover:bg-blue-700">
              Start 7-Day Free Trial
            </button>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 bg-white">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">
            How It Works (For Busy People)
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">1️⃣</span>
              </div>
              <h3 className="font-semibold mb-2">Upload Your Portfolio</h3>
              <p className="text-gray-600">
                Simple CSV upload - takes 30 seconds
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-blue-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">2️⃣</span>
              </div>
              <h3 className="font-semibold mb-2">We Monitor Daily</h3>
              <p className="text-gray-600">
                AI checks for volatility spikes every 4 hours
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-blue-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">3️⃣</span>
              </div>
              <h3 className="font-semibold mb-2">Get Alerts That Matter</h3>
              <p className="text-gray-600">
                Clear emails only when you need to act
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Social Proof */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">
            What Our Users Say
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-lg shadow">
              <p className="text-gray-700 mb-4">
                "Got an alert about Tesla volatility spike. Sold 30% of my position 
                at $245. Stock dropped to $198 two days later. Saved me $4,700!"
              </p>
              <p className="font-semibold">- Sarah M., Software Engineer</p>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow">
              <p className="text-gray-700 mb-4">
                "I check my portfolio maybe once a month. This service is perfect - 
                only bothers me when something important happens."
              </p>
              <p className="font-semibold">- Michael T., Busy Dad</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}