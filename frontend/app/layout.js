import './globals.css'

export const metadata = {
  title: 'AI Stock Predictor',
  description: 'Machine Learning-powered stock price predictions',
  keywords: 'stock prediction, machine learning, AI, finance, trading',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  )
}