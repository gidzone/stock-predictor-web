import { Inter } from 'next/font/google'
import { ClerkProvider } from '@clerk/nextjs'
import { dark } from '@clerk/themes'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Portfolio Watchdog - Protect Your Wealth',
  description: 'Get alerted 48 hours before major price drops. 70% accurate.',
}

export default function RootLayout({ children }) {
  return (
    <ClerkProvider
      appearance={{
        baseTheme: dark,
        variables: {
          colorPrimary: '#10b981', // emerald-500
          colorBackground: '#1e293b', // slate-800
          colorInputBackground: '#334155', // slate-700
          colorText: '#f1f5f9', // slate-100
        },
        elements: {
          formButtonPrimary: 
            'bg-emerald-500 hover:bg-emerald-600 text-white',
          footerActionLink: 
            'text-emerald-400 hover:text-emerald-300',
          identityPreviewText:
            'text-slate-300',
          formFieldInput:
            'bg-slate-700 border-slate-600 text-white',
          dividerLine:
            'bg-slate-700',
          socialButtonsBlockButton:
            'bg-slate-700 border-slate-600 hover:bg-slate-600 text-white',
          socialButtonsBlockButtonText:
            'text-white font-medium',
        }
      }}
    >
      <html lang="en">
        <body className={inter.className}>{children}</body>
      </html>
    </ClerkProvider>
  )
}