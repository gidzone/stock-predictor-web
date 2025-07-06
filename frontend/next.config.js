/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',  // Important for Heroku
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5040/api'
  }
}