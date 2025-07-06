# backend/monitor.py
#!/usr/bin/env python3
import os
import sys
import json
import logging
import requests
from datetime import datetime

API_URL = "http://localhost:5040/api"

# Set up logging
logging.basicConfig(
    filename='monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def check_alerts():
    try:
        response = requests.get(f"{API_URL}/check-volatility-alerts")
        data = response.json()
        
        if data.get('alerts'):
            print(f"\n🚨 ALERTS FOUND at {datetime.now()}")
            for alert in data['alerts']:
                print(f"\n{alert['symbol']} - {alert['severity']}")
                print(alert['message'])
                print("-" * 50)
        else:
            print(f"✅ No alerts at {datetime.now()}")
            
    except Exception as e:
        print(f"Error checking alerts: {e}")

if __name__ == "__main__":
    print("Starting portfolio monitor...")
    print("Checking every 30 minutes...")
    
    while True:
        check_alerts()
        time.sleep(30 * 60)  # Check every 30 minutes
        
        
def check_user_portfolio(user_data):
    """Check one user's portfolio for alerts"""
    try:
        # Check volatility
        response = requests.get(f"{API_URL}/check-volatility-alerts")
        data = response.json()
        
        if data.get('alerts') and user_data.get('email'):
            logging.info(f"Found {len(data['alerts'])} alerts for {user_data['email']}")
            
            # Send email
            send_alert_email(
                user_data['email'], 
                data['alerts'],
                data['portfolio_summary']
            )
            
            return len(data['alerts'])
        
        return 0
        
    except Exception as e:
        logging.error(f"Error checking portfolio: {str(e)}")
        return 0

def run_monitoring():
    """Main monitoring function"""
    logging.info("Starting portfolio monitoring run")
    
    # For MVP, just check the one uploaded portfolio
    if os.path.exists('alert_preferences.json'):
        with open('alert_preferences.json', 'r') as f:
            prefs = json.load(f)
            
        if prefs.get('enabled') and prefs.get('email'):
            alerts_sent = check_user_portfolio(prefs)
            logging.info(f"Monitoring complete. Sent {alerts_sent} alerts")
    else:
        logging.warning("No user preferences found")

if __name__ == '__main__':
    # Only run during market hours (9 AM - 4 PM ET, Mon-Fri)
    now = datetime.now()
    if now.weekday() < 5 and 9 <= now.hour < 16:
        run_monitoring()
    else:
        logging.info("Skipping - outside market hours")