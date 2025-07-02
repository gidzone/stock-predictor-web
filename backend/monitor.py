#!/usr/bin/env python3
import time
import requests
import json
from datetime import datetime

API_URL = "http://localhost:5040/api"

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