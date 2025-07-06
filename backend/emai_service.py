# backend/email_service.py
import os
from flask_mail import Mail, Message
from datetime import datetime

def init_mail(app):
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.environ.get('ALERTS_EMAIL')
    app.config['MAIL_PASSWORD'] = os.environ.get('ALERTS_PASSWORD')  # Use App Password
    return Mail(app)

def send_alert_email(mail, user_email, alerts, portfolio_summary):
    """Send beautifully formatted alert email"""
    
    # Create subject line that catches attention
    critical_count = len([a for a in alerts if a['severity'] == 'CRITICAL'])
    if critical_count > 0:
        subject = f"🚨 URGENT: {critical_count} stocks need immediate attention"
    else:
        subject = f"⚠️ Portfolio Alert: {len(alerts)} positions showing high volatility"
    
    msg = Message(
        subject,
        sender=('Portfolio Watchdog', app.config['MAIL_USERNAME']),
        recipients=[user_email]
    )
    
    # HTML email for better formatting
    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #d32f2f;">Portfolio Volatility Alert</h2>
        <p style="color: #666; font-size: 16px;">
            Your portfolio value: <strong>${portfolio_summary['total_value']:,.2f}</strong>
        </p>
        
        <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 5px;">
            <strong>Action Required:</strong> The following positions are experiencing unusual volatility
        </div>
    """
    
    for alert in alerts:
        severity_color = '#d32f2f' if alert['severity'] == 'CRITICAL' else '#ff9800'
        html_body += f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 5px;">
            <h3 style="color: {severity_color}; margin-top: 0;">
                {alert['symbol']} - {alert['severity']} Alert
            </h3>
            <p style="color: #333; white-space: pre-line;">{alert['message']}</p>
        </div>
        """
    
    html_body += f"""
        <div style="margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 5px;">
            <p style="margin: 5px 0;"><strong>What should you do?</strong></p>
            <ol style="color: #666;">
                <li>Review each alert carefully</li>
                <li>Consider the recommended actions</li>
                <li>Log into your broker if you decide to act</li>
                <li>Remember: These are warnings, not guarantees</li>
            </ol>
        </div>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
        
        <p style="color: #999; font-size: 12px;">
            You're receiving this because you subscribed to Portfolio Watchdog alerts.
            <br>Next check scheduled for: {get_next_check_time()}
        </p>
    </div>
    """
    
    msg.html = html_body
    mail.send(msg)
    
    # Log the alert
    log_alert_sent(user_email, alerts)