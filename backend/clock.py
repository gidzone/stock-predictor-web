# backend/clock.py (Heroku scheduler)
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess

sched = BlockingScheduler()

@sched.scheduled_job('cron', day_of_week='mon-fri', hour='9,13,15')
def scheduled_job():
    """Run monitoring at 9am, 1pm, and 3pm ET on weekdays"""
    subprocess.run(['python', 'monitor.py'])

sched.start()