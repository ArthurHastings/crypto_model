import time
from datetime import datetime, timedelta
import pytz

def wait_until_1631():
    tz = pytz.timezone("Europe/Bucharest")
    now = datetime.now(tz)
    target_time = now.replace(hour=16, minute=31, second=0, microsecond=0)

    if now >= target_time:
        print(f"Current time {now.strftime('%H:%M:%S')} is already past 16:31, proceeding immediately.")
        return

    wait_seconds = (target_time - now).total_seconds()
    print(f"Waiting {wait_seconds:.0f} seconds until {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    time.sleep(wait_seconds)

if __name__ == "__main__":
    wait_until_1631()
    print("Reached 16:31 PM Bucharest time. Proceeding...")
