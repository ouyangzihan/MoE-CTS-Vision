import sys
from datetime import datetime

print(2**100000)



# Get current date and time
now = datetime.now()

# 1. Print formatted time (HH:MM:SS)
print("Current Time:", now.strftime("%H:%M:%S"))

# 2. Print formatted 12-hour time with AM/PM
print("12-Hour Time:", now.strftime("%I:%M:%S %p"))

# 3. Print just the raw time object (includes microseconds)
print("Raw Time Object:", now.time())