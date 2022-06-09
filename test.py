import time
import sys

for i in range(100, 0, -1):
	print(i, end="\r", flush=True)
	time.sleep(0.2)
	sys.stdout.write("\x1b[2K")
print("\n")

