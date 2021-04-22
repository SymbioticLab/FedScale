import os, sys

base = int(sys.argv[1])
jobs = int(sys.argv[2])
for i in range(jobs):
	print(os.system("bkill -r " + str(i+base)))
