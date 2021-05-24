import os, sys

assert(len(sys.argv) == 1)
os.system('bjobs > jobinfo')
tries = 3

with open('jobinfo', 'r') as fin:
    line = fin.readlines()

for ii in range(tries):
    for job in line[1:]:
        os.system('bkill -r ' + str(job.split()[0]))

os.system('rm jobinfo')
