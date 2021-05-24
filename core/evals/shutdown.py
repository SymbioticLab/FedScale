import os
import sys
import argparse

job_name = sys.argv[1]

if job_name == 'all':
    os.system("ps -ef | grep python | grep Kuiper > kuiper_running_temp")
else:
    os.system("ps -ef | grep python | grep job_name={} > kuiper_running_temp".format(job_name))
[os.system("kill -9 "+str(l.split()[1])) for l in open("kuiper_running_temp").readlines()]
os.system("rm kuiper_running_temp")
