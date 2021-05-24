import sys, os, time, datetime, random

ps_port = random.randint(1000, 8000)
manager_port = random.randint(1000, 8000)


paramsCmd = ' --model=shufflenet_v2_x2_0 --epochs=20000 --dump_epoch=1000 --learning_rate=0.05 --batch_size=20 ' + \
            ' --ps_port='+str(ps_port)+' --manager_port='+str(manager_port) + ' '

os.system("bhosts > vms")
os.system("rm *.o")
os.system("rm *.e")

avaiVms = {}
quotalist = {}

with open('quotas', 'r') as fin:
    for v in fin.readlines():
        items = v.strip().split()
        quotalist[items[0]] = int(items[1])

threadQuota = 1

with open('vms', 'r') as fin:
    lines = fin.readlines()
    for line in lines:
        if 'gpu-cn0' in line:
            items = line.strip().split()

            status = items[1]
            threadsGpu = int(items[5])
            vmName = items[0]
            #print(vmName,'#', quotalist[vmName])
            maxQuota = quotalist[vmName] if vmName in quotalist else 999

            if status == "ok" and (40-threadsGpu) >= threadQuota and maxQuota >= threadQuota:
                avaiVms[vmName] = min(40 - threadsGpu, maxQuota)

# remove all log files, and scripts first
files = [f for f in os.listdir('.') if os.path.isfile(f)]

print(avaiVms)
for file in files:
    if 'learner' in file or 'server' in file:
        os.remove(file)
        
# get the number of workers first
numOfWorkers = int(sys.argv[1])
learner = ' --learners=1'

for w in range(2, numOfWorkers+1):
    learner = learner + '-' + str(w)

# load template
with open('template.lsf', 'r') as fin:
    template = ''.join(fin.readlines())

# load template
with open('template.lsf', 'r') as fin:
    template_server = ''.join(fin.readlines())

_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
timeStamp = str(_time_stamp) + '_'
jobPrefix = 'learner_' + timeStamp

# get the join of parameters
params = ' '.join(sys.argv[2:]) + learner + ' --time_stamp=' + _time_stamp + ' '

rawCmd = '\npython ~/FLPerf-Private/training/executor.py' + paramsCmd

assignedVMs = []
# assert(len(availGPUs) > numOfWorkers)

# generate new scripts, assign each worker to different vms
for w in range(1, numOfWorkers + 1):
    
    rankId = ' --this_rank=' + str(w)
    fileName = jobPrefix+str(w)
    jobName = 'learner' + str(w)

    _vm = sorted(avaiVms, key=avaiVms.get, reverse=True)[0]
    print('assign ...{} to {}'.format(str(w), _vm))
    assignedVMs.append(_vm)

    avaiVms[_vm] -= threadQuota
    if avaiVms[_vm] < threadQuota:
        del avaiVms[_vm]

    assignVm = '\n#BSUB -m "{}"\n'.format(_vm)
    runCmd = template + assignVm + '\n#BSUB -J ' + jobName + '\n#BSUB -e ' + fileName + '.e\n'  + '#BSUB -o '+ fileName + '.o\n'+ rawCmd + params + rankId

    with open('learner' + str(w) + '.lsf', 'w') as fout:
        fout.writelines(runCmd)

# deal with ps
rawCmdPs = '\npython ~/FLPerf-Private/training/aggregator.py ' + paramsCmd + ' --this_rank=0 ' + params

with open('server.lsf', 'w') as fout:
    master_node = sorted(avaiVms, key=avaiVms.get, reverse=True)[0]
    scriptPS = template_server + '\n#BSUB -J server\n#BSUB -e server_{}'.format(timeStamp) + '.e\n#BSUB -o server_{}'.format(timeStamp) + '.o\n' + '#BSUB -m "'+master_node+'"\n\n' + rawCmdPs
    #scriptPS = template_server + '\n#BSUB -J server\n#BSUB -e server{}'.format(timeStamp) + '.e\n#BSUB -o server{}'.format(timeStamp) + '.o\n' + '\n' + rawCmdPs
    fout.writelines(scriptPS)

# execute ps
os.system('bsub < server.lsf')

time.sleep(3)
os.system('rm vms')

vmSets = set()
for w in range(1, numOfWorkers + 1):
    # avoid gpu contention on the same machine
    if assignedVMs[w-1] in vmSets:
      time.sleep(1)
    vmSets.add(assignedVMs[w-1])
    os.system('bsub < learner' + str(w) + '.lsf')
