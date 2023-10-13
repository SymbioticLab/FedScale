# Submit job to the remote cluster

import datetime
import os
import pickle
import random
import shlex
import subprocess
import sys
import time
import json
from typing import Dict
import yaml
import socket

from kubernetes import client, config, utils
from yaml_generator import generate_aggr_template, generate_exec_template


def flatten(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten(subdict).items()
                out.update({key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def process_cmd(yaml_file, local=False):

    yaml_conf = load_yaml_conf(yaml_file)

    if 'use_container' in yaml_conf:
        if yaml_conf['use_container'] == "docker":
            use_container = "docker"
            ports = yaml_conf['ports']
        elif yaml_conf['use_container'] == "k8s":
            submit_to_k8s(yaml_conf)
            return
        else:
            print(f'Error: unknown use_container:{yaml_conf["use_container"]}, the supported options are ["docker", "k8s"].')
            exit(1)
    else:
        use_container = "default"



    ps_ip = yaml_conf['ps_ip']
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    executor_configs = "=".join(yaml_conf['worker_ips']) if 'worker_ips' in yaml_conf else ''
    if 'worker_ips' in yaml_conf:
        for ip_gpu in yaml_conf['worker_ips']:
            ip, gpu_list = ip_gpu.strip().split(':')
            worker_ips.append(ip)
            total_gpus.append(eval(gpu_list))

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    job_name = 'fedscale_job'
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp': time_stamp,
                'ps_ip': ps_ip,
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "

    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name} {job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = os.path.join(
                job_conf[conf_name], 'log', job_name, time_stamp)

    total_gpu_processes = sum([sum(x) for x in total_gpus])

    # error checking
    if use_container == "docker" and total_gpu_processes + 1 != len(ports):
        print(f'Error: there are {total_gpu_processes + 1} processes but {len(ports)} ports mapped, please check your config file')
        exit(1)

    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    if use_container == "docker":
        # store ip, port of each container
        ctnr_dict = dict()
        ps_name = f"fedscale-aggr-{time_stamp}"
        ctnr_dict[ps_name] = {
            "type": "aggregator",
            "ip": ps_ip,
            "port": ports[0]
        }
        print(f"Starting aggregator container {ps_name} on {ps_ip}...")
        ps_cmd = f" docker run -i --name {ps_name} --network {yaml_conf['container_network']} -p {ports[0]}:30000 --mount type=bind,source={yaml_conf['data_path']},target=/FedScale/benchmark fedscale/fedscale-aggr"
    else:
        print(f"Starting aggregator on {ps_ip}...")
        ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} "

    with open(f"{job_name}_logging", 'wb') as fout:
        pass

    with open(f"{job_name}_logging", 'a') as fout:
        if local:
            local_process = subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
            local_pid = local_process.pid
            print(f'Aggregator local PID {local_pid}. Run kill -9 {local_pid} to kill the job.')
        else:
            subprocess.Popen(f'ssh {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"',
                             shell=True, stdout=fout, stderr=fout)

    time.sleep(10)
    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)

        if use_container == "default":
            print(f"Starting workers on {worker} ...")

        for cuda_id in range(len(gpu)):
            for _ in range(gpu[cuda_id]):
                if use_container == "docker":
                    exec_name = f"fedscale-exec{rank_id}-{time_stamp}"
                    print(f'Starting executor container {exec_name} on {worker}')
                    ctnr_dict[exec_name] = {
                        "type": "executor",
                        "ip": worker,
                        "port": ports[rank_id],
                        "rank_id": rank_id,
                        "cuda_id": cuda_id
                    }

                    worker_cmd = f" docker run -i --name fedscale-exec{rank_id}-{time_stamp} --network {yaml_conf['container_network']} -p {ports[rank_id]}:32000 --mount type=bind,source={yaml_conf['data_path']},target=/FedScale/benchmark fedscale/fedscale-exec"
                else:
                    worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={rank_id} --num_executors={total_gpu_processes} --cuda_device=cuda:{cuda_id} "
                rank_id += 1

                with open(f"{job_name}_logging", 'a') as fout:
                    time.sleep(2)
                    if local:
                        subprocess.Popen(f'{worker_cmd}',
                                         shell=True, stdout=fout, stderr=fout)
                    else:
                        subprocess.Popen(f'ssh {submit_user}{worker} "{setup_cmd} {worker_cmd}"',
                                         shell=True, stdout=fout, stderr=fout)

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = os.path.join(current_path, job_name)
    with open(job_name, 'wb') as fout:
        if use_container == "docker":
            job_meta = {'user': submit_user, 'vms': running_vms, 'container_dict': ctnr_dict, 'use_container': use_container}
        else:
            job_meta = {'user': submit_user, 'vms': running_vms, 'use_container': use_container}
        pickle.dump(job_meta, fout)

    # =========== Container: initialize containers ============
    if use_container == "docker":
        # init aggregator
        send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        start_time = time.time()
        while time.time() - start_time <= 10:
            # avoid busy waiting
            time.sleep(0.1)
            try:
                send_socket.connect((ctnr_dict[ps_name]["ip"], ctnr_dict[ps_name]["port"]))
            except socket.error:
                continue
            msg = {}
            msg["type"] = "aggr_init"
            msg['data'] = job_conf.copy()
            msg['data']['this_rank'] = 0
            msg['data']['num_executors'] = total_gpu_processes
            msg['data']['executor_configs'] = executor_configs
            msg = json.dumps(msg)
            send_socket.sendall(msg.encode('utf-8'))
            send_socket.close()
            break
        time.sleep(10)
        # get the assigned ip of aggregator
        docker_cmd = f"docker network inspect {yaml_conf['container_network']}"
        process = subprocess.Popen(f'ssh {submit_user}{ps_ip} "{docker_cmd}"',
                                    shell=True, stdout=subprocess.PIPE)
        output = json.loads(process.communicate()[0].decode("utf-8"))
        ps_ip_cntr = None
        for _, value in output[0]['Containers'].items():
            if value['Name'] == ps_name:
                ps_ip_cntr = value['IPv4Address'].split("/")[0]
        if ps_ip_cntr == None:
            print(f"Error: no aggregator container with name {ps_name} found in network {yaml_conf['container_network']}, aborting")
            # terminiate?
            exit(1)
        # init all executors
        for name, meta_dict in ctnr_dict.items():
            if name == ps_name:
                continue
            send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            start_time = time.time()
            while time.time() - start_time <= 10:
                # avoid busy waiting
                time.sleep(0.1)
                try:
                    send_socket.connect((meta_dict["ip"], meta_dict["port"]))
                except socket.error:
                    continue
                msg = {}
                msg["type"] = "exec_init"
                msg['data'] = job_conf.copy()
                msg['data']['this_rank'] = meta_dict['rank_id']
                msg['data']['num_executors'] = total_gpu_processes
                msg['data']['cuda_device'] = f"cuda:{meta_dict['cuda_id']}"
                msg['data']['ps_ip'] = ps_ip_cntr
                msg = json.dumps(msg)
                send_socket.sendall(msg.encode('utf-8'))
                send_socket.close()
                break


    print(f"Submitted job, please check your logs {job_conf['log_path']}/logs/{job_conf['job_name']}/{time_stamp} for status")
    print(f"if you cannot find logs directory on the path, you need to check ""{job_name}_logging"" log file under FEDSCALE root directory.")


def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    if job_meta['use_container'] == "docker":
        for name, meta_dict in job_meta['container_dict'].items():
            print(f"Shutting down container {name} on {meta_dict['ip']}")
            with open(f"{job_name}_logging", 'a') as fout:
                subprocess.Popen(f'ssh {job_meta["user"]}{meta_dict["ip"]} "docker rm --force {name}"',
                                shell=True, stdout=fout, stderr=fout)
    elif job_meta['use_container'] == "k8s":
        # for now, assume we run in k8s admin mode, placeholder for client job submission in the future
        config.load_kube_config()
        core_api = client.CoreV1Api()
        for name, meta_dict in job_meta['k8s_dict'].items():
            if os.path.exists(meta_dict["yaml_path"]):
                os.remove(meta_dict["yaml_path"])

            print(f"Shutting down container {name}...")
            core_api.delete_namespaced_pod(name, namespace="fedscale")

    else:
        for vm_ip in job_meta['vms']:
            print(f"Shutting down job on {vm_ip}")
            with open(f"{job_name}_logging", 'a') as fout:
                subprocess.Popen(f'ssh {job_meta["user"]}{vm_ip} "python {current_path}/shutdown.py {job_name}"',
                                shell=True, stdout=fout, stderr=fout)

def submit_to_k8s(yaml_conf):
    # TODO: switch to real deployment configs, pod configs are only for testing usage right now
    # TODO: check if k8s is online?
    # for now, assume we run in k8s admin mode, placeholder for client job submission in the future
    config.load_kube_config()
    k8s_client = client.ApiClient()
    core_api = client.CoreV1Api()

    # check fedscale namespace
    field_selector = "metadata.name=fedscale"
    namespaces = core_api.list_namespace(field_selector=field_selector).items
    if len(namespaces) == 0:
        print("Namespace 'fedscale' not found, creating new namespace...")
        namespace_config = client.V1Namespace(
            metadata=client.V1ObjectMeta(name="fedscale"))
        core_api.create_namespace(namespace_config)

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp': time_stamp,
                }
    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)
    job_name = job_conf["job_name"]

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    k8s_dict = dict()

    # =========== Submit aggregator to k8s ============
    # generate aggregator yaml
    if yaml_conf["num_aggregators"] != 1:
        print("Error: currently only support single aggregator!")
        exit(1)

    aggr_name = f'fedscale-aggr-{time_stamp}'.replace("_", "-")
    print(f"Generating yaml for aggregator container {aggr_name}...")
    aggr_config = {
        "data_path": yaml_conf['data_path'],
        "pod_name": aggr_name
    }
    aggr_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{aggr_name}.yaml')
    generate_aggr_template(aggr_config, aggr_yaml_path)
    k8s_dict[aggr_name] = {
        "type": "aggregator",
        "rank_id": 0,
        "yaml_path": aggr_yaml_path
    }

    print(f"Submitting aggregator container {aggr_name} to k8s...")

    # TODO: logging?
    utils.create_from_yaml(k8s_client, aggr_yaml_path, namespace="fedscale")


    time.sleep(5)
    # =========== Submit executors to k8s ============
    for rank_id in range(1, yaml_conf["num_executors"]+1):
        exec_name = f"fedscale-exec{rank_id}-{time_stamp}".replace("_", "-")
        # generate executor yaml
        exec_config = {
            "data_path": yaml_conf["data_path"],
            "pod_name": exec_name,
            "use_cuda": job_conf["use_cuda"]
        }

        exec_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{exec_name}.yaml')
        generate_exec_template(exec_config, exec_yaml_path)
        k8s_dict[exec_name] = {
            "type": "executor",
            "rank_id": rank_id,
            "yaml_path": exec_yaml_path
        }
        print(f'Submitting executor container {exec_name} to k8s...')
        # TODO: logging?
        utils.create_from_yaml(k8s_client, exec_yaml_path, namespace="fedscale")
    # =========== Wait for containers to be ready ===========
    executor_configs = []
    for name, meta_dict in k8s_dict.items():
        if meta_dict['type'] == 'aggregator':
            print(f'Waiting aggregator container {name} to be ready...')
            aggr_ip = -1
            start_time = time.time()
            # a cold start would take 5-6min, depends on network status
            while time.time() - start_time < 600:
                resp = core_api.read_namespaced_pod(name, namespace="fedscale")
                if resp.status.phase == 'Pending':
                    continue
                if resp.status.container_statuses[0].ready:
                    aggr_ip = resp.status.pod_ip
                    break
                time.sleep(1)
            if aggr_ip == -1:
                print(f"Error: aggregator {name} not ready after maximum waiting time allowed, aborting...")
                exit(1)
            meta_dict["ip"] = aggr_ip
        elif meta_dict['type'] == 'executor':
            print(f'Waiting executor container {name} to be ready...')
            exec_ip = -1
            start_time = time.time()
            while time.time() - start_time < 600:
                resp = core_api.read_namespaced_pod(name, namespace="fedscale")
                if resp.status.phase == 'Pending':
                    continue
                if resp.status.container_statuses[0].ready:
                    exec_ip = resp.status.pod_ip
                    break
                time.sleep(1)
            if exec_ip == -1:
                print(f"Error: executor {name} not ready after maximum waiting time allowed, aborting...")
                exit(1)
            # update meta data
            meta_dict["ip"] = exec_ip
            # for now, assume only 1 gpu process for one executor container
            executor_configs.append(f'{exec_ip}:[1]')
        else:
            print(f"Error: unrecognized type {meta_dict['type']}!")
            exit(1)


    # TODO: make executors init multi-threaded to boost performance
    for name, meta_dict in k8s_dict.items():
        if meta_dict["type"] == "aggregator":
            # TODO: refactor the code so that docker/k8s version invoke the same init function
            print(f'Initializing aggregator container {aggr_name}...')
            send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            start_time = time.time()
            while time.time() - start_time <= 10:
                # avoid busy waiting
                time.sleep(0.1)
                try:
                    send_socket.connect((meta_dict["ip"], 30000))
                except socket.error:
                    continue
                msg = {}
                msg["type"] = "aggr_init"
                msg['data'] = job_conf.copy()
                msg['data']['this_rank'] = 0
                msg['data']['num_executors'] = yaml_conf["num_executors"]
                msg['data']['executor_configs'] = "=".join(executor_configs)
                msg = json.dumps(msg)
                send_socket.sendall(msg.encode('utf-8'))
                send_socket.close()
                break
            time.sleep(10)
        elif meta_dict["type"] == "executor":
            print(f'Initializing executor container {name}...')
            send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            start_time = time.time()
            while time.time() - start_time <= 10:
                # avoid busy waiting
                time.sleep(0.1)
                try:
                    send_socket.connect((meta_dict["ip"], 32000))
                except socket.error:
                    continue
                msg = {}
                msg["type"] = "exec_init"
                msg['data'] = job_conf.copy()
                msg['data']['this_rank'] = meta_dict['rank_id']
                msg['data']['num_executors'] = yaml_conf["num_executors"]
                # TODO: support CUDA device
                # assume single aggregator for now
                msg['data']['ps_ip'] = aggr_ip
                if job_conf["use_cuda"]:
                    # TODO: what if there are multiple GPUs?
                    msg['data']['cuda_device'] = f"cuda:0"
                msg = json.dumps(msg)
                send_socket.sendall(msg.encode('utf-8'))
                send_socket.close()
                break
        else:
            print(f"Error: unrecognized type {meta_dict['type']}!")
            exit(1)

    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), job_name)
    with open(current_path, "wb") as fout:
        meta_data = {"user": submit_user, "k8s_dict": k8s_dict, "use_container": "k8s"}
        pickle.dump(meta_data, fout)

def check_log(job_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)
    if not os.path.isfile(job_meta_path):
        print(f"Error: fail to terminate {job_name}, as it does not exist")
        exit(1)

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    if job_meta['use_container'] == 'k8s':
        for name, meta_dict in job_meta['k8s_dict'].items():
            if meta_dict['type'] != 'aggregator':
                continue
            # don't use k8s python api here, need interactive log
            print(f"%%%%%%%%%% Start of {name} log %%%%%%%%%%")
            cmd = f"kubectl logs {name} --follow -n fedscale"
            proc = subprocess.Popen(shlex.split(cmd))
            proc.communicate()
    else:
        print("Error: only support checking job logs running in k8s mode!")
        exit(1)


print_help: bool = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'submit' or sys.argv[1] == 'start':
        process_cmd(sys.argv[2], False if sys.argv[1] == 'submit' else True)
    elif sys.argv[1] == 'stop':
        terminate(sys.argv[2])
    elif sys.argv[1] == 'logs':
        check_log(sys.argv[2])
    else:
        print_help = True
else:
    print_help = True

if print_help:
    # TODO: add support for reporting k8s job status
    print("\033[0;32mUsage:\033[0;0m\n")
    print("submit $PATH_TO_CONF_YML     # Submit a job")
    print("logs $JOB_NAME               # Check the aggregator log of a job")
    print("stop $JOB_NAME               # Terminate a job")
    print()
