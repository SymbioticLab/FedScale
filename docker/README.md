# Containerized FedScale Tutorial
This is a tutorial on how to run FedScale jobs under k8s/docker's management service and contribute to FedScale Docker images. It is written mostly for FedScale developers' reference. We assume the readers start with more than one Linux host machines running Ubuntu 16.04 or newer. 

## K8S Deployment (Recommended)
Kubernetes (k8s) is a popular container management framework used in cloud applications. It is integrated with auto-configured network and offers features like load balancing, autoscaling, which is suitable for deploying FedScale framework. 

### K8S Setup
We provide scripts for setting up k8s master and worker nodes from a bare-metal cluster in [this repo](https://github.com/IKACE/k8s_setup). The rest part of k8s deployment tutorial assumes the user has set up a k8s network properly. 

To run GPU jobs, follow [K8S CUDA Plugin Tutorial](https://github.com/IKACE/k8s_setup#kubernetes-cuda-plugin-setup) to setup your k8s cluster with CUDA plugin support. FedScale is also fully integrated with time-slicing GPU feature, follow [ Time-slicing GPU Tutorial](https://github.com/IKACE/k8s_setup#optional-enable-time-slicing-feature) to setup time-slicing GPU feature in your k8s cluster.

### FedScale K8S Demo
Example configs for submitting jobs to k8s cluster are provided under `$FEDSCALE_HOME/benchmark/configs` (e.g. `$FEDSCALE_HOME/benchmark/configs/femnist/conf_k8s.yml`). Suppose we want to run `Femnist` job with 1 aggregator and 2 executor, modify the config file `conf_k8s.yml` as following:

- Specify parameter `use_container: k8s`, this indicates you want to submit job to k8s cluster.

- Prepare dataset and specify `data_path` in your `conf_k8s.yml`. This path will be mounted into the container so that datasets can be reused in future runs. Note that we assume the same `data_path` will be used across all nodes in the cluster, so it is always good to make sure `data_path` exists on every k8s node.

- Specify number of aggregators `num_aggregators: 1`, for now we only support a single aggregator for one job, but we are developing hierarchical aggregators feature that will be added in the future.

- Specify number of executors `num_executors: 2`. 

- Set `use_cuda` flag to `True` if you want to use GPU during the training. If enabled, the executors will be running on CUDA GPUs. Each executor is mapped to a *logical GPU unit*, where the physical GPU could either be an entire single GPU or a time-slicing sharable GPU, depending on your k8s configuration.

- Submit job

	```
	cd $FEDSCALE_HOME/docker
	python3 driver.py submit $FEDSCALE_HOME/benchmark/configs/femnist/conf_k8s.yml
	```

- Check logs: FedScale will generate logs under `data_path` you provided by default. Keep in mind that k8s may load balancing your job to any node on the cluster, so make sure you are checking the `data_path` on the correct node.

- Stop job

	```
	cd $FEDSCALE_HOME/docker
	python3 driver.py stop $YOUR_JOB
	```


## Docker Deployment
Apart from k8s deployment, we also support deploying FedScale onto docker service. 

### Install Docker and its dependencies

The first step is to install Docker on all the available host machines.

- Run `$FEDSCALE_HOME/docker/config/docker_setup.sh` to install Docker and its dependencies. 


	 ```bash
	 bash $FEDSCALE_HOME/docker/config/docker_setup.sh
	 ```
	 
- (Optional) Docker stores its data files under `/var/lib/docker` by default. If you have limited disk space under your root `/` partition, consider reconfiguring Docker's data folder by running the following commands:


	```
	# stop Docker service
	sudo systemctl stop docker
	# make sure Docker service is not running
	sudo systemctl status docker
	# create the target folder to store new Docker data files, change it to your own path
	mkdir ~/docker
	# migrate old files over
	rsync -avxP /var/lib/docker/ /home/docker
	# update Docker service configure file
	# change line: ExecStart=/usr/bin/dockerd -H fd:... to ExecStart=/usr/bin/dockerd -g /home/docker -H fd:...
	sudo nano /lib/systemd/system/docker.service
	# reload daemons
	sudo systemctl daemon-reload
	# restart Docker
	systemctl start docker
	# verify Docker is available
	sudo docker run hello-world
		
	```
	
### Configure Docker Network
To allow docker containers running in different host machines to communicate with each other, we need to creater a docker network. We assume there are two host machines: `host1` and `host2`, more could be added in a similar fashion.

- First initialize a Docker swarm

	```
	# find host1's local ip
	host1$ hostname -I
	# configure host1 as the master, this will return a join command for workers
	host1$ docker swarm init --advertise-addr $HOST1_LOCAL_IP --listen-addr $HOST1_LOCAL_IP:6666
	# configure host2 as the worker, using command generated from host1's swarm init
	docker swarm join --token SWMTKN-1-4d61y6l7dgw7cycz5wxyabfg6ew5q6qgeb1fgqfas2pba1jspj-3d8ph41e33q2bb0uixaht1eft $HOST1_LOCAL_IP:6666
	```
	
- Create an overlay network for container communications

	```
	docker network create --driver=overlay --attachable fedscale-net
	```
	
### FedScale Docker Demo
This is a walkthrough to run `Femnist` benchmark using docker. We have provided example containerized configs files under `$FEDSCALE_HOME/benchmark/configs` (e.g. `$FEDSCALE_HOME/benchmark/configs/femnist/conf_docker.yml`) for docker jobs and integrated our `driver.py` with docker support.  Suppose for the `Femnist` benchmark task, we want to run `Aggregator` on `host1` and one `Executor` on `host2`:

- Prepare dataset and specify `data_path` in your `conf_docker.yml`. This path will be mounted into the container so that datasets can be reused in future runs.

- Specify `container_network` as the docker network name you have configured (e.g. `fedscale-net`).

- Each `Aggregator` or `Executor` will need a port for process communication, make sure you choose ports that are open and specify `ports` in your `conf_docker.yml`. Note that in our example we can choose the same port for `Aggregator` and `Executor` since they are on different host machines.


- Submit job

	```
	cd $FEDSCALE_HOME/docker
	python3 driver.py submit $FEDSCALE_HOME/benchmark/configs/femnist/conf_docker.yml
	```

- Check logs: FedScale will generate logs under `data_path` you provided by default.

- Stop job

	```
	cd $FEDSCALE_HOME/docker
	python3 driver.py stop $YOUR_JOB
	```

	
## Contribute to FedScale Container Images
If you make changes to FedScale's aggregator or executor source files and want to build new images, follow the steps below.

### Update Aggregator Image

```
cd $FEDSCALE_HOME/docker/config
docker build --tag fedscale-aggr -f  Dockerfile.fedscale_aggr ../../
docker tag fedscale-aggr fedscale/fedscale-aggr
docker push fedscale/fedscale-aggr
```

### Update Executor Image

```
cd $FEDSCALE_HOME/docker/config
docker build --tag fedscale-exec -f  Dockerfile.fedscale_exec ../../
docker tag fedscale-exec fedscale/fedscale-exec
docker push fedscale/fedscale-exec
```

### Update Aggr/Exec Initialization
Aggregator/executor image will run scripts `$FEDSCALE_HOME/docker/config/run-aggr.sh` or `$FEDSCALE_HOME/docker/config/run-exec.sh` by default. The scripts will directly invoke `$FEDSCALE_HOME/fedscale/cloud/aggregation/aggregator.py` or `$FEDSCALE_HOME/fedscale/cloud/execution/executor.py`. If you want to change, for example, how to initialize aggregator/executor using different arguments, make sure you change commands in the corresponding `.sh` scripts.


		