# Containerized FedScale Tentative Tutorial
This is a tentative tutorial on how to configure a Linux node for Docker service, get FedScale running as container images under Docker's management and contribute to FedScale Docker images. It is written mostly for FedScale developers' reference. We assume the readers start with more than one Linux host machines running Ubuntu 16.04 or newer. 

## Install Docker and its dependencies

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
	
## Configure Container Network
To allow containers running in different host machines to communicate with each other, we need to creater a container network. We assume there are two host machines: `host1` and `host2`, more could be added in a similar fashion.

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
	
## Containerized FedScale Demo
This is a walkthrough to run `Femnist` benchmark on containerized Fedscale. We have provided example containerized configs files under `$FEDSCALE_HOME/benchmark/configs` (e.g. `$FEDSCALE_HOME/benchmark/configs/femnist/conf_ctnr.yml`) for containrized jobs and integrated our `driver.py` with container support.  Suppose for the `Femnist` benchmark task, we want to run `Aggregator` on `host1` and one `Executor` on `host2`:

- Prepare dataset and specify `data_path` in your `conf_ctnr.yml`. This path will be mounted into the container so that datasets can be reused in future runs.

- Specify `container_network` as the docker network name you have configured (e.g. `fedscale-net`).

- Each `Aggregator` or `Executor` will need a port for process communication, make sure you choose ports that are open and specify `ports` in your `conf_ctnr.yml`. Note that in our example we can choose the same port for `Aggregator` and `Executor` since they are on different host machines.


- Submit job

	```
	cd $FEDSCALE_HOME/docker
	python3 driver.py submit $FEDSCALE_HOME/benchmark/configs/femnist/conf_ctnr.yml
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
Aggregator/executor image will run scripts `$FEDSCALE_HOME/docker/config/run-aggr.sh` or `$FEDSCALE_HOME/docker/config/run-exec.sh` by default. The scripts will directly invoke `$FEDSCALE_HOME/fedscale/core/aggregation/aggregator.py` or `$FEDSCALE_HOME/fedscale/core/execution/executor.py`. If you want to change, for example, how to initialize aggregator/executor using different arguments, make sure you change commands in the corresponding `.sh` scripts.


		