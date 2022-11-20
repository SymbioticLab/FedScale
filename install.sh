#!/bin/bash
set -Eeuo pipefail

INST_CUDA=0
INST_REDIS=0

printHelp() {
    echo "Usage: $0 [ --cuda ] [ --redis ] [ --help ]"
}

while [[ $# -gt 0 ]];
do
	case "$1" in
	--cuda)
		INST_CUDA=1
		;;
	--redis)
		INST_REDIS=1
		;;
	--help)
		printHelp
		exit 0
		;;
	--)
		break
		;;
	*)
		echo "Unexpected option: $1"
		printHelp
		exit 2
		;;
	esac
	shift;
done

isPackageNotInstalled() {
  $1 --version &> /dev/null
  if [ $? -eq 0 ]; then
    echo "$1: Already installed"
  else
    install_dir=$HOME/anaconda3
    wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    bash Anaconda3-2020.11-Linux-x86_64.sh -b -p  $install_dir
    export PATH=$install_dir/bin:$PATH
  fi
}

installCuda() {
  wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
  sudo apt-get purge nvidia-* -y
  sudo sh -c "echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf"
  sudo update-initramfs -u
  sudo sh cuda_10.2.89_440.33.01_linux.run --override --driver --toolkit --samples --silent
  export PATH=$PATH:/usr/local/cuda-10.2/
  conda install cudatoolkit=10.2 -y
}

installRedisServer() {
	curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
	echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
	sudo apt-get update
	sudo apt-get install redis
}

FEDSCALE_HOME=$(pwd)
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc
echo alias fedscale=\'bash ${FEDSCALE_HOME}/fedscale.sh\' >> ~/.bashrc

# un-comment to install anaconda
isPackageNotInstalled conda

# create conda env
conda init bash
. ~/.bashrc
conda env create -f environment.yml # Install dependencies
conda activate fedscale

if [ $INST_CUDA -eq 1 ]; then
	installCuda
fi

# install Redis server (client in python package)
if [ $INST_REDIS -eq 1 ]; then
	installRedisServer
fi