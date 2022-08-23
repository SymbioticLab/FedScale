#!/bin/bash
FEDSCALE_HOME=$(pwd)
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc
echo alias fedscale=\'bash ${FEDSCALE_HOME}/fedscale.sh\' >> ~/.bashrc

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

# un-comment to install anaconda
isPackageNotInstalled conda


# create conda env
conda init bash
. ~/.bashrc
conda env create -f environment.yml # Install dependencies
conda activate fedscale


if [ "$1" == "--cuda" ]; then
  wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
  sudo apt-get purge nvidia-* -y
  sudo sh -c "echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf"
  sudo update-initramfs -u
  sudo sh cuda_10.2.89_440.33.01_linux.run --override --driver --toolkit --samples --silent
  export PATH=$PATH:/usr/local/cuda-10.2/
  conda install cudatoolkit=10.2 -y
fi

# install Redis server

curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

sudo apt-get update
sudo apt-get install redis