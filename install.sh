#!/usr/bin/env python
#!/bin/bash
FEDSCALE_HOME=$(pwd)
if [[ $(uname -s) == 'Darwin' ]]; then
  echo MacOS
  echo export FEDSCALE_HOME=$(pwd) >> ~/.bash_profile
  echo alias fedscale=\'bash ${FEDSCALE_HOME}/fedscale.sh\' >> ~/.bash_profile
  
else
  echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc
  echo alias fedscale=\'bash ${FEDSCALE_HOME}/fedscale.sh\' >> ~/.bashrc
fi


isPackageNotInstalled() {
  $1 --version &> /dev/null
  if [ $? -eq 0 ]; then
    echo "$1: Already installed"
  elif [[ $(uname -p) == 'arm' ]]; then
    install_dir=$HOME/miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash  Miniconda3-latest-MacOSX-arm64.sh -b -p  $install_dir
    export PATH=$install_dir/bin:$PATH
  else
    install_dir=$HOME/anaconda3
    wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    bash Anaconda3-2020.11-Linux-x86_64.sh -b -p  $install_dir
    export PATH=$install_dir/bin:$PATH

  fi
}

# un-comment to install conda
isPackageNotInstalled conda


# create conda env


if [[ $(uname -p) == 'arm' ]]; then
  source ~/miniconda/bin/activate
  . ~/.bash_profile
  conda env create -f environment-arm.yml
  conda install -c apple tensorflow-deps
  conda activate fedscale
  python -m pip install tensorflow-macos==2.9
  python -m pip install tensorflow-metal==0.5.0
  
else
  conda init bash
  . ~/.bashrc
  conda env create -f environment.yml
  conda activate fedscale

fi



if [ "$1" == "--cuda" ]; then
  wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
  sudo apt-get purge nvidia-* -y
  sudo sh -c "echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf"
  sudo update-initramfs -u
  sudo sh cuda_10.2.89_440.33.01_linux.run --override --driver --toolkit --samples --silent
  export PATH=$PATH:/usr/local/cuda-10.2/
  conda install cudatoolkit=10.2 -y
fi

