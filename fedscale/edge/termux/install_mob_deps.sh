# Install system packages
sudo apt update
# Newer versions of libc6 cannot work with kernels with revision number > 255
# E.g. Pixel 3 has a kernel 4.19.270 (270>255)
sudo apt install --allow-downgrades -y libc6=2.31-0ubuntu9
sudo apt-mark hold libc6
sudo apt install -y libc6-dev=2.31-Oubuntu9 libc-dev-bin=2.31-0ubuntu9
sudo apt install -y git python3.8 python3-dev python3-pip build-essential gfortran liblapack-dev libatlas-base-dev libfreetype6-dev pkg-config

# Install python dependences
# NOTE: We use sudo since some wheels dont build in user-space
sudo pip install -r mob_pip_deps.txt -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
# Remove torch, so that a custom version be installed later
sudo pip uninstall torch
