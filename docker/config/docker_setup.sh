#!/bin/bash
# A bash script for automating Docker setup
# Author: Yile Gu

echo "Installing required packages for docker..."
sudo apt-get update -y
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo "Adding Docker GPG key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

printf "%s\n" "deb [arch=amd64  signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu xenial stable" |\
sudo tee /etc/apt/sources.list.d/docker.list

echo "Installing Docker Engine..."
sudo apt-get update -y
sudo apt-get install docker-ce docker-ce-cli containerd.io -y


echo "Configuring Docker deamon's cgroup driver..."
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF

sudo systemctl enable docker
sudo systemctl daemon-reload
sudo systemctl restart docker

echo "Verifying Docker is correctly installed..."
sudo docker run hello-world
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Docker returns error!"
    exit $retVal
fi

echo "Adding user to docker user group..."
sudo usermod -a -G docker $USER