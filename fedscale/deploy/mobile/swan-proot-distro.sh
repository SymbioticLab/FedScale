# Added By Sanjay
DISTRO_NAME="Ubuntu (focal) for Swan"

TARBALL_URL['aarch64']="https://github.com/termux/proot-distro/releases/download/v2.2.0/ubuntu-aarch64-pd-v2.2.0.tar.xz"
TARBALL_SHA256['aarch64']="486de37668963c1b6a0d131e33b91486be8eb1919b0813ad03726885753feba6"

distro_setup() {
	# Termux Limitations
	run_proot_cmd apt-mark hold gvfs-daemons udisks2
	# Setup for Swan
	run_proot_cmd apt update
	run_proot_cmd apt install --allow-downgrades -y libc6=2.31-0ubuntu9
	run_proot_cmd apt-mark hold libc6
	run_proot_cmd apt install -y sudo htop git libc6-dev=2.31-0ubuntu9 libc-dev-bin=2.31-0ubuntu9 python3.8 python3-dev python3-pip build-essential libomp-dev gfortran liblapack-dev libatlas-base-dev libfreetype6-dev pkg-config
        # Create user
	run_proot_cmd useradd fedscale
	run_proot_cmd usermod -aG sudo fedscale
	run_proot_cmd echo "fedscale:123456" | chpasswd
	run_proot_cmd mkdir /home/fedscale
}
