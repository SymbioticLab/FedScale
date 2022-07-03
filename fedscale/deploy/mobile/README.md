# Setup Android Edge Device From Scratch

This tutorial will help you setup the environment required to use an Android device as a FedScale edge-device from scratch.

## Installing Termux

FedScale will be installed within [Termux](https://termux.org/), which provides a Linux terminal environment within Android.

1. Download and install the [FDroid app store](https://f-droid.org/).
2. Install the [Termux app](https://f-droid.org/en/packages/com.termux/) and the accompanying [Termux-API app](https://f-droid.org/en/packages/com.termux.api/).

## Setting up Termux-environment

First, update the package registry.
```
apt update
```

Install necessary apps within the Termux environment to enable making the.
```
apt install openssh termux-api
```

Start the SSH server and find the IP-address (e.g. W.X.Y.Z below) to connect to the phone.
```
sshd && ifconfig
[...]
wlan0: ...
        inet W.X.Y.Z
```

Set the password using `passwd`. Connect to the phone using `ssh` in the following way.
```
ssh W.X.Y.Z -p 8022
```

## Install Ubuntu Distro

Follow the instructions [here](https://github.com/tuanpham-dev/termux-ubuntu) to install the Ubuntu Focal (20.04) distro within Termux.

## (OR) Install Ubuntu proot-distro for Swan

1. Install [proot\-distro](https://github.com/termux/proot-distro) using `apt install proot-distro`.
2. Place the [swan\-proot\-distro.sh](swan-proot-distro.sh) in `$HOME/../usr/etc/proot-distro`.
3. Install the distro using `proot-distro install swan-proot-distro`.
4. Log-in to the distro using `proot-distro login swan-proot-distro --user fedscale`.

## Setting up Ubuntu environment

Once within the distro installation of your choice, update the package registry and install dependences by running the script [install\_mob\_deps.sh](install_mob_deps.sh).

Set and export the `SWAN_PREFIX` environment varaible to point the scripts to installation location.

## Build OpenBLAS

Using the script [build\_blas.sh](build_blas.sh), build and install OpenBLAS with OpenMP support for multi-threading.

## Build PyTorch

Using the script [build\_pytorch.sh](build_pytorch.sh), build and install PyTorch with BLAS acceleration and parallel processing.

## Run FedScale

After setting up the environment, you can enjoy playing with FedScale ```core``` implementations (e.g., model training) without reinventing your code in the (cluster) evaluations.