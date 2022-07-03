# Setup Android Edge Device

This tutorial will help you setup the environment required to use an Android device as a FedScale edge-device

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
apt install openssh
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

## Download and Install Swan Container

After connecting to the phone using the previous steps, run the [mobile\_setup\_simple.sh](./mobile_setup_simple.sh) script to install all depdencies for the container and then the container itself. Start the container with `proot-distro swan-proot-distro --user fedscale`.
