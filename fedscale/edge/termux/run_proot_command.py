# To be placed in Termux $HOME
import datetime
import os
import sys
import time

# Get arg
print(sys.argv)
assert(len(sys.argv) >= 2)

command = " ".join(sys.argv[1:])
print(f"Command: {command}")

now = time.time()

# Save to file
with open(f'proot_commands/{now}.sh','w') as f:
    f.write('#' + datetime.datetime.fromtimestamp(now).strftime('%c') + '\n')
    f.write(command)

# Run proot
proot_login = 'proot-distro login swan-proot-distro --user fedscale'
os.system(f"{proot_login} -- bash /home/fedscale/termux-home/proot_commands/{now}.sh")
