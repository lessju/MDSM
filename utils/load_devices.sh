#!/bin/bash

modprobe nvidia

if [ "$?" -eq 0 ]; then
# Count the number of NVIDIA controllers found.
N3D=`/usr/bin/lspci | grep -i NVIDIA | grep "3D controller" | wc -l`
NVGA=`/usr/bin/lspci | grep -i NVIDIA | grep "VGA compatible controller" | wc -l`

N=`expr $N3D + $NVGA - 1`
for i in `seq 0 $N`; do
mknod -m 660 /dev/nvidia$i c 195 $i
done

mknod -m 660 /dev/nvidiactl c 195 255
else
exit 1
fi
