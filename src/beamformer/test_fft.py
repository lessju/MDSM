from matplotlib import pyplot as plt
from optparse import OptionParser
import numpy as np, struct
import sys, math, os, sys

if len(sys.argv) < 2:
    print "Input file needed"
    exit()

# Complex
#nsamp  = 4096
#nchans = 1024
#f = open(sys.argv[1], 'rb')
#data = f.read()
#data = np.array(struct.unpack('f' * nchans * nsamp * 2, data))
#data = np.reshape(data, (nchans, nsamp, 2))

#power  = np.sqrt(data[:,:,0] ** 2.0 + data[:,:,1] ** 2.0)
#power[np.where(power > 1.2e10)] = 0

#plt.imshow(power.T, aspect='auto')
#plt.colorbar()
#plt.show()

# Real
nsamp  = 16384 / 256
nchans = 8 * 256
f = open(sys.argv[1], 'rb')
data = f.read()
data = np.array(struct.unpack('f' * nchans * nsamp, data))
data = np.reshape(data, (nsamp, nchans))

#plt.imshow(data, aspect='auto')
plt.plot(np.log10(np.sum(data, axis=0)))
#plt.colorbar()
plt.show()
