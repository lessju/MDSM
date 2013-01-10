from matplotlib import pyplot as plt
from sys import argv
import numpy as np
import struct
import pylab

if __name__ == "__main__":

    orig    = open(argv[1], 'rb')
    decoded = open(argv[2], 'rb')
    nchans  = int(argv[3])
    nbits   = int(argv[4])

    if nbits == 32:
        format = 'f'
    elif 16:
        format = 'h'

    # Read original data
    orig_data = orig.read()
    orig_data = np.array(struct.unpack(format * (len(orig_data) / (nbits / 8)), orig_data))

    # Read decoded data
    decoded_data = decoded.read()
    decoded_data = np.array(struct.unpack(format * (len(decoded_data) / (nbits / 8)), decoded_data))

    # Create plots
    fig = pylab.figure()
    fig.subplots_adjust(left = 0.2, wspace = 0.2)

    ax = fig.add_subplot(221)  
    ax.hist(orig_data, bins=100)

    ax = fig.add_subplot(222)  
    orig_data = np.reshape(orig_data, (nchans, len(orig_data) / nchans))
    ax.imshow(orig_data, aspect='auto')

    ax = fig.add_subplot(223)  
    ax.hist(decoded_data, bins=100)

    ax = fig.add_subplot(224)  
    decoded_data = np.reshape(decoded_data, (nchans, len(decoded_data) / nchans))
    ax.imshow(decoded_data, aspect='auto')

    pylab.show()


