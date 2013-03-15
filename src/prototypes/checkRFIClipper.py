from matplotlib import pyplot as plt
from sys import argv
import numpy as np
import struct

if __name__ == "__main__":

    # Open file and reshape data
    f = open(argv[1], 'rb')
    nchans = int(argv[2])
    data = f.read()
    data = np.array(struct.unpack('f' * (len(data) / 4), data))
    data = np.reshape(data, (nchans, len(data) / nchans))

    # Plot data and fix axes
    plt.imshow(data, aspect='auto')
    plt.xlabel('Time spectrum')
    plt.ylabel('Frequency channel')
    plt.colorbar()
    plt.show()
    
#        plt.plot((np.sum((data), axis=1) / ((len(data) / nchans))).tolist())
#        plt.show()

