from matplotlib import pyplot as plt
from sys import argv
import numpy as np
import struct

checkingHuber = False

if __name__ == "__main__":

    if checkingHuber:

        # Load thresholded data
        f = open(argv[1], 'rb')
        nchans = int(argv[2])
        means = f.read(nchans * 4)
        stddevs = f.read(nchans * 4)
        data = f.read(8192 * 512 * 4)

        means = np.array(struct.unpack('f' * nchans, means))
        stddevs = np.array(struct.unpack('f' * nchans, stddevs))
        data = np.array(struct.unpack('B' * len(data), data))
 
        data = np.reshape(data, (len(data) / nchans, nchans))
        data = (data * stddevs) + means

        plt.imshow(data, aspect='auto')
        plt.colorbar()
        plt.show()
        
#        plt.plot((np.sum((data), axis=0) / ((len(data) / nchans))).tolist())
#        plt.show()

    else:
        f = open(argv[1], 'rb')
        nchans = int(argv[2])
        data = f.read()
        data = np.array(struct.unpack('f' * (len(data) / 4), data))
        data = np.reshape(data, (nchans, len(data) / nchans))

        plt.imshow(data, aspect='auto')
        plt.colorbar()
        plt.show()
        
#        plt.plot((np.sum((data), axis=1) / ((len(data) / nchans))).tolist())
#        plt.show()

