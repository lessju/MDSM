from matplotlib import pyplot as plt
from sys import argv
import numpy as np
import struct

ndms  = 32

if __name__ == "__main__":

    f = open(argv[1], 'rb')
    data = f.read()
    data = np.array(struct.unpack('f' * (len(data) / 4), data))
    data = np.reshape(data, (ndms, len(data) / ndms))

    plt.imshow(data, aspect='auto')
    plt.colorbar()
    plt.show()

#    for i in range(0,ndms):
#        plt.plot(data[i,:])
#    plt.show()

#        plt.plot((np.sum((data), axis=1) / ((len(data) / nchans))).tolist())
#        plt.show()

