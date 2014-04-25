from matplotlib import pyplot as plt
import struct, sys
import numpy as np

PACKET = False 

nchans = 1024
nants  = 32

lookup = np.array([0.,1.,2.,3.,4.,5.,6.,7.,-8.,-7.,-6.,-5.,-4.,-3.,-2.,1.])

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Need input file"
        exit()

    if PACKET:
        f = open(sys.argv[1], 'rb')
        data = f.read()
        data = np.array(struct.unpack('B' * len(data), data))
    
        # Fix packet
        fixed = np.zeros(128 * 32, dtype=int)
        for i in range(16):
            for j in range(128):
                fixed[j * 32 + i * 2]     = data[i * 128 * 2 + j * 2]
                fixed[j * 32 + i * 2 + 1] = data[i * 128 * 2 + j * 2 + 1]

        real = lookup[np.bitwise_and(np.right_shift(fixed, 4), 0x0F)]
        imag = lookup[np.bitwise_and(fixed, 0x0F)]
        fixed = np.sqrt(real * real + imag * imag)
        fixed = np.reshape(fixed, (128,32))
        plt.imshow(fixed, aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.show()

    else:
        f = open(sys.argv[1], 'rb')
        data = f.read()
        data = np.array(struct.unpack('B' * len(data), data))
        nsamp = len(data) / (nchans * nants)
        real = lookup[np.bitwise_and(np.right_shift(data, 4), 0x0F)]
        imag = lookup[np.bitwise_and(data, 0x0F)]
        data = np.sqrt(real * real + imag * imag)
        data = np.reshape(data, (nchans, nsamp, nants))

        # Create subplot for every antenna
        fig = plt.figure(figsize=(12,12))
        for i in range(nants):
            ax = fig.add_subplot(6, 6, i + 1)
            new_data = np.sum(data[:,:,i], axis=1) / nsamp
            new_data[np.where(new_data == 0)] = 0.0001
            ax.plot(np.log10(new_data))
            ax.set_xlim((0,1024))
            
#            ax.imshow(data[:,:,i], aspect='auto', interpolation='nearest')

        plt.show()
#        plt.savefig('fig.png')
