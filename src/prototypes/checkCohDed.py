from matplotlib import pyplot as plt
import struct
import numpy as np

if __name__ == "__main__":

    f = open('/home/lessju/Code/MDSM/release/pelican-mdsm/pipelines/channel128_power_B0329+54_2.dat', 'rb')
    data = f.read(512 * 16384 * 4)
    data = np.array(struct.unpack('f' * (len(data) / 4), data))

    bins      = int(0.714519699726 / 5.12e-5);
    profile   = np.zeros(bins)
    added     = np.zeros(bins)
    decFactor = 8

    for i in range(len(data)):
        profile[i % bins] += data[i]
        added[i % bins] += 1

    for i in range(int(np.min(added))):
        signal = [];
        for j in range(bins / decFactor):
            x = 0
            for k in range(decFactor):
                x += data[i * bins + j * decFactor + k]
            signal.append(x / decFactor + i * 2e5)
        plt.plot(signal, 'b')            
    plt.show()

#    signal = []
#    print "Folded ", np.min(added), " periods"
#    for i in range(len(profile) / decFactor):
#        x = 0;
#        for j in range(decFactor):
#            x += profile[i * decFactor + j] / added[i * decFactor + j]
#        signal.append(x / decFactor)
#    plt.plot(signal)
#    plt.show()
