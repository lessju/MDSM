from matplotlib import pyplot as plt
import struct, numpy as np

f = open('median-test.dat', 'rb')
d = f.read()
orig = np.array(struct.unpack('f' * (len(d) / 4), d))

f = open('median-test-result.dat', 'rb')
d = f.read()
result = np.array(struct.unpack('f' * len(orig), d))

f = open('median-test-gpu.dat', 'rb')
d = f.read()
gpu = np.array(struct.unpack('f' * len(orig), d))

#plt.plot(orig, 'k')
#plt.plot(result, 'b', linewidth=1.2)
#plt.plot(gpu, 'r')
plt.plot(result - gpu, 'k')
plt.show()


