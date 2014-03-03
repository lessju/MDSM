from matplotlib import pyplot as plt
import numpy as np

nsamp = 65536
nsub  = 1024

# Generate real signal 
data = np.sin([complex(i,1) for i in np.arange(0, nsamp * nsub * 0.01, 0.01)])

# Split into subbands
data = np.reshape(data, (nsub, nsamp))

# Fourier transform each spectrum
waterfall = np.zeros((nsub, nsamp), dtype=complex)
for i in xrange(nsamp):
    waterfall[:,i] =np.fft.fft(data[:,i])

plt.imshow(np.log10(np.abs(waterfall)), aspect='auto', interpolation='nearest')
plt.show()
