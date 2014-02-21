import struct, numpy as np
from matplotlib import pyplot as plt

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


nsamp = 65536
nchans = 1024
nsubs  = 32
nspectra = nsamp / nchans

# Input
f = open('input.dat', 'rb')
data = struct.unpack('f' * nsamp * nsubs * 2, f.read())
data = [complex(r,i) for r, i in zip(*[iter(data)]*2)]
#plt.plot(np.fft.fftshift(np.fft.fft(data)))
plt.show()

# Output (output directly from FFT, need to fix things)
f = open('output.dat', 'rb')
data = struct.unpack('f' * nsamp * nsubs * 2, f.read())
data = [complex(r,i) for r, i in zip(*[iter(data)]*2)]

transposed = np.reshape(data, (nsubs, nspectra, nchans))
transposed = np.transpose(transposed[30,:,:])
plt.imshow(np.abs(transposed), aspect='auto')
plt.colorbar()
plt.show()
