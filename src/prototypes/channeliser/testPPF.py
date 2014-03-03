import struct, numpy as np
from matplotlib import pyplot as plt

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


nsamp = 65536
nchans = 256
nsubs  = 32
nspectra = nsamp / nchans

# Input
f = open('input.dat', 'rb')
data = struct.unpack('f' * nsamp * nsubs * 2, f.read())
data = [complex(r,i) for r, i in zip(*[iter(data)]*2)]
#plt.plot(np.fft.fftshift(np.fft.fft(data)))
#plt.show()

# Output (output directly from FFT, need to fix things)
#f = open('output.dat', 'rb')
#data = struct.unpack('f' * nsamp * nsubs * 2, f.read())
#data = [complex(r,i) for r, i in zip(*[iter(data)]*2)]

#fig = plt.figure()
#transposed = np.reshape(data, (nsubs, nspectra, nchans))
#transposed = np.transpose(transposed[0,:,:])
#data = np.abs(transposed)
#data[np.where(data == 0)] = 0.00001
#data = np.log10(data)

#ax = fig.add_subplot(211)
#ax.imshow(data, aspect='auto')

#ax = fig.add_subplot(223)
#ax.set_title("Bandpass")
#ax.plot(np.sum(data, axis=1))

#ax = fig.add_subplot(224)
#ax.plot(np.sum(data, axis=0))
#ax.set_title("Time series")
#plt.show()


# Process output file
f = open('output.dat', 'rb')
data = struct.unpack('f' * nsamp * nsubs, f.read())
data = np.reshape(data, (nspectra, nchans * nsubs))
data[np.where(data == 0)] = 0.0001
plt.imshow(np.log10(data[:,0:nchans].T), aspect='auto', interpolation='nearest')
plt.colorbar()
plt.show()
