from matplotlib import pyplot as plt
import math, random
import numpy as np

nfft = 512

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def pfb_fir(x):
    N = len(x)    # x is the incoming data time stream.
    taps = 4
    L = nfft   # Points in subsequent FFT.
    bin_width_scale = 1.0
    dx = math.pi/L
    X = np.array([n*dx-taps*math.pi/2 for n in range(taps*L)])
    coeff = np.sinc(bin_width_scale*X/math.pi)*np.hanning(taps*L)

    y = np.array([0+0j]*(N-taps*L))
    for n in range((taps-1)*L, N):
        m = n%L
        coeff_sub = coeff[L*taps-m::-L]
        y[n-taps*L] = (x[n-(taps-1)*L:n+L:L]*coeff_sub).sum()

    return y

a = np.zeros(16384)
for i in xrange(16384):
   a[i] = np.sin(i * 0.1) + 5 * np.sin(i * 0.5) + random.random() * 10 
b = pfb_fir(a).tolist()

spectra = np.zeros((nfft, len(b) / nfft))
for i in range(len(b) / nfft):
    spectra[:,i] = np.abs(np.fft.fft(b[i * nfft : (i+1)*nfft]))

plt.imshow(spectra, aspect='auto')
plt.colorbar()
plt.show()
