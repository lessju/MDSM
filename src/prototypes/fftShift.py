from matplotlib import pyplot as plt
import numpy as np
import struct


def dmDelay(dm, fhi, flo):
    """ Cacluate dispersion delay"""
    return 4148.741601 * (fhi**-2 - flo**-2) * dm

def generateFakeSignal(N, nchans, fch1, foff, dm, tsamp):
    """ Generate a fake signal to test FFT shift"""

    data = np.random.normal(size=(nchans, N))
    for i in range(nchans):
        delay = dmDelay(dm, fch1 - foff * i, fch1) / tsamp
        data[i, N * 0.25 + delay : N * 0.25 + delay + N * 0.01] += 4
    return data
    

if __name__ == "__main__":
    """ Prototype for FFT shift """

    N = 1024*32

    nchans = 32
    dm     = 200
    foff   = 100.0 / nchans
    fch1   = 1200 - foff / 2.0
    tsamp  = 5.12e-5
    
    # Generate data
    data = generateFakeSignal(N, nchans, fch1, foff, dm, tsamp)

    # FFT into fourier domain 
    X = np.fft.fft(data)
    
    # Generate co-efficients and perform shift for each channel
    C = np.concatenate((np.arange(0,N/2), np.arange(-N/2, 0)))
    
    for i in range(1, 2):
        delay = dmDelay(dm, fch1 - foff * i, fch1) / tsamp
        W     = np.array(2 * np.pi * -delay * C / N )
        print  delay, 2 * np.pi * -delay / N

        if N % 2 == 0:
	        W[N/2+1] = np.real(W[N/2+1])

#        plt.plot(np.fft.fftshift(np.real(X[i] * (np.cos(W) - 1j * np.sin(W)))))
        plt.plot(np.imag(X[i]))
        X[0] += X[i] * (np.cos(W) - 1j * np.sin(W))

    # Inverse FFT
    Y = np.fft.ifft(X[0])
    
    # Plot stuff
#    plt.plot(np.abs(W))
    plt.show()
    
    
