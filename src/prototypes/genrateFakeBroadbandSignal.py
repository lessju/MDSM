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

    nsamp = 1024*64

    nchans = 32
    dm     = 200
    foff   = 100.0 / nchans
    fch1   = 1200 - foff / 2.0
    tsamp  = 5.12e-5
    
    # Generate data
    data = generateFakeSignal(nsamp, nchans, fch1, foff, dm, tsamp)

    # Write data to file (loop over same signal for N times)
    f = open('generatedData.raw', 'wb')
    N = 512
    zero = struct.pack('f', 0)
    for i in range(N):
        for j in range(nchans):
            for k in range(nsamp):
                f.write(zero)
                f.write(struct.pack('f', data[j,k]))
    f.close()
