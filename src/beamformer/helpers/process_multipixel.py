from matplotlib import pyplot as plt
import numpy as np, struct
from time import sleep
from scipy import signal
import sys

nbeams = 32
nchans = 4096
tsamp  =  1 / (20e6 / (1024 * 1024.0))

if __name__ == "__main__":

    # Create beam map
    beammap = [(0,3), (0,5), 
               (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8),
               (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), 
               (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), 
               (4,3), (4,4), (4,5)]

    # Open file
    f = open('/data/2014/15_04_2014/37820/37820_2014-04-14_10:19:56.dat', 'rb')
    data = f.read()
    f.close()

    # Read and structure data
    data = np.array(struct.unpack('f' * (len(data) / 4), data), dtype=float)
    nsamp = len(data) / (nchans * nbeams)
    data = np.reshape(data, (nsamp, nchans, nbeams))

    # For each beam
    for i in range(nbeams):

        # Calculate bandpass
        bandpass = np.sum(data[:,:,i], axis=0)

        # Apply median filter bandpass on
        filtered = signal.medfilt(bandpass, 5)
        x        = bandpass - filtered
        outliers = np.where(x - np.mean(x) > np.std(x) * 3)
        bandpass = bandpass / nchans
        mean     = np.mean(bandpass)

        # Remove outlisers
        for item in outliers[0]:
            data[:,item,i] = np.zeros(nsamp) + mean

        # Remove bandpass from data
        data[:,:,i] = data[:,:,i] - bandpass

        # Normalise data
        data[:,:,i] = (data[:,:,i] - np.mean(data[:,:,i])) / np.std(data[:,:,i])

    # Initialise plotting
#    fig = plt.figure()
#    plt.ion()

    global_plot = np.zeros((5,9))

    # Generate multipixel plot
    for i in range(nsamp):

        # Grab array from data
        sample = np.sum(data[i,:,:], axis=0) / nchans

        # Reduce output clutter
        sample[np.where(np.abs(sample) < np.std(sample) * 1.5)] = 0

        for j in range(nbeams):
            if sample[j] != 0:
                global_plot[beammap[j]] += 1

       # global_plot = global_plot + toplot


    plt.imshow(global_plot.T, aspect='auto', interpolation='nearest')
    plt.xlabel("RA")
    plt.ylabel("DEC")
    plt.colorbar()
    plt.show()

    
