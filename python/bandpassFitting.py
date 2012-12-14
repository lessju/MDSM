from matplotlib import pyplot as plt
from math import ceil, sqrt, fabs
import matplotlib.cm as cm
import numpy as np
import struct
import os, sys

# Some other parameters
channelRejectionRMS   = 40
spectrumRejectionRMS  = 5

# Store global parameters
args = { 'nsamp'     : 65536,
         'nchans'    : 256,
         'nbits'     : 16,
         'ncoeffs'   : 4,
         'topFreq'   : 1899,
         'bandwidth' : 100}

def read_data(f):
    """ Read data from the file and store as a numpy matrix """

    if nbits == 8:
        mode = 'B'
    elif nbits == 16:
        mode = 'H'
    elif nbits == 32:
        mode = 'f'
    else:
        print nbits + " bits not supported"
        exit()

    data = f.read(nsamp * nchans * nbits / 8)
    data = np.array(struct.unpack( nsamp * nchans * mode, data ))
    data = np.reshape(data, (nsamp, nchans))

    # Inject some RFI
    data[512:728,:] *= 1.5
    data[:,264:266] *= 4
    return data

def fit_bandpass(data):
    """ Fit bandpass to channelised data """

    ncoeffs = 9

    # Sum data to generate bandpass
    bandpass = np.sum(data, axis=0) / nsamp
    x_vals   = np.arange(0, 1, 1.0 / nchans)

    # Fit bandpass
    bandpassFit = np.polyfit(x_vals, bandpass, ncoeffs - 1)
    bandpassFit = np.poly1d(bandpassFit)
    bandpassFit = bandpassFit(x_vals)

    # Calculate channel rejection margin (channel rejection RMS * bandpass RMS)
#    channel_margin = channelRejectionRMS * np.sqrt(np.mean(bandpassFit) * np.mean(bandpassFit) + np.std(bandpassFit) * np.std(bandpassFit))
#    channel_margin = channelRejectionRMS * np.sqrt(np.sum(bandpassFit ** 2) / nchans)
    channel_margin = channelRejectionRMS * np.std((bandpassFit - np.mean(bandpassFit)))

    # Cast data as a numpy matrix
    data = np.matrix(data)

    # Subtract bandpass from spectra
    subtracted_data = data - bandpassFit

    print channel_margin

#    plt.plot( ((np.sum(data, axis=0) / nsamp).tolist())[0] )
#    plt.plot( ((np.sum(subtracted_data, axis=0) / nsamp).tolist())[0], 'r')
#    plt.plot(bandpassFit, 'g')
#    plt.show()

    # Clip bright channels (data - bandpassFit - median of each spectrum)
#    subtracted_data[np.where(subtracted_data - np.median(subtracted_data, axis = 1) > channel_margin)] = 10e7


    # Calculate spectrum sums and spectrum sum squared
    spectrumSum   = np.sum(subtracted_data, axis = 1)
    spectrumSumSq = np.sum(np.multiply(subtracted_data, subtracted_data), axis = 1)

    # Scale by number of good channels in each spectrum and 
    # calculate RMS for each spectrum, and scale by nbins / goodChannels in case
    # number of good channels is substantially lower than the total number
    spectrumRMS = np.zeros(nsamp)
    for i in range(nsamp):
        goodChannels =  len([1 for x in subtracted_data[i,:].T if x != 0])
        spectrumSum[i]   /= goodChannels
        spectrumSumSq[i] /= goodChannels

        spectrumRMS[i] = np.sqrt(spectrumSumSq[i] - np.multiply(spectrumSum[i], spectrumSum[i]))
        spectrumRMS[i] *= np.sqrt((nchans * 1.0) / goodChannels)

    # Compute bandpass median
    bandpassMedian = np.median(bandpassFit)
    
    # Re-compute spectrum median and check whether it fits the model
    # If no, then something's not right and we attribute this to RFI    
    # If yes, we use the value to update the model

    # Calucate new spectra median (since some channels might have been chopped off)
    spectraMedian = np.median(subtracted_data, axis = 1)

    # Calculate spectrum RMS tolerance
#    spectrumRMStolerance = spectrumRejectionRMS * (np.sqrt(np.mean(bandpassFit) + np.std(bandpassFit))) / sqrt(nchans);
    spectrumRMStolerance = spectrumRejectionRMS * np.std((bandpassFit - np.mean(bandpassFit)))

    print np.sqrt(np.mean(bandpassFit) + np.std(bandpassFit)), channel_margin, spectrumRMStolerance

    # if spectrum median is higher than accepted tolerance, clip it
    for i in range(nsamp):
        if fabs(spectraMedian[i]) > spectrumRMStolerance:
            subtracted_data[i,:] = np.zeros(nchans) * 10e7

    plt.imshow(subtracted_data.T.tolist() , aspect='auto')
    plt.colorbar()
    plt.show()





if __name__ == "__main__":

    # Process command-line arguments
    if len(sys.argv) < 3:
	    print "Not enough arguments!"
	    print "python processRawFile.py filename nsamp=x nchans=x nbits=x ncoeffs=x"

    args['filename'] = sys.argv[1]

    for item in sys.argv[2:]:
	    ind = item.find('=')
	    if ind > 0:
		    args[item[:ind]] = eval(item[ind + 1:])

    for k, v in args.iteritems():
	    globals()[k] = v

    # Open file and read data
    f = open(filename, 'rb')	
    data = read_data(f)
    print "Read data"

    # Fit bandpass
    fit_bandpass(data)


