from matplotlib import pyplot as plt
from math import ceil, sqrt, fabs
import matplotlib.cm as cm
import numpy as np
import struct
import os, sys

# Some other parameters
channelRejectionRMS   = 7
spectrumRejectionRMS  = 8

# Store global parameters
args = { 'nsamp'     : 65536,
         'nchans'    : 256,
         'nbits'     : 16,
         'ncoeffs'   : 4,
         'topFreq'   : 1899,
         'bandwidth' : 100 }

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
    data[:,264:266] *= 1.5
    data[0:400,364:366] *= 2

    for i in range(nchans):
        data[i*2:i*2+1,i] *= 4
    
    return data

def fit_bandpass(data):
    """ Fit bandpass to channelised data """

    ncoeffs = 9

#    plt.imshow(data.T, aspect='auto')
#    plt.colorbar()
#    plt.show()

    # Sum data to generate bandpass
    bandpass = np.sum(data, axis=0) / nsamp
    x_vals   = np.arange(0, 1, 1.0 / nchans)

    # Fit bandpass
    bandpassFit = np.polyfit(x_vals, bandpass, ncoeffs - 1)
    bandpassFit = np.poly1d(bandpassFit)
    bandpassFit = bandpassFit(x_vals)

    # Calculate channel rejection margin (channel rejection RMS * bandpass RMS)
    corrected_bandpass = bandpass - bandpassFit
    channel_margin = channelRejectionRMS *  np.sqrt(np.mean(corrected_bandpass) ** 2 + np.std(corrected_bandpass ** 2))
 #   channel_margin = channelRejectionRMS * np.std(corrected_bandpass)    
  #  print np.sqrt(np.mean(corrected_bandpass) ** 2 + np.std(corrected_bandpass ** 2)), np.std(corrected_bandpass)

    # Caluclate spectrum threshold
    spectrumThresh = spectrumRejectionRMS * np.std(bandpassFit) / sqrt(nchans)  
    print "Channel Thresh: ", channel_margin,  " Spectrum Thresh: ", spectrumThresh

    # Cast data as a numpy matrix
    data = np.matrix(data)

    # Subtract bandpass from spectra
    subtracted_data = data - bandpassFit

    # Corrected bandpass
    corrected_bandpass = bandpass - bandpassFit

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax1.imshow(subtracted_data.T.tolist(), aspect='auto')

    box_width = 256
    corrected_data = subtracted_data - np.mean(subtracted_data, axis = 1)
    for i in range(nchans):
        # Use a boxed-mean to clip channels
        for j in range(nsamp - box_width):
            if np.mean(corrected_data[j:j+box_width, i]) > channel_margin:
                subtracted_data[j:j+box_width,i] = corrected_bandpass[i]    
            j += box_width

#    ax1 = fig1.add_subplot(212)
#    ax1.imshow(subtracted_data.T.tolist(), aspect='auto')
#    plt.show()

    # Calculate all spectra mean
    spectrumMean = np.zeros(nsamp)
    for i in range(nsamp):
        goodChannels =  len([1 for x in subtracted_data[i,:].T if x != 0])
        spectrumMean[i] = np.mean(subtracted_data[i,:]) / goodChannels

    for i in range(nsamp):
        if np.mean(subtracted_data[i,:]) > spectrumThresh:
#            print i, np.mean(subtracted_data[i,:])
            subtracted_data[i,:] = subtracted_data[i,:] - (np.mean(subtracted_data[i,:]) - np.mean(corrected_bandpass))

    ax1 = fig1.add_subplot(212)
    ax1.imshow(subtracted_data.T.tolist(), aspect='auto')
    plt.show()
#    plt.plot( ((np.sum(data, axis=1) / nsamp).tolist()) , 'b')
#    ax1.plot(((np.sum(subtracted_data, axis=1) / nsamp).tolist()), 'r')
#    plt.show()


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


