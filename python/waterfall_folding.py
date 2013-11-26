from matplotlib import pyplot as plt
import matplotlib.cm as cm
from math import ceil
import numpy as np
import struct
import os, sys

# Store global parameters
args = { 'nsamp'   : 65536,
         'nchans'  : 256,
         'nbits'   : 16,
         'offline' : False,
         'integs'  : 32}

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
    return np.reshape(data, (nsamp, nchans))

def plot_bandpass(data, ax):
    """ Plot bandpass """

    print "Generating Bandpass"
    x = range(nchans)
    if 'fch1' in args.keys() and 'foff' in args.keys():
        x = np.arange(fch1, fch1 - (foff * nchans), -foff)

    ax.semilogy(x[::-1], np.sum(data, axis=0) / (nsamp * integs * 1.0), 'r')
    ax.grid(True)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('dB')
    ax.set_title('Bandpass plot')

if __name__ == "__main__":

    # Process command-line arguments
 	if len(sys.argv) < 3:
		print "Not enough arguments!"
		print "python processRawFile.py filename nsamp=x nchans=x nbits=x dm=x tsamp=x period=x"

	args['filename'] = sys.argv[1]

	for item in sys.argv[2:]:
		ind = item.find('=')
		if ind > 0:
			args[item[:ind]] = eval(item[ind + 1:])

	for k, v in args.iteritems():
		globals()[k] = v

	f = open(filename, 'rb')	
	data = np.zeros((nsamp, nchans))

	try:
		while True:
			data += read_data(f)
	except Exception as a:
		print a

	plt.imshow(data, aspect='auto')
	plt.show()

