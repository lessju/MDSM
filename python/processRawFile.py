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
         'integs'  : 1 }

# Create main figure
fig = plt.figure()
fig.subplots_adjust(left=0.2, wspace=0.2)

def read_data():
    """ Read data from the file and store as a numpy matrix """
    f = open(args['filename'])

    if nbits == 8:
        mode = 'B'
    elif nbits == 16:
        mode = 'H'
    elif nbits == 32:
        mode = 'f'
    else:
        print nbits + " bits not supported"
        exit()

#    if integs > 1:
#        i = 0
#        vector = np.array([])
#        while i < nsamp:
#            data = f.read(integs * nchans * nbits / 8)
#            data = struct.unpack(integs * nchans * mode, data)
#            data = np.reshape(data, (integs, nchans))
#            data = np.sum(data, axis=0) / integs
#            vector = np.append(vector, data)
#            if i % 100 == 99:
#                print i
#            i += 1
#        return np.reshape(vector, (nsamp, nchans))

#    else:
    f.seek(1024 * nchans * nbits / 8)  # HACK
    data = f.read(nsamp * nchans * nbits / 8)
    data = struct.unpack(nsamp * nchans * mode, data)
    return np.reshape(data, (nsamp, nchans))

def plot_bandpass(data, ax):
    """ Plot bandpass """

    print "Generating Bandpass"
    x = range(nchans)
    if 'fch1' in args.keys() and 'foff' in args.keys():
        x = np.arange(fch1, fch1 - (foff * nchans), -foff)

    band = np.sum(data, axis=0) / (nsamp * 1.0)

    ax.plot(x[::-1], band, 'r')
    ax.grid(True)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('dB')
    ax.set_title('Bandpass plot')

def plot_dedispersed(data, ax):
    """ Dedisperse data with a given DM """

    print "Dedispersing Data"
    dedisp = lambda f1, f2, dm, tsamp: 4148.741601 * (f1**-2 - f2**-2) * dm / tsamp
    shifts = [dedisp(fch1 + diff, fch1, dm, tsamp) 
             for diff in -np.arange(0, foff * nchans, foff)]

    # Roll each subband by its shift to remove dispersion
    for i in range(nchans):
        data[:,i] = np.roll(data[:,i], -int(shifts[nchans - 1 - i]))
    dedispersed = np.sum(data, 1)

    x = np.arange(0, tsamp * int(nsamp - ceil(max(shifts))), tsamp)
    ax.plot(x, dedispersed[:np.size(x)])
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power')
    ax.set_title('Dedispersed Time Series (DM %.3f)' % args['dm'])
 
    return dedispersed

def plot_profile(data, ax):
    """ Fold the data with a given period """

    print "Creating Pulsar Profile"
    bins = period / tsamp
    int_bins = int(bins)
    profile = np.zeros(int_bins)

    for i in range(int_bins):
        for j in range(int(nsamp / bins)):
            profile[i] += data[int(j * bins + i)]

    x = np.arange(0, tsamp * int_bins, tsamp)
    ax.plot(x[:np.size(profile)], profile)
    ax.grid(True)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power')
    ax.set_title('Folded Time Series (Period %.3fms)' % (period * 1000))

    return profile

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

    # Update params
    integs=1
    tsamp = tsamp * integs

    # Read data
    print "Reading data..."
    data = read_data()

    # Generate plots
    if not set(['dm', 'fch1', 'foff', 'tsamp']).issubset(args.keys()):
        plot_bandpass(data, fig.add_subplot(111))

    elif not set(['period']).issubset(set(args.keys())):
        plot_bandpass(data, fig.add_subplot(212))
        dedispersed = plot_dedispersed(data, fig.add_subplot(211))

    else:
        plot_bandpass(data, fig.add_subplot(224))
        dedispersed = plot_dedispersed(data, fig.add_subplot(211))
        profile = plot_profile(dedispersed, fig.add_subplot(223))
    
    if offline:
        fig.set_size_inches(12, 10)
        plt.savefig("processedRaw.png", dpi = 100)
    else:
        plt.show()
