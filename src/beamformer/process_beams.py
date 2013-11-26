from matplotlib import pyplot as plt
from matplotlib import ticker
from optparse import OptionParser
import numpy as np, struct
import sys, math, os, sys
import glob, pylab

nbeams = 4
nchans = 1024

def plot(opts):
    """ Plot data """

    print "Generating plots"

    nbeams = opts.nbeams
    nchans = opts.nchans

    f = open(args[0], 'rb')
    data = f.read()
    data = np.array(struct.unpack('f' * (len(data) / 4), data), dtype=float)
    nsamp = len(data) / (nbeams * nchans)
    data = np.reshape(data, (nsamp, nchans, nbeams))

    time      = np.arange(0, opts.tsamp * data.shape[0], opts.tsamp)
    frequency = (np.arange(opts.fch1 * 1e6, opts.fch1 * 1e6 + opts.foff * (data.shape[1]), opts.foff)) * 1e-6
    formatter = pylab.FormatStrFormatter('%2.3f')

    # Process only one beam
    if opts.beam != -1:

        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(1,2,1)
        ax.set_title("Beam %d" % opts.beam)
        ax.imshow(np.log10(data[:,:,opts.beam]), aspect='auto', origin='lower', extent=[frequency[0], frequency[-1], 0, time[-1]])
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Channel (kHz)")
        ax.set_ylabel("Time (s)")

        ax = fig.add_subplot(2,2,2)
        toplot = np.log10(np.sum(data[:,:,opts.beam], axis=0))
        ax.plot(frequency[:len(toplot)], toplot)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Channel (MHz)")
        ax.set_ylabel("Log Power (Arbitrary)")
        ax.set_xlim((frequency[0], frequency[-1]))
        
        ax = fig.add_subplot(2,2,4)
        toplot = np.sum(data[:,:,opts.beam] / nchans, axis=1)
        ax.plot(time[:len(toplot)], toplot)
        ax.xaxis.set_major_formatter(pylab.FormatStrFormatter('%2.1f'))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (Arbitrary)")
        ax.set_xlim((time[0], time[-1]))

        fig.tight_layout()
        plt.show()
        f.close()

        return

    # Plot each beam separately
    fig = plt.figure(figsize=(8,8))
    num_rows = math.ceil(math.sqrt(nbeams))
    for i in range(nbeams):

        ax = fig.add_subplot(num_rows, num_rows, i + 1)
        ax.set_title("Beam %d" % i)

        # Show beam
        if opts.waterfall:
            ax.imshow(np.log10(data[:,:,i]), aspect='auto', origin='lower', extent=[frequency[0], frequency[-1], 0, time[-1]])
            ax.xaxis.set_major_formatter( pylab.FormatStrFormatter('%2.2f'))
            ax.set_xlabel("Channel (kHz)")
            ax.set_ylabel("Time (s)")
        
        # Plot bandpass
        if opts.bandpass:
            ax.plot(frequency, np.log10(np.sum(data[:,:,i], axis=0)))
            ax.xaxis.set_major_formatter( pylab.FormatStrFormatter('%2.2f'))
            ax.set_xlabel("Channel (MHz)")
            ax.set_ylabel("Log Power (Arbitrary)")
            ax.set_xlim((frequency[0], frequency[-1]))

        # Plot time series
        if opts.time:
            ax.plot(time, np.sum(data[:,:,i] / nchans, axis=1))
            ax.xaxis.set_major_formatter(pylab.FormatStrFormatter('%2.1f'))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power (Arbitrary)")
            ax.set_xlim((time[0], time[-1]))

    fig.tight_layout()
    plt.show()
    f.close()

def transpose(opts):
    """ Transpose data so it can be viewed with the Qt plotter 
        Each beam is saved to a different file """

    print "Generating beam files"

    nbeams = opts.nbeams
    nchans = opts.nchans

    # Open files
    f = open(args[0], 'rb')

    # Create file per beam
    basename = os.path.basename(args[0]).split('.')[0]
    files = [open(os.path.join(os.path.dirname(args[0]), "%s_beam%d.dat" % (basename, i)), 'wb') for i in range(nbeams) ]

    # Get file size to calculate number of time spectra
    filesize = os.path.getsize(args[0])    
    nsamp = filesize / (4 * nchans * nbeams)

    # Read each time spectrum separately
    for i in range(nsamp):
        data = f.read(nchans * nbeams * 4)
        data = np.array(struct.unpack('f' * nchans * nbeams, data))
        data = np.reshape(data, (nchans, nbeams)).T

        # Write to respective file
        for j in range(nbeams):
            files[j].write(struct.pack('f'*nchans, *data[j,:]))

        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i, nsamp, (i / float(nsamp) * 100)))
        sys.stdout.flush()

    # Close files
    f.close()
    [f.close for f in files]

def integrate(opts):
    """ Integrate samples """
    
    print "Time samples:       %d" % opts.samples
    print "===== Genberating time series ====="

    for filename in glob.glob(args[0]):

        print "===== Processing %s =====" % filename

        # Open file
        f = open(filename, 'rb')

        # Create output file
        basename = os.path.basename(filename).split('.')[0]
        w = open(os.path.join(os.path.dirname(args[0]), "%s_timeseries_%d.dat" % (basename, opts.samples)), 'wb')

        # Get file size to calculate number of time spectra
        filesize = os.path.getsize(filename)    
        iterations = filesize / (4 * nchans * opts.samples)

        # Process file
        for i in range(iterations):
            data = f.read(nchans * opts.samples * 4)
            nsamp = len(data) / (nchans * 4)
            data = np.array(struct.unpack('f' * nchans * nsamp, data), dtype=float)
            data = np.reshape(data, (nsamp, nchans))
            w.write(struct.pack('f', np.sum(np.sum(data, axis = 1) / nchans) / nsamp))

            sys.stdout.write("===== Processing %d of %d [%.2f%%]   \r" % (i, iterations, (i / float(iterations) * 100)))
            sys.stdout.flush()

        # Close files
        f.close()
        w.flush()
        w.close()

if __name__ == "__main__":

    p = OptionParser()
    p.set_usage('process_beams.py [options] INPUT_FILE')
    p.set_description(__doc__)

    p.add_option('-f', '--bandpass', dest='bandpass', action='store_true', default=False,
        help='Show bandpass plots')
    p.add_option('-w', '--waterfall', dest='waterfall', action='store_true', default=False,
        help='Show beam waterfall plots')
    p.add_option('-t', '--time', dest='time', action='store_true', default=False,
        help='Show power time series')
    p.add_option('-p', '--process', dest='process', action='store_true', default=False,
        help='Generate beam file for future processing. Default option')
    p.add_option('-i', '--integrate', dest='integrate', action='store_true', default=False,
        help='Generate integrated time series. File must already have been processed. Default integration length 128')
    p.add_option('-s', '--samples', dest='samples', type='int', default=128,
        help='Number of samples to integrate. Default value 128')
    p.add_option('-c', '--nchans', dest='nchans', type='int', default=1024,
        help='Number of frequency channels. Default 1024')
    p.add_option('-b', '--nbeams', dest='nbeams', type='int', default=4,
        help='Number of beams. Default 4')
    p.add_option('', '--beam', dest='beam', type='int', default=-1,
        help='Select beam to be processed. Default -1 (process all beams)')
    p.add_option('', '--fch1', dest='fch1', type='float', default=418,
        help='Channel 0 frequency in MHz. Default 418')
    p.add_option('', '--foff', dest='foff', type='float', default=-19531.25,
        help='Channel bandwidth in Hz. Default -19531.25')
    p.add_option('', '--tsamp', dest='tsamp', type='float', default=0.0000512,
        help='Sampling time in seconds. Default 0.0000512')

    opts, args = p.parse_args(sys.argv[1:])

    print
    print "Number of channels:    %d"       % opts.nchans
    print "Number of beams:       %d"       % opts.nbeams
    print "Frequency Resolution:  %.2f Hz"  % abs(opts.foff)
    print "Temporal Resolution:   %.2f s"   % opts.tsamp
    print "Top Frequency:         %.2f MHz" % opts.fch1
    print

    if args==[]:
        print 'Please specify an input file! \nExiting.'
        exit()

    if opts.process:
        transpose(opts)
    elif opts.integrate:
        integrate(opts)
    else:
        plot(opts)

    
