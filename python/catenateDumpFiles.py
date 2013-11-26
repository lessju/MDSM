from optparse import OptionParser
import struct, sys, fnmatch, os
import numpy as np

from matplotlib import pyplot as plt

if __name__ == "__main__":

    # Parse command line options
    p = OptionParser()
    p.set_usage('CatenateDumpFiles.py [options] DIR FILE_REGEX')
    p.set_description(__doc__)
    p.add_option('-c', '--nchans', dest='nchans',action='store_true', default=512, 
        help='Number of frequency channels.  Default: 512')
    p.add_option('-b', '--nbits', dest='nbits',action='store_true', default=32, 
        help='Number of bits per value.  Default: 32')
    p.add_option('-y', '--nbeams', dest='nbeams',action='store_true', default=2, 
        help='Number of beams in the file.  Default: 2')
    p.add_option('-q', '--quantised', dest='quantised',action='store_true', default=0, 
        help='True if data is 8-bit quantised.  Default: 0')
    p.add_option('-v', '--verbose', dest='verbose',action='store_true', default=False, 
        help='Be verbose about errors.')

    opts, args = p.parse_args(sys.argv[1:])

    # Check if we have any input files
    if len(args) < 2:
        print 'Regular Expression describing filename to catenate is required'
        exit(0)
    else:
        file_dir   = args[0]
        file_regex = args[1]

    nchans = opts.nchans    
    nbits  = opts.nbits
    nbeams = 2 #opts.nbeams
    quantised = opts.quantised

    print nbeams, quantised, nbits, nchans

    # OK, list and sort dump file to catenate
    # TODO: We need to properly sort the dates in the files
    files = sorted([item for item in os.listdir(file_dir) if fnmatch.fnmatch(item, file_regex)])

    # Create output file
    output_files = [open(os.path.join(file_dir, "catenated_file_%d.dat" % i), "wb") for i in range(nbeams)]

    # Loop all files
    for item in files:

        print "Processing file %s" % item

        if quantised: # For quantised data

            # Read file content
            current_file = open(item, "rb")
            means = current_file.read(nchans * 4)
            stddevs = current_file.read(nchans * 4)
            data = current_file.read()
        
            # Unpack data format
            means = np.array(struct.unpack('f' * nchans, means))
            stddevs = np.array(struct.unpack('f' * nchans, stddevs))

            if (nbits == 32):
                data = np.array(struct.unpack('B' * len(data), data))

            # Convert shape
            nsamp = len(data) / nchans
            data = np.reshape(data, (nsamp, nchans))
            data = (data * stddevs) + means

            # Write to file in sample order
            for i in range(nsamp):
                for j in range(nchans):
                    output.write(struct.pack('f', data[i,j]))

            current_file.close()

        else:
             # Read file content
            current_file = open(os.path.join(file_dir, item), "rb")

            data = current_file.read()
        
            # Unpack data format
            if (nbits == 32):
                data = np.array(struct.unpack('f' * (len(data) / 4), data))

            # Convert shape
            nsamp = len(data) / nchans / nbeams
            data = np.reshape(data, (nbeams, nchans, nsamp))

            # Write to file in sample order
            for f in range(nbeams):
                for i in range(nsamp):
                    for j in range(nchans):
                        output_files[f].write(struct.pack('f', data[f, j, i]))

            current_file.close()
