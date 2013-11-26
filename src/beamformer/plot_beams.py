from matplotlib import pyplot as plt
from optparse import OptionParser
import numpy as np, struct
import sys, math, os, sys
import glob, re, operator

if __name__ == "__main__":

    p = OptionParser()
    p.set_usage('process_beams.py [options] FILE_PATTERN')
    p.set_description(__doc__)

    p.add_option('-d', '--dec', dest='dec', type='float', default=0,
        help='Declination of first beam. Default value 0')
    p.add_option('', '--delta_dec', dest='delta_dec', type='float', default=1,
        help='Declination difference between beams. Default value 1')
    p.add_option('-n', '--name', dest='name', type='string', default="Cygnus",
        help='Source name. Default Cygnus')
    p.add_option('-c', '--center', dest='center', type='int', default=0,
        help='Central beam (or beam on source). Default value 0')
    p.add_option('-t', '--tsamp', dest='tsamp', type='float', default=51.2e-6,
        help='Sampling Time. Default value 51.2e-6')

    opts, args = p.parse_args(sys.argv[1:])

    # Get filenames
    files = {}
    for f in glob.glob(sys.argv[1]):
        number = int(re.match(".*beam(?P<number>\d+)_.*", f).groupdict()['number'])
        files[number] = f

    # Plot all the files
    for k, v in files.iteritems():
        data = open(v, 'rb').read()
        data = np.array(struct.unpack("f" * (len(data) / 4), data))
        x = [ (0 + i * opts.tsamp) / 60.0 for i in range(len(data))]
        plt.plot(x, data, label="Beam %d [DEC = %.2f]" % (k, opts.dec + opts.delta_dec * k))

    plt.title("%s transit (Beam %d on source)" % (opts.name, opts.center))
    plt.xlabel("Time since start of observation (minutes)")
    plt.ylabel("Arbitrary power")
#    plt.legend()
    plt.show()
