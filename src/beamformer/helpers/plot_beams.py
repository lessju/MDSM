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
#    files = {}
#    for f in glob.glob(sys.argv[1]):
#        number = int(re.match(".*beam(?P<number>\d+).*", f).groupdict()['number'])
#        files[number] = f

    files  = {
#    0 : "/data/2014/14/CasA/CasA_beam0.dat",
#    1 : "/data/2014/14/CasA/CasA_beam1.dat",
#    2 : "/data/2014/14/CasA/CasA_beam2.dat",
#    3 : "/data/2014/14/CasA/CasA_beam3.dat",
#    4 : "/data/2014/14/CasA/CasA_beam4.dat",
#    5 : "/data/2014/14/CasA/CasA_beam5.dat",
#    6 : "/data/2014/14/CasA/CasA_beam6.dat",
#    7 : "/data/2014/14/CasA/CasA_beam7.dat",
#    8 : "/data/2014/14/CasA/CasA_beam8.dat",
#    9 : "/data/2014/14/CasA/CasA_beam9.dat",
    10 : "/data/2014/14/CasA/CasA_beam10.dat",
    11 : "/data/2014/14/CasA/CasA_beam11.dat" ,
    12 : "/data/2014/14/CasA/CasA_beam12.dat",
    13 : "/data/2014/14/CasA/CasA_beam13.dat",
    14 : "/data/2014/14/CasA/CasA_beam14.dat",
#    15 : "/data/2014/14/CasA/CasA_beam15.dat",
#    16 : "/data/2014/14/CasA/CasA_beam16.dat",
#    17 : "/data/2014/14/CasA/CasA_beam17.dat",
#    18 : "/data/2014/14/CasA/CasA_beam18.dat",
#    19 : "/data/2014/14/CasA/CasA_beam19.dat",
#    20 : "/data/2014/14/CasA/CasA_beam20.dat",
#    21 : "/data/2014/14/CasA/CasA_beam21.dat",
#    22 : "/data/2014/14/CasA/CasA_beam22.dat",
#    23 : "/data/2014/14/CasA/CasA_beam23.dat",
#    24 : "/data/2014/14/CasA/CasA_beam24.dat",
#    25 : "/data/2014/14/CasA/CasA_beam25.dat",
#    26 : "/data/2014/14/CasA/CasA_beam26.dat",
#    27 : "/data/2014/14/CasA/CasA_beam27.dat",
#    28 : "/data/2014/14/CasA/CasA_beam28.dat",
#    29 : "/data/2014/14/CasA/CasA_beam29.dat",
#    30 : "/data/2014/14/CasA/CasA_beam30.dat",
#    31 : "/data/2014/14/CasA/CasA_beam31.dat",
    }


    # Plot all the files
    for k, v in files.iteritems():
        data = open(v, 'rb').read()
        data = np.log10(np.array(struct.unpack("f" * (len(data) / 4), data)))
        x = [ (0 + i * opts.tsamp) / 60.0 for i in range(len(data))]
        plt.plot(x, data, label="Beam %d" % k)

    plt.legend()
    plt.title("%s transit (Beam %d on source)" % (opts.name, opts.center))
    plt.xlabel("Time since start of observation (minutes)")
    plt.ylabel("Arbitrary power")
    plt.show()
