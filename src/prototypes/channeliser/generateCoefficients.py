#!/usr/bin/python2.7

# tut5_gencoeff.py
# CASPER Workshop 2011 Tutorial 5: Heterogeneous Instrumentation
#   Generate PFB filter coefficients. The filter coefficients array will
#   contain duplicates for optimised reading from the GPU.
#
# Created by Jayanth Chennamangalam based on code by Sean McHugh, UCSB

import sys
import getopt
import math
import numpy
import matplotlib.pyplot as plotter
import struct, cmath

# function definitions
def PrintUsage(ProgName):
    "Prints usage information."
    print "Usage: " + ProgName + " [options]"
    print "    -h  --help                 Display this usage information"
    print "    -n  --nfft <value>         Number of points in FFT"
    print "    -t  --taps <value>         Number of taps in PFB"
    print "    -b  --sub-bands <value>    Number of sub-bands in data"
    print "    -p  --no-plot              Do not plot coefficients"
    return

# default values
NFFT      = 32768       # number of points in FFT
NTaps     = 8           # number of taps in PFB
NSubBands = 1           # number of sub-bands in data
Plot      = True        # plot flag

# get the command line arguments
ProgName = sys.argv[0]
OptsShort = "hn:t:b:d:p"
OptsLong = ["help", "nfft=", "taps=", "sub-bands=", "data-type=", "no-plot"]

# check if the minimum expected number of arguments has been passed
# to the program
if (1 == len(sys.argv)):
    sys.stderr.write("ERROR: No arguments passed to the program!\n")
    PrintUsage(ProgName)
    sys.exit(1)

# Get the arguments using the getopt module
try:
    (Opts, Args) = getopt.getopt(sys.argv[1:], OptsShort, OptsLong)
except getopt.GetoptError, ErrMsg:
    sys.stderr.write("ERROR: " + str(ErrMsg) + "!\n")
    PrintUsage(ProgName)
    sys.exit(1)

# parse the arguments
for o, a in Opts:
    if o in ("-h", "--help"):
        PrintUsage(ProgName)
        sys.exit()
    elif o in ("-n", "--nfft"):
        NFFT = int(a)
    elif o in ("-t", "--taps"):
        NTaps = int(a)
    elif o in ("-b", "--sub-bands"):
        NSubBands = int(a)
    elif o in ("-d", "--data-type"):
        DataType = a
    elif o in ("-p", "--no-plot"):
        Plot = False
    else:
        PrintUsage(ProgName)
        sys.exit(1)

# Total number of coefficients
M = NTaps * NFFT

# the filter-coefficient-generation section
X = numpy.array([(float(i) / NFFT) - (float(NTaps) / 2) for i in range(M)])
PFBCoeff = numpy.sinc(X) * numpy.hanning(M)

# write the coefficients to disk and also plot it
FileCoeff = open("coeff_" + str(NTaps) + "_" + str(NFFT)  + ".dat", "wb")
FileCoeff.write(struct.pack("f" * NTaps * NFFT, *PFBCoeff))
FileCoeff.close()

# Plot the coefficients if required
if (Plot):
    plotter.plot(PFBCoeff)
    plotter.show()

