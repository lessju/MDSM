from matplotlib import pyplot as plt
import matplotlib.cm as cm
from math import ceil
import numpy as np
import sys

# Store global parameters
args = { 'samples'  : 256,         
         'fch1'     : 418,           # MHz
         'foff'     : 20 / 1024.0,   # MHz
         'tsamp'    : 1.0 / (20e6 / 1024.0), # s
         'startdm'  : 0,
         'dmstep'   : 0.5,
         'ndms'     : 1024,
         'nchans'   : 512}

if __name__ == "__main__":

    # Process command-line arguments
    for item in sys.argv[2:]:
        ind = item.find('=')
        if ind > 0:
            args[item[:ind]] = eval(item[ind + 1:])

    for k, v in args.iteritems():
        globals()[k] = v

    # Generate DM-independent shifts with provided parameters
    dedisp = lambda f1, f2, tsamp: 4148.741601 * (f1**-2 - f2**-2) / tsamp
    shifts = [dedisp(fch1 + diff, fch1, tsamp) 
             for diff in -np.arange(0, foff * nchans, foff)]
    maxshift = shifts[nchans - 1] * (startdm + ndms * dmstep)

    # Generate empty memory grid
    grid = np.zeros((nchans, samples + maxshift))

    # Populate grid for each samples in pulse, for each DM value
    for s in range(samples):
        for d in range(ndms):
            dm = startdm + d * dmstep
            for c in range(nchans):
                grid[c, s + round(shifts[c] * dm)] += 1

    # Plot grid
    plt.imshow(grid, aspect='auto')
    plt.show()
