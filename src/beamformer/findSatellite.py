from matplotlib import pyplot as plt
import numpy as np, struct
from math import ceil, sqrt
import sys, glob, re

if len(sys.argv) < 2:
    print "Need input file"

nchans = 15616
start  = 8756
nsamp  = 15556 - 8756


# Get filenames
files = {}
for f in glob.glob(sys.argv[1]):
    number = int(re.match(".*beam(?P<number>\d+).dat", f).groupdict()['number'])
    files[number] = f

# Plot all the files
rows = ceil(sqrt(len(files)))
fig = plt.figure(figsize=(8,8))
for k, v in files.iteritems():

    print "Processing %d of %d \r" % (k + 1, len(files))

    f = open(v, 'rb')
    f.seek(start * nchans * 4)
    data = f.read(nsamp * nchans * 4)
    data = np.array(struct.unpack('f' * nsamp * nchans, data))
    data = np.reshape(data, (nsamp, nchans))

    data = np.log10(data[3000:4400:,5300:5800])
    data[:,180:230] = np.ones((1400,50)) + 1

    ax = fig.add_subplot(rows, rows, k + 1)
    ax.imshow(data, origin='lower', aspect='auto')
    ax.set_title("Beam %d" % k)
    
fig.tight_layout()
plt.show()

