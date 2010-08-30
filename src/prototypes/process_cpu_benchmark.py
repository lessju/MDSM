from matplotlib import pyplot as plt
from sys import argv
import re

# arg 1 = filename
# arg 2 = key=value
# arg 3 = key=value
if len(argv) < 3:
    print "Not enough arguments"
    exit(0)

# Extract arguments
filepath = argv[1]
key1, value1, key2, value2 = 0, 0, 0, 0
try:
    key1, values1 = argv[2].split('=')[0], argv[2].split('=')[1].split(',')
    key2, values2 = argv[3].split('=')[0], argv[3].split('=')[1].split(',')
    if key1 not in ['nsamp', 'nchans', 'tdms'] or key2 not in ['nsamp', 'nchans', 'tdms']:
        print "Invalid key/value parameters"
        exit(0)
    if len(values2) > 1 and len(values1) > 1:
        print "Only one key can have multiple values"
        exit(0)
except:
    print "Invalid parameters"
    exit(0)

# Get third argument
key3 = list(set(['nsamp', 'nchans', 'tdms']) - set([key1, key2]))[0]

f = open(argv[1], 'r')
fileData = f.read()
f.close()

# Plot lines
lines = ['k-', 'k--', 'k-.', 'k:']

# Plot for a given nsamp, tdms or nchans
if len(values1) > 1:
    value2 = values2[0]
    plt.title("Benchmark for %s = %s, variying %s" % (key2, value2, key3))
    for value1 in values1:
        matchedIter = re.finditer('nsamp: (?P<nsamp>\d*), nchans: (?P<nchans>\d*),.*tdms: (?P<tdms>\d*),.*\nTime: (?P<time>\d*)', fileData)
        x, y = [], []
        for item in matchedIter:
            vals = item.groupdict()
            if vals[key1] == value1 and vals[key2] == value2:
               x.append(vals[key3])
               y.append(vals['time']) 
        plt.plot(x, y, lines[values1.index(value1)], label='%s = %s' % (key1, value1))
else:
    value1 = values1[0]
    plt.title("Benchmark for %s = %s, variying %s" % (key1, value1, key3))    
    for value2 in values2:
        matchedIter = re.finditer('nsamp: (?P<nsamp>\d*), nchans: (?P<nchans>\d*),.*tdms: (?P<tdms>\d*),.*\nTime: (?P<time>\d*)', fileData)     
        x, y = [], []
        for item in matchedIter:
            vals = item.groupdict()
            if vals[key1] == value1 and vals[key2] == value2:
               x.append(vals[key3])
               y.append(vals['time']) 

        plt.plot(x, y, lines[values2.index(value2)], label='%s = %s' % (key2, value2))

# Create plot
plt.grid(True)
plt.xlabel(key3)
plt.ylabel("Time (s)")
plt.legend(loc="upper left")
plt.show()
