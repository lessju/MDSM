from matplotlib import pyplot as plt
from sys import argv
import numpy as np

f = open(argv[1],'r')
data = f.read().split('\n')

sample, dm, flux = [], [], []
for item in data:
    if len(item.split(',')) > 2:
        sample.append(float(item.split(',')[0]))
        dm.append(float(item.split(',')[1]))
        flux.append(float(item.split(',')[2]))

#plt.plot(sample, flux, 'b.')
#plt.show()

x = np.linspace(min(dm), max(dm), 100)
y = [0 for i in range(len(x))]
iflux = [0 for i in range(len(x))]

for d in range(len(dm)):
    for i in range(1, len(x)):
        if dm[d] >= x[i-1] and dm[d] < x[i]:
            iflux[i] += flux[d]
            y[i] = y[i] + 1

plt.plot(x, np.array(y) * np.array(iflux), 'b-')
plt.show()
