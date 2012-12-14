import struct, numpy as np
from matplotlib import pyplot as plt

ndms = 768

if __name__ == "__main__":
    f = open('hist_output.dat','rb'); 
    d = f.read();  
    d = np.array(struct.unpack('f' * (len(d)/4), d))
    
    clusters = []
    for i in range(len(d) / ndms - 1):
        clusters.append(np.array(d[ndms * i: ndms * (i + 1)]))

    minimum = 1
    for i in range(min(len(clusters), minimum)):
        index = i + 5
        coeffs = np.polyfit(np.arange(0, ndms), clusters[index], 4)
        p = np.poly1d(coeffs)
        plt.plot([p(x) for x in np.arange(0, ndms)])
        plt.plot(clusters[index])

    plt.show()
