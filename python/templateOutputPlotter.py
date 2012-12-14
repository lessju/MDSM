from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv
import os, fnmatch
import numpy as np

def get_args():
    """ Dump command line argument into a dict """
    args = { }
    for item in argv:
        ind = item.find('=')
        if ind > 0:
            args[item[:ind]] = eval(item[ind + 1:])
    return args

labels = ["DM", "Intensity", "Time", "Downfactor"]

#NOTE params: outputPlotter.py basedir 'file_regex' x:y [opt=value]*
if __name__ == "__main__":

    args = get_args()
    print argv

    # Get filenames
    files = [item for item in os.listdir(argv[1]) if fnmatch.fnmatch(item, argv[2])]
    
    # Load files and create datastructures
    data = ([],[],[],[])
    for f in files:
        f = open(argv[1] + '/' + f, 'r')
        d = f.read().split('\n')
        for line in d[:-1]:
            try:
                item = line.split(',')     
                data[0].append(float(item[0]))
                data[1].append(float(item[1]))
                data[2].append(float(item[2]))
                data[3].append(float(item[3]))
            except: 
                break

	# Do intensity plot
	intensity = np.array(data[1])
	plt.scatter(data[2], data[0], s = (intensity - np.min(intensity)) ** 2, facecolors='none', marker = 'o')
	plt.grid(True)
	plt.xlabel(labels[2])
	plt.ylabel(labels[0])
	plt.title("Template Search result plot")
	plt.show()
