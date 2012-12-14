import pyinotify
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab
from sys import argv
import os, fnmatch, sys, time, logging
import numpy as np

def processFiles():

    # Get filenames
    files = [item for item in os.listdir(argv[1]) if fnmatch.fnmatch(item, argv[2])]
    
    # Load data from all the files
    alldata = np.zeros((0,5))
    for f in files: alldata = np.concatenate((alldata, np.loadtxt(argv[1] + '/' + f, dtype=float, delimiter=',')))

    # Plot in 3D
#    fig = plt.figure()
#    ax = Axes3D(fig)

#    ax.scatter(alldata[alldata[:,3] == -1][:,0], alldata[alldata[:,3] == -1][:,1], alldata[alldata[:,3] == -1][:,2], c = 'r', marker='o')
#    data = alldata[np.where(alldata[:,3] != -1)]

#    colours = ['g', 'b', 'k', 'y', 'c']
#    clusters = np.unique(data[:,3])
#    print clusters
#    for i in clusters.tolist():
#        ax.scatter(data[data[:,3] == i][:,0], data[data[:,3] == i][:,1], data[data[:,3] == i][:,2], c = colours[int(i % len(colours))], marker='o')
#    plt.show()

    # Start figure
    fig = pylab.figure()
    fig.subplots_adjust(left = 0.2, wspace = 0.2)

    # DM vs Time plot
    ax1 = fig.add_subplot(224)   

    # Plot noise
    ax1.plot(alldata[alldata[:,3] == -1][:,0], alldata[alldata[:,3] == -1][:,1], 'r.', markersize=1)
    
    # Extract all valid clusters from data
    data = alldata[np.where(alldata[:,3] != -1)]
    
    # Get number of unique clusters
    clusters = np.unique(data[:,3])
    
    # Plot all clusters separately
    colours = ['b+', 'k+', 'g+', 'c+', 'm+', 'y+']
    for i in range(1, len(clusters) + 1):
        ax1.plot(data[data[:,3] == i][:,0], data[data[:,3] == i][:,1], colours[int((i-1) % len(colours))])
    
    ax1.grid(True)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('DM')
    ax1.set_title('DM vs Time plot')

    # SNR vs DM plot
    ax2 = fig.add_subplot(223)  
    ax2.plot(data[:,1], data[:,2], 'r+')
    ax2.grid(True)
    ax2.set_xlabel('DM')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Intensity vs DM plot')

    # SNR vs Time plot
    ax3 = fig.add_subplot(211)
    ax3.plot(data[:,0], data[:,2], 'r+')
    ax3.grid(True)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Intensity vs Time plot')

    pylab.show()


def get_args():
    """ Dump command line argument into a dict """
    args = { }
    for item in argv:
        ind = item.find('=')
        if ind > 0:
            args[item[:ind]] = eval(item[ind + 1:])
    return args


labels = ["DM", "Time", "Intensity", "Cluster"]

#NOTE params: outputPlotter.py basedir 'file_regex' x:y [opt=value]*
if __name__ == "__main__":

    # Get command-line arguments
    args = get_args()

    # Pre-process existing files
    processFiles()

