import pyinotify
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab
from sys import argv
import os, fnmatch, sys, time, logging
import numpy as np

args = { } 

def normalPlot():
    """ Generate normal plots """    

    # Get filenames
    files = [item for item in os.listdir(argv[1]) if fnmatch.fnmatch(item, argv[2])]
    
    # Load data from all the files
    alldata = np.zeros((0,5))
    for f in files: alldata = np.concatenate((alldata, np.loadtxt(argv[1] + '/' + f, dtype=float, delimiter=',')))

    if len(alldata) == 0:
        print "File is empty"
        exit(0)

    # Start figure
    fig = pylab.figure()
    fig.subplots_adjust(left = 0.2, wspace = 0.2)

    # DM vs Time plot
    ax1 = fig.add_subplot(224) 
    ax2 = fig.add_subplot(223)   
    ax3 = fig.add_subplot(211)

    # Plot noise
    ax1.plot(alldata[alldata[:,3] == -1][:,0], alldata[alldata[:,3] == -1][:,1], 'r.', markersize=1)

    # Plot clusters caused due to RFI
    ax1.plot(alldata[alldata[:,3] == -2][:,0], alldata[alldata[:,3] == -2][:,1], 'b.', markersize=1)
    
    # Extract all valid clusters from data
    data = alldata[np.where(alldata[:,3] >= 0)]
    
    # Get number of unique clusters
    clusters = np.unique(data[:,3])

    print clusters
    # Plot all clusters separately
    colours = ['k+', 'g+', 'c+', 'm+', 'y+']
    for i in range(1, len(clusters) + 1):
            clusterId = clusters[i - 1]
            ax1.plot(data[data[:,3] == clusterId][:,0], data[data[:,3] == clusterId][:,1], colours[int((i-1) % len(colours))])    
            ax2.plot(data[data[:,3] == clusterId][:,1], data[data[:,3] == clusterId][:,2], colours[int((i-1) % len(colours))])
            ax3.plot(data[data[:,3] == clusterId][:,0], data[data[:,3] == clusterId][:,2], colours[int((i-1) % len(colours))])

    ax1.grid(True)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('DM')
    ax1.set_title('DM vs Time plot')

    # SNR vs DM plot
    ax2.grid(True)
    ax2.set_xlabel('DM')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Intensity vs DM plot')

    # SNR vs Time plot
    ax3.grid(True)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Intensity vs Time plot')


    if len(argv) == 5:
        plt.savefig(argv[4])
    else:
        pylab.show()

def templatePlot():
    """ Generate intensity plots """

    # Get filenames
    files = [item for item in os.listdir(argv[1]) if fnmatch.fnmatch(item, argv[2])]
    
    # Load data from all the files
    alldata = np.zeros((0,5))
    for f in files: alldata = np.concatenate((alldata, np.loadtxt(argv[1] + '/' + f, dtype=float, delimiter=',')))

    if len(alldata) == 0:
        print "File is empty"
        exit(0)

    # Start figure
    fig = pylab.figure(figsize=(14, 12))

    # Generate axes
    ax1 = fig.add_subplot(211) 
    ax2 = fig.add_subplot(223)   
    ax3 = fig.add_subplot(224)

    # Find axes limit
    time_min = np.min(alldata[:,0])
    time_max = np.max(alldata[:,0])
    dm_min   = np.min(alldata[:,1])
    dm_max   = np.max(alldata[:,1])
    snr_min  = np.min(alldata[:,2])
    snr_max  = np.max(alldata[:,2])
    num_bins = int((snr_max - snr_min) * 10)

    # Check if we need to plot noise points as well
    if ("skip_noise" in args and not args['skip_noise']) or "skip_noise" not in args:

        # Plot noise
        time = alldata[alldata[:,3] == -1][:,0]
        dm   = alldata[alldata[:,3] == -1][:,1]
        snr  = alldata[alldata[:,3] == -1][:,2]
        ax1.scatter(time, dm, s = (snr - snr_min) ** 2, facecolors='none', edgecolors='r',  marker='o')
        a1, = ax2.plot(dm, snr, 'r+')
        ax3.hist(snr, bins = num_bins, color='r', log=True)

        # Plot RFI-induced clusters
        time = alldata[alldata[:,3] == -2][:,0]
        dm   = alldata[alldata[:,3] == -2][:,1]
        snr  = alldata[alldata[:,3] == -2][:,2]
        if (len(time) != 0):
            ax1.scatter(time, dm, s = (snr - snr_min) ** 2, facecolors='none', edgecolors='k', marker='o')
            a2, = ax2.plot(dm, snr, 'k+')
            ax3.hist(snr, bins = num_bins, color='k', log=True)
        else:
            a2, = ax2.plot([],[],'k+')

    # Extract all valid clusters from data
    data = alldata[np.where(alldata[:,3] >= 0)]
    clusters = np.unique(data[:,3])

    # Generate global histogram plot for all clusters
    snr = data[:,2]
    if len(snr) != 0:
        ax3.hist(snr, bins = num_bins, color='b',log=True)

    # Plot the rest of the clusters
    for i in range(1, len(clusters) + 1):

        # Get cluster data points
        clusterId = clusters[i - 1]
        time = data[data[:,3] == clusterId][:,0]
        dm   = data[data[:,3] == clusterId][:,1]
        snr  = data[data[:,3] == clusterId][:,2]

        # Update plots
        ax1.scatter(time, dm, s = (snr - snr_min) ** 2, facecolors='none', edgecolors='b', marker='+')
        a3, = ax2.plot(dm, snr, 'b+')

    # Set axis labels
    ax1.set_xlim([time_min, time_max])    
    ax1.set_ylim([dm_min, dm_max])
    ax1.set_title("Intensity plot in DM & Time space")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("DM")

    ax2.set_yscale('log', basey=2)
    ax2.set_xlim([dm_min, dm_max])
    ax2.set_ylim([snr_min, snr_max])
    ax2.set_title("DM vs SNR Plot")
    ax2.set_xlabel("DM")
    ax2.set_ylabel("SNR")

    if ("skip_noise" in args and not args['skip_noise']) or "skip_noise" not in args:
        try:
            ax2.legend([a1, a2, a3], ["Noise", "RFI-induced Clusters", "Valid Clusters"])
        except:
            ax2.legend([a1, a2], ["Noise", "RFI-induced Clusters"])

    ax3.set_title("Number of pulses vs SNR")
    ax3.set_xlabel("SNR")
    ax3.set_ylabel("Number of pulses")

    pylab.savefig("Plots/detections.png")
    pylab.show()


def get_args():
    """ Dump command line argument into a dict """
    for item in argv:
        ind = item.find('=')
        if ind > 0:
            args[item[:ind]] = eval(item[ind + 1:])

labels = ["DM", "Time", "Intensity", "Cluster"]

#NOTE params: outputPlotter.py basedir 'file_regex' x:y [opt=value]*
if __name__ == "__main__":

    # Get command-line arguments
    get_args()

    # Pre-process existing files
    if argv[3] == 'all':
        normalPlot()
    else:
        templatePlot()

