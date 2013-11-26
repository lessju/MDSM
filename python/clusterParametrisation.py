import pyinotify
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rc
import pylab
from sys import argv
import os, fnmatch, sys, time, logging
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from scipy.special import erf
from scipy.interpolate import interp1d

startDM = 0
numDMs  = 2048
dmStep  = 0.04

bw   = 20
freq = 408e-3

minWidth = 0.01

def smoothListGaussian(array, degree=5):  
    """ Gaussian line smoothing """

    window=degree*2-1  
    weight=np.array([1.0]*window)  
    weightGauss=[]  

    for i in range(window):  
        i=i-degree+1  
        frac=i/float(window)  
        gauss=1/(np.exp((4*(frac))**2))  
        weightGauss.append(gauss)  

    weight=np.array(weightGauss)*weight  
    smoothed=[0.0]*(len(array)-window)  

    for i in range(len(smoothed)):  
        smoothed[i]=sum(np.array(array[i:i+window])*weight)/sum(weight)  

    return smoothed 

def processFiles():

    # Get filenames
    files = [item for item in os.listdir(argv[1]) if fnmatch.fnmatch(item, argv[2])]
    
    # Load data from all the files
    alldata = np.zeros((0,5))
    for f in files: alldata = np.concatenate((alldata, np.loadtxt(argv[1] + '/' + f, dtype=float, delimiter=',')))
    
    # Extract all valid clusters from data
    data = alldata[np.where(alldata[:,3] != -1)]
    
    # Get number of unique clusters
    clusters = np.unique(data[:,3])
    
    # Loop over all clusters
    for i in clusters:

        if i == -2: continue

        # Initialise font
        rc('text', usetex=True)
        rc('font', family='serif')
        rc('xtick', labelsize=12)
        rc('ytick', labelsize=12)
        rc('axes', labelsize=14)

        # Initialise figure and canvas
        fig = plt.figure()
        ax = fig.add_subplot(111)

        cluster = data[data[:,3] == i][:,0:3]

        # Create SNR-DM curve
        dmHist = np.zeros(numDMs)
        for (time, dm, snr) in cluster:
            dmHist[int(round((dm - startDM) / dmStep))] += snr

        # Smoothen curve
        p1, = ax.plot(dmHist / max(dmHist), '-r')

        dmHistSmooth = np.zeros(numDMs)
        for j in range(1, numDMs-1):
            dmHistSmooth[j] = (dmHist[j-1] + dmHist[j] + dmHist[j+1])/3.0
        dmHistSmooth[0] = dmHist[0]
        dmHistSmooth[numDMs-1] = dmHist[numDMs-1]
        dmHist = dmHistSmooth

        p2, = ax.plot(dmHist / max(dmHist), '-b')
        
        # Find maximum DM 
        maxDM = 0
        for j, v in enumerate(dmHist):
            if dmHist[maxDM] < v:
                maxDM = j
        maxDM = maxDM * dmStep + startDM

        # Sanity check on DM
#        if maxDM <= 2.0:
#            print "Cluster caused due to RFI"
#            continue

        # Find max SNR for given DM
        maxSNR = 0
        for (time, dm, snr) in cluster:
            if np.abs(dm - maxDM) < dmStep * 0.5:
                maxSNR = maxSNR if snr < maxSNR else snr

        # Find pulse width
        minT, maxT = 9e25, 0
        for (time, dm, snr) in cluster:
            if np.abs(dm - maxDM) < dmStep * 0.5:
                minT = minT if time > minT else time
                maxT = maxT if time < maxT else time
        width = (maxT - minT) * 1e3 * 0.4

        # Sanity check on pulse width
        if width and width <= minWidth:
            print "Invalid cluster: pulse width is %f" % width
            continue

        print "maxDM = %f, maxSNR = %f,  minT = %f, maxT = %f, width = %f ms" % (maxDM, maxSNR, minT, maxT, width)

        # Compute curve for incorrect de-dispersion
        dmrange     = np.arange(startDM - maxDM, numDMs * dmStep + startDM - maxDM, dmStep)
        snrFunction = 6.91e-3 * dmrange * (bw / (freq ** 3 * width))   
        snrFunction = np.sqrt(np.pi) * 0.5 * (1.0 / snrFunction) * erf(snrFunction)
        p3, = ax.plot(snrFunction, '--g')

        # Check if snrFunction is valid
        mse = 0
        if (abs(max(snrFunction) - 1.0) < 0.1):

            # Calculate MSE between the two
            dmHist = dmHist / max(dmHist)

            res = 0
            for j in range(len(dmHist)):
                res += (dmHist[j] - snrFunction[j])**2
            mse = res / len(dmHist)
            print "MSE: %f" % mse
        else:
            print "Invalid cluster"

        # Update canvas
        plt.draw()        
        
        labels = [float(item.get_text().replace("$", "")) * dmStep + startDM for item in ax.get_xticklabels()]
        labels = ["$%.2f$" % item for item in labels ]
        ax.set_xticklabels(labels)
        ax.set_xlim([0, numDMs])
        ax.set_xticklabels(labels)
        ax.set_xlabel(r"DM ($pc\ cm^{-3}$)")
        ax.set_ylabel(r"Normalised SNR")

        # Add text box to plot
        color = 'r' if mse > 0.05 or mse == 0 or maxDM <= 2.0 else 'g'
        
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.6, color=color)
        ax.text(0.49, 0.75, "Max DM = %.2f, MSE = %.5f" % (maxDM, mse), transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        plt.legend([p1, p2, p3], ["SNR-DM curve", "Smoothed SNR-DM Curve", "Analytical model"])
        pylab.savefig("Plots/%d.png" % i)


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

