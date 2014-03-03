####################################################
# This has to:
# - Take a directory as an input file
# - Process all the files inside that directory
# - Given an output directory, place processed file in that directory
# - Run the code
# - Generate plots and place in output directory (separate and joined)
####################################################

from matplotlib import pyplot as plt
import numpy as np, pylab
import subprocess as sp
import sys, os, re
import fnmatch

# Define a global directory to store all XML configuration values
pipelineConfig = {
    'lowDM'                     : 0,
    'numDMs'                    : 8192,
    'dmStep'                    : 0.1,
    'nchans'                    : 2048,
    'npols'                     : 1,
    'ncoeffs'                   : 12,
    'tsamp'                     : 0.0003276,
    'nsamp'                     : 65536,
    'nbits'                     : 8,
    'voltage'                   : 0,
    'applyRFIClipper'           : 0,
    'channelBlock'              : 1024,
    'spectrumThreshold'         : 7,
    'channelThreshold'          : 4,
    'detectionThreshold'        : 4,
    'applyMedianFilter'         : 0,
    'applyDetrending'           : 0,
    'enableTBB'                 : 0,
    'applyClustering'           : 1,
    'clusteringMinPoints'       : 15,
    'clusteringTimeRange'       : 40,
    'clusteringDmRange'         : 4,
    'clusteringSnrRange'        : 4,
    'clusteringMinPulseWidth'   : 0.01,
    'applyClassification'       : 1,
    'writeToFile'               : 0,
    'outputBits'                : 32,
    'compression'               : 255,
    'outputFilePrefix'          : "GBNCC_Test",
    'outputBaseDirectory'       : ".",
    'outputSingleFileMode'      : "1",
    'outputSecondsPerFile'      : "300",
    'outputUsePcTime'           : "1",
    'gpuIDs'                    : '0',
    'beams' : [ {'id'              : 0, 
                 "topFrequency"    : 399.951172, 
                 "frequencyOffset" : -0.048828, 
                 "ra"              : 0, 
                 "dec"             : 0, 
                 "ha"              : 0 }]
}

configHelp = {
    'lowDM'                     : "Lowest DM value (float)",
    'numDMs'                    : "Number of DM values to process (unsigned int)",
    'dmStep'                    : "Difference between consecutive DMs (float)",
    'nchans'                    : "Number of frequency channels (unsigned int)",
    'npols'                     : "Number of polarisations (currently 1 supported)",
    'ncoeffs'                   : "Number of coefficients for polynomical bandpass fitting (unsigned int)",
    'tsamp'                     : "Sampling time (float)",
    'nsamp'                     : "Number of samples to process in one GPU iteration (unsigned int)",
    'nbits'                     : "Number of bits per sample in input data (unsigned int)",
    'voltage'                   : "Input data can be either voltage (1) or power (0)",
    'applyRFIClipper'           : "Turn on RFI Clipper (1)",
    'channelBlock'              : "Channel thresholder window block length (unsigned int, ideally power of 2)",
    'spectrumThreshold'         : "Spectrum clipper threshold (float)",
    'channelThreshold'          : "Channel clipper threshold (float)",
    'detectionThreshold'        : "Output detection thresold (float)",
    'applyMedianFilter'         : "Turn on median filtering (1)",
    'applyDetrending'           : "Turn on detrending (1)",
    'enableTBB'                 : "Enable Transient Buffer Board Mode (1)",
    'applyClustering'           : "Turn on DBSCAN clutering (1)",
    'clusteringMinPoints'       : "Minimum number of points in a cluser (unsigned int)",
    'clusteringTimeRange'       : "Clustering eps-value for time dimension (int)",
    'clusteringDmRange'         : "Clustering eps-value for DM dimension (int)",
    'clusteringSnrRange'        : "Clustering eps-value for SNR dimension (int)",
    'clusteringMinPulseWidth'   : "Minimum pulse width for cluster classification (float)",
    'applyClassification'       : "Turn on cluster classification (1)",
    'writeToFile'               : "Output raw input data to file (1)",
    'outputBits'                : "Output bits (currently 32 or 8 for power data, 16 or 4 for voltages)",
    'compression'               : "Mu-law compression factor (1-255)",
    'outputFilePrefix'          : "Output file prefix (string)",
    'outputBaseDirectory'       : "Output base directory (string)",
    'outputSingleFileMode'      : "Results output to a single file (1)",
    'outputSecondsPerFile'      : "Number of seconds per file for split file mode (int)",
    'outputUsePcTime'           : "Use PC time for filename (1)",
    'gpuIDs'                    : 'Comma seperated list of GPUs to use',
    'beams' : [ {'id'              : 0, 
                 "topFrequency"    : 399.951172, 
                 "frequencyOffset" : -0.048828, 
                 "ra"              : 0, 
                 "dec"             : 0, 
                 "ha"              : 0 }]
}

def displayConfig():
    """ Print config to screen""" 
    
    print "\nPipeline Configuration"
    print "----------------------\n"

    for k in sorted(pipelineConfig.iterkeys()):
        v = pipelineConfig[k]
        if k != "beams":
            print "\t%s%s: %s" % (k, ' ' * (26 - len(k)), v)

    print "\nBeam Parameters"
    print "----------------\n"
    
    for beam in pipelineConfig['beams']:
        print "Beam %d" % beam['id']
        for k, v in beam.iteritems():
            if k != 'id':
                print "  %s%s: %s" % (k, ' ' * (26 - len(k)), v)
        print 

def displayParameterHelp():
    """ Print config to screen""" 
    
    print "\nPipeline Configuration Parameters"
    print "----------------------------------\n"

    for k in sorted(configHelp.iterkeys()):
        v = configHelp[k]
        if k != "beams":
            print "\t%s%s: %s" % (k, ' ' * (26 - len(k)), v)


def explainParameter(key):
    """ Print help text for key """
    
    # If key is a valid parameter display associated help text
    if key in pipelineConfig.keys():
        print "%s%s: %s" % (key, ' ' * (26 - len(key)), configHelp[key])
    else:
        print "Invalid parameter %s" % key
        

def generateXMLFile(filename):
    """ Use template XML file and parameters in pipelineConfig
        to generate configuration file"""

    # Read template config
    f = open("configTemplate.xml", "r")
    template = f.read()
    f.close()

    # Loop over all parameters (except beams) and replace values
    # with ones provided in config dictionary
    for k, v in pipelineConfig.iteritems():
        if k != 'beams':
            template = re.sub('%s=""' % k, '%s="%s"' % (k, v), template)

    # Handle beams
    beams = pipelineConfig['beams']
    beamConfig = "<beams>\n"
    for beam in beams:
        beamConfig = beamConfig + '\t\t<beam beamId="%d" topFrequency="%f" frequencyOffset="%f" />\n' % (beam['id'], beam['topFrequency'], beam['frequencyOffset'])
    beamConfig = beamConfig + "\t</beams>"
    
    # Insert beam config in output XML file
    template = re.sub('<beams/>', beamConfig, template)

    # Write generated config to disk
    f = open(filename, "w")
    f.write(template)   
    f.close()      

def generatePlot(directory, filePattern, skipNoise=False):
    """ Generate intensity plots """

    # Get filenames
    files = [item for item in os.listdir(directory) if fnmatch.fnmatch(item, filePattern)]

    # Load data from all the files
    alldata = np.zeros((0,5))
    for f in files: alldata = np.concatenate((alldata, np.loadtxt(os.path.join(directory, f), dtype=float, delimiter=',')))

    if len(alldata) == 0:
        print "Could not generate plot for %s. File is empty" % os.path.join(directory, filePattern)
        return

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
    if not skipNoise:

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

    # Insert legend
    if not skipNoise:
        try:
            ax2.legend([a1, a2, a3], ["Noise", "RFI-induced Clusters", "Valid Clusters"])
        except:
            ax2.legend([a1, a2], ["Noise", "RFI-induced Clusters"])

    # Finalise plot
    ax3.set_title("Number of pulses vs SNR")
    ax3.set_xlabel("SNR")
    ax3.set_ylabel("Number of pulses")

    # Generate plot
    pylab.savefig(os.path.join(directory, "detections.png"))
  

def runPipeline(inputFile, pipelinePath, makePlot=False, skipNoise=True):
    """ Run pipeline """

    # Check if input file exists
    if not os.path.isfile(inputFile):
        print "Invalid input file"
        return

    # Extract paths from configuration and parameters
    outputDir = pipelineConfig['outputBaseDirectory']
    filePrefix = '.'.join(inputFile.split('/')[-1].split('.')[:-1])

    # Avoid nasty surprises
    if outputDir == '.':
        outputDir = os.getcwd()
    elif outputDir == '..':
        outputDir = '/'.join(os.getcwd().split('/')[:-1])
    elif outputDir[0] == '~':
        outputDir = os.path.expanduser(outputDir)
    pipelineConfig['outputBaseDirectory'] = outputDir
    
    # Update file prefix in configuration
    pipelineConfig['outputFilePrefix'] = filePrefix

    # Check if directory exists, if not create it
    try:
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
    except:
        print "Couldn't create output directory %s" % outputDir
        return

    # Save config file in output directory
    configFile = os.path.join(outputDir, "%s_config.xml" % filePrefix)
    generateXMLFile(configFile)

    # Run pipeline
    print "Processing %s" % inputFile
    p = sp.Popen([pipelinePath, inputFile, "-obs", configFile], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = p.communicate()

    # Write output and error log files to disk
    f = open(os.path.join(outputDir, "%s_log.txt" % filePrefix), "w")
    f.write(out)
    f.close()

    f = open(os.path.join(outputDir, "%s_err.txt" % filePrefix), "w")
    f.write(err)
    f.close()

    # Call plot generation script if required 
    if makePlot:
        print "Generating plot for %s" % inputFile
        generatePlot(outputDir, "*.dat", skipNoise=skipNoise)


# Script entry point if run from console
if __name__ == "__main__":

    # Get input file
    inputFile = "/home/lessju/GBNCC/GBNCC04743_55169.fil"  
    
    # Update pipeline config
    pipelineConfig['outputBaseDirectory'] = '/'.join(inputFile.split('/')[:-1])
    pipelineConfig['dmStep'] = 0.02 
    pipelineConfig['applyClassification'] = 0

    # Run pipeline and generate plot
    runPipeline(inputFile, "/home/lessju/Code/MDSM/build/standalone-mdsm", makePlot=True)
