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
    
    # Load files and create datastructures
    data = ([],[],[])
    for f in files:
        f = open(argv[1] + '/' + f, 'r')
        d = f.read().split('\n')[:-1]
        for line in d[:-1]:
            try:
                item = line.split(',') 
                data[0].append(float(item[0]))
                data[1].append(float(item[1]))
                data[2].append(float(item[2]))
            except:
                break

    # Get dimensions to plot
    if len(argv) < 4:
        print "Not enough parameters"
        exit()
    else:
        ind = argv[3].find(':')

        # Single plot
        if ind > 0:
            pylab.cla()

            x_index = int(argv[3][:ind]) - 1
            y_index = int(argv[3][ind + 1:]) - 1

            pylab.plot(data[x_index], data[y_index], '+r')
            pylab.grid(True)
            pylab.xlabel(labels[x_index])
            pylab.ylabel(labels[y_index])
            pylab.title("%s vs %s plot" % (labels[y_index], labels[x_index]))
            pylab.show()

        # All plots
        elif argv[3] == 'all':

            fig = pylab.figure()
            fig.subplots_adjust(left=0.2, wspace=0.2)
            
            ax1 = fig.add_subplot(224)
            ax1.plot(data[0], data[1], 'r+')
            ax1.grid(True)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('DM')
            ax1.set_title('DM vs Time plot')

            ax2 = fig.add_subplot(223)  
            ax2.plot(data[1], data[2], 'r+')
            ax2.grid(True)
            ax2.set_xlabel('DM')
            ax2.set_ylabel('Intensity')
            ax2.set_title('Intensity vs DM plot')

            ax3 = fig.add_subplot(211)
            ax3.plot(data[0], data[2], 'r+')
            ax3.grid(True)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Intensity')
            ax3.set_title('Intensity vs Time plot')

            pylab.show()

        else:
            print "Invalid x/y values"
            exit()

class FileProcessor(pyinotify.ProcessEvent):

    def process_IN_CREATE(self, event):
        processFiles()
        print "Create: %s " % os.path.join(event.path, event.name)

    def process_IN_MODIFY(self, event):
        processFiles()
        print "Modified: %s " % os.path.join(event.path, event.name)

def get_args():
    """ Dump command line argument into a dict """
    args = { }
    for item in argv:
        ind = item.find('=')
        if ind > 0:
            args[item[:ind]] = eval(item[ind + 1:])
    return args

labels = ["DM", "Time", "Intensity"]

#NOTE params: outputPlotter.py basedir 'file_regex' x:y [opt=value]*
if __name__ == "__main__":

    # Get command-line arguments
    args = get_args()

    # Enable matplotlib interactive mode
#    pylab.hold(True)
#    pylab.ion()

    # Set up notifications
#    wm = pyinotify.WatchManager()
#    mask = pyinotify.IN_MODIFY | pyinotify.IN_CREATE
#    notifier = pyinotify.Notifier(wm, FileProcessor())
#    wdd = wm.add_watch(argv[1], mask, rec = True, auto_add = True)

    # Pre-process existing files
    processFiles()

#    while True:
#        try:
#            notifier.process_events()
#            if notifier.check_events():
#                notifier.read_events()
#        except KeyboardInterrupt:
#            notifier.stop()
#            break

