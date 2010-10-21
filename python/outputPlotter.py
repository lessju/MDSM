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

labels = ["DM", "Time", "Intensity"]

#NOTE params: outputPlotter.py basedir 'file_regex' x:y [opt=value]*
if __name__ == "__main__":

    args = get_args()
    print argv

    # Get filenames
    files = [item for item in os.listdir(argv[1]) if fnmatch.fnmatch(item, argv[2])]
    
    # Load files and create datastructures
    data = ([],[],[])
    for f in files:
        f = open(argv[1] + '/' + f, 'r')
        d = f.read().split('\n')
        for line in d[:-1]:
            item = line.split(',')     
            data[0].append(float(item[0]))
            data[1].append(float(item[1]))
            data[2].append(float(item[2]))

    # Get dimensions to plot
    if len(argv) < 4:
        print "Not enough parameters"
        exit()
    else:
        ind = argv[3].find(':')

        # Single plot
        if ind > 0:
            x_index = int(argv[3][:ind]) - 1
            y_index = int(argv[3][ind + 1:]) - 1

            plt.plot(data[x_index], data[y_index], '+r')
            plt.grid(True)
            plt.xlabel(labels[x_index])
            plt.ylabel(labels[y_index])
            plt.title("%s vs %s plot" % (labels[y_index], labels[x_index]))
            plt.show()

        # All plots
        elif argv[3] == 'all':
            fig = plt.figure()
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

            plt.show()

#        elif argv[3] == '3d' or argv[3] == '3D':
#            fig = plt.figure()
#            ax = Axes3D(fig)
#            ax.scatter(np.array(data[0]), np.array(data[1]), np.array(data[2]),'ro')
#            ax.set_xlabel('Time')
#            ax.set_ylabel('DM')
#            ax.set_zlabel('Intensity')
#            plt.show()

        else:
            print "Invalid x/y values"
            exit()

