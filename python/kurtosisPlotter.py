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
			item = line.split(',')     
			data[0].append(float(item[0]))
			data[1].append(float(item[1]))
			data[2].append(float(item[2]))
			data[3].append(float(item[3]))
	
	# Create image data structure
	time = np.array(data[0])
	dm   = np.array(data[1])
	kurt = np.array(data[2])
	skew = np.array(data[3])

	dm   = np.unique(np.unique(dm))
	time = np.unique(np.unique(time))
	kurt = np.reshape(kurt, (np.size(time), np.size(dm)))
	skew = np.reshape(skew, (np.size(time), np.size(dm)))
	kurt = np.rot90(kurt)
	skew = np.rot90(skew)

	fig = plt.figure()
	fig.subplots_adjust(left=0.2, wspace=0.2)

	ax1 = fig.add_subplot(211)
	ax1.imshow(kurt, aspect="auto", extent=[np.min(time), np.max(time),np.min(dm),np.max(dm)])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('DM')
	ax1.set_title('DM vs Time plot')

	ax1 = fig.add_subplot(212)
	ax1.imshow(skew, aspect="auto", extent=[np.min(time), np.max(time),np.min(dm),np.max(dm)])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('DM')
	ax1.set_title('DM vs Time plot')

	plt.show()

