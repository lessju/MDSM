from __future__ import with_statement
from sys import argv
import re

def process_logfile(outfile, infile=None , data=None):
        """ Process logfile """

        if infile and not data:
            f = open(infile, 'r')
	    data = f.read()
	    f.close()
        elif not data:
            print "Invalid parameters"

	# Extract general run parameters
	values = re.search('nchans: (?P<nchans>\d*), nsamp: (?P<nsamp>\d*), tsamp: (?P<tsamp>\S*), foff: (?P<foff>\S*)', data).groupdict()
	nchans, nsamp, tsamp, foff = int(values['nchans']), int(values['nsamp']), float(values['tsamp']), float(values['foff'])

	values = re.search('ndms: (?P<ndms>\d*), max dm: (?P<maxdm>\S*), maxshift: (?P<maxshift>\d*)', data).groupdict()
	ndms, maxdm, maxshift = int(values['ndms']), float(values['maxdm']), int(values['maxshift'])

	values = re.search('input size: (?P<isize>\S*), output size per GPU: (?P<osize>\S*)', data).groupdict()
	isize, osize = float(values['isize']), float(values['osize'])

	# Extract GPU specs
	a = re.findall('\d*: Copied data to GPU (?P<num>\d): (?P<time>\S*)', data)
	b = re.findall('\d*: Processed (?P<num>\d): (?P<time>\S*)', data)
	c = re.findall('\d*: Written output (?P<num>\d): (?P<time>\S*)', data)

	output = ""
	gpus = [{'togpu' : [], 'process': [], 'tohost': []} for i in range(4)]
	for i in range(min([len(a), len(b), len(c)])):
	    gpus[int(a[i][0]) - 1]['togpu'].append(float(a[i][1]))
	    gpus[int(b[i][0]) - 1]['process'].append(float(b[i][1]))
	    gpus[int(c[i][0]) - 1]['tohost'].append(float(c[i][1]))

	output += "\nRUN PARAMETERS\n--------------\n"
	output +=  "\tnsamp:\t\t%d\n\tnchans:\t\t%d\n\tndms:\t\t%d\n" % (nsamp, nchans, ndms)

	output +=  "\nGPU STATISTICS\n--------------\n"
	for i in range(4):
	    output +=  "Stats for GPU %d:\n" % i
	    output +=  "\tCopy %f MB to GPU:\t %f\n" % (isize, sum(gpus[i]['togpu']) / len(gpus[i]['togpu']))
	    output +=  "\tProcess %d x %d samples:\t %f\n" % (nchans, nsamp, sum(gpus[i]['process']) / len(gpus[i]['process']))
	    output +=  "\tCopy %f MB to host:\t %f\n" % (osize, sum(gpus[i]['tohost']) / len(gpus[i]['tohost']))
	    output +=  "\tCalculated output BW: \t\t %f MB/s\n" % (isize * 1000 / (sum(gpus[i]['togpu']) / len(gpus[i]['togpu'])))
	    output +=  "\tCalculated input BW: \t\t %f MB/s\n" % (osize * 1000 / (sum(gpus[i]['tohost']) / len(gpus[i]['tohost'])))
	    output +=  "\tCalculated DDPS: \t\t %e\n\n" % ( 1000 * nchans * nsamp * ndms / (sum(gpus[i]['process']) / len(gpus[i]['process'])) )

 	# Create processed log file
        with open(outfile, 'w') as fw:
	    fw.write(output)
	    fw.write('\n'.join(data.split('\n')))

        return gpus

if __name__ == '__main__':
    res = process_logfile(str(argv[1]), str(argv[2]))


