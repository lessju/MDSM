import subprocess as sp
import re, os

configs = [('64', str(100 * 1024)), ('128', str(100 * 1024)), ('256', str(100 * 1024)), ('512', str(100 * 1024)), ('1024', str(100 * 1024)), 
           ('2048', str(100 * 1024)), ('4096', str(100 * 1024)), 
           ('1024', str(32 * 1024)), ('1024', str(64 * 1024)), ('1024', str(128 * 1024)), ('1024', str(256 * 1024))]#, ('1024', str(512 * 1024))]
infile = open('log.txt', 'w')

for config in configs:
    if not os.path.exists("Timing/tree_%s_%s_output.txt" % config):
        print config
        p1 = sp.Popen([os.path.join(os.path.abspath('.'), 'dmtree'), "-nchans", config[0], '-nsamp', config[1]], stdout = sp.PIPE)
        output = p1.communicate()[0]
        f = open("Timing/%s_%s_output.txt" % config, 'w')
        f.write(output)
        f.close()

        values = re.search('nsamp: (?P<nsamp>\d*), nchans: (?P<nchans>\d*)', output).groupdict()
	nchans, nsamp = int(values['nchans']), int(values['nsamp'])
	times = [float(item) for item in re.findall('\d* processed in: (?P<time>\S*)', output)]
        out = '%d, %d, %0.2f: %.3f\n'%( nchans, nsamp, sum(times) / len(times), nchans * nchans * nsamp / (1e6 * sum(times) / len(times)))
        infile.write(out)
        print out

infile.close()


