import subprocess as sp

# Define benchmark parametes
nsamp = [8192, 16384, 32768, 65536, 131072, 262144]
tdms = [1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
nchans = [32, 64, 128, 256, 512, 1024, 2048, 4096]

outputFile = open('benchmarks.txt', 'w')

for d in tdms:
    for s in nsamp:
        for c in nchans:
            p = sp.Popen(['./cpu', '-nsamp', str(s), '-nchans', str(c), '-tdms', str(d)], stdout = sp.PIPE)
            out = p.communicate()
            outputFile.write(out[0] + '\n')
            outputFile.flush()
