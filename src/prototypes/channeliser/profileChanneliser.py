import subprocess
import os, sys, re
import pickle

iterations = 2

if __name__ == "__main__":

    # Third run, loop over run parameters
    nchans = [16, 32, 64, 128, 256, 512, 1024]
    nsamp  = [2048, 4096, 8192, 16384]
    nsubs  = [8, 16, 32, 64, 128, 256, 512, 1024]
    beams  = [1, 2, 4, 8, 16, 32]
    taps   = [4, 8, 16, 32]
    values = []

    # Loop over beams
    for t in taps:

        # Build beamformer executable                
        print "\n\n############################## Building Channeliser #######################################"
        os.system("make -B NTAPS=%d" % (t))
        print "###############################################################################################\n"

        # Loop over beams
        for b in beams:
        
            # Loop over nsubs
            for s in nsubs:

                # Loop over channels:
                for c in nchans:

                    # Loop over nsamp:
                    for samp in nsamp:

                        firTime, firFlops, fftTime = 0, 0, 0

                        try:
                            # Run for a number of iterations to average running time
                            for i in xrange(iterations):

                                sys.stdout.write("Running for nsamp = %d, nchans = %d, beams = %d, nsubs = %d, ntaps = %d [Iteration %d]    \r" % (samp, c, b, s, t, i + 1) )
                                sys.stdout.flush()

                                # Launch process to run kernel
                                p = subprocess.Popen(["./ppf", "-nchans", str(c), "-nsamp", str(samp), "-nsubs", str(s), "-nbeams", str(b)], 
                                                      stdout=subprocess.PIPE)
                                out, err = p.communicate()

                                # Check for errors
                                if out.find('requirements') < 0:
                                    # Error detected, break from inner loop
                                    break


                                # Launch ready, extract running parameters
                                firTime  += float(re.search("Performed FIR in: (?P<time>\d+\.\d+)", out).groupdict()['time'])
                                firFlops += float(re.search("Flops: (?P<flops>\d+\.\d+)", out).groupdict()['flops'])
                                fftTime  += float(re.search("Performed FFT in: (?P<time>\d+\.\d+)", out).groupdict()['time'])

                        except:
                            pass

                        values.append({ 'nchans' : c, 'nsamp' : samp, 'nbeams' : b, 'nsubs' : s,
                                        'firtime' : firTime / iterations, 'flops' : firFlops / iterations,
                                        'ffttime' : fftTime / iterations,
                                        'taps' : t})

    # Save values list for future processing
    pickle.dump(values, open("channeliser_parameter_timings.pkl", "w"))
