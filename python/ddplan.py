import elementtree.ElementTree as ET
import sys
from math import pow
from numpy import *

class observation:
    def __init__(self, dt, f_ctr, BW, numchan):
        # dt in sec, f_ctr and in MHz
        self.dt = dt
        self.f_ctr = f_ctr
        self.BW = BW
        self.numchan = numchan
        self.chanwidth = BW/numchan
        
    def guess_dDM(self, DM):
        """
        guess_dDM(self, DM):
            Choose a reasonable dDM by setting the maximum smearing across the
                'BW' to equal the sampling time 'dt'.
        """
        return self.dt*0.0001205*self.f_ctr**3.0/(0.5*self.BW)

class dedisp_method:
    def __init__(self, obs, downsamp, loDM, hiDM, dDM, numDMs=0,
                 numsub=0, smearfact=2.0):
        self.obs = obs
        self.downsamp = downsamp
        self.loDM = loDM
        self.dDM = dDM
        self.numsub = numsub
        self.BW_smearing = BW_smear(dDM, self.obs.BW, self.obs.f_ctr)
        self.numprepsub = 0
        if (numsub):  # Calculate the maximum subband smearing we can handle
            DMs_per_prepsub = 2
            while(1):
                next_dsubDM = (DMs_per_prepsub+2) * dDM
                next_ss = subband_smear(next_dsubDM, numsub, self.obs.BW, self.obs.f_ctr)
                # The 0.8 is a small fudge factor to make sure that the subband
                # smearing is always the smallest contribution
                if (next_ss > 0.8*min(self.BW_smearing, 1000.0*obs.dt*downsamp)):
                    self.dsubDM = DMs_per_prepsub*dDM
                    self.DMs_per_prepsub = DMs_per_prepsub
                    self.sub_smearing = subband_smear(self.dsubDM, numsub,
                                                      self.obs.BW, self.obs.f_ctr)
                    break
                DMs_per_prepsub += 2
        else:
            self.dsubDM = dDM
            self.sub_smearing = subband_smear(self.dsubDM, numsub, self.obs.BW, self.obs.f_ctr)
        # Calculate the nominal DM to move to the next method
        cross_DM = self.DM_for_smearfact(smearfact)
        if (cross_DM > hiDM):
            cross_DM = hiDM
        if (numDMs==0):
            self.numDMs = int(ceil((cross_DM-loDM)/dDM))
            if (numsub):
                self.numprepsub = int(ceil(self.numDMs*dDM / self.dsubDM))
                self.numDMs = self.numprepsub * DMs_per_prepsub
        else:
            self.numDMs = numDMs
        self.hiDM = loDM + self.numDMs*dDM
        self.DMs = arange(self.numDMs, dtype='d')*dDM + loDM
        
    def chan_smear(self, DM):
        """
        Return the smearing (in ms) in each channel at the specified DM
        """
        try:
            DM = where(DM==0.0, self.dDM/2.0, DM)
        except TypeError:
            if (DM==0.0): DM = self.dDM/2.0
        return dm_smear(DM, self.obs.chanwidth, self.obs.f_ctr)
    def total_smear(self, DM):
        """
        Return the total smearing in ms due to the sampling rate,
        the smearing over each channel, the smearing over each subband
        (if numsub > 0) and the smearing over the full BW assuming the
        worst-case DM error.
        """
        return sqrt((1000.0*self.obs.dt)**2.0 +
                    (1000.0*self.obs.dt*self.downsamp)**2.0 +
                    self.BW_smearing**2.0 +
                    self.sub_smearing**2.0 +
                    self.chan_smear(DM)**2.0)
    
    def DM_for_smearfact(self, smearfact):
        """
        Return the DM where the smearing in a single channel is a factor smearfact
        larger than all the other smaring causes combined.
        """
        other_smear = sqrt((1000.0*self.obs.dt)**2.0 +
                           (1000.0*self.obs.dt*self.downsamp)**2.0 +
                           self.BW_smearing**2.0 +
                           self.sub_smearing**2.0)
        return smearfact*0.001*other_smear/self.obs.chanwidth*0.0001205*self.obs.f_ctr**3.0
    
    def DM_for_newparams(self, dDM, downsamp):
        """
        Return the DM where the smearing in a single channel is causes the same smearing
        as the effects of the new dosnsampling rate and dDM.
        """
        other_smear = sqrt((1000.0*self.obs.dt)**2.0 +
                           (1000.0*self.obs.dt*downsamp)**2.0 +
                           BW_smear(dDM, self.obs.BW, self.obs.f_ctr)**2.0 +
                           self.sub_smearing**2.0)
        return 0.001*other_smear/self.obs.chanwidth*0.0001205*self.obs.f_ctr**3.0


    def __str__(self):
        if (self.numsub):
            return "%9.3f  %9.3f  %6.2f    %4d  %6.2f  %6d  %6d  %6d " % \
                   (self.loDM, self.hiDM, self.dDM, self.downsamp, self.dsubDM,
                    self.numDMs, self.DMs_per_prepsub, self.numprepsub)
        else:
            return "%9.3f  %9.3f  %6.2f    %4d  %6d" % \
                   (self.loDM, self.hiDM, self.dDM, self.downsamp, self.numDMs)
        
def dm_smear(DM, BW, f_ctr):
    """
    dm_smear(DM, BW, f_ctr):
        Return the smearing in ms caused by a 'DM' over a bandwidth
        of 'BW' MHz centered at 'f_ctr' MHz.
    """
    return 1000.0*DM*BW/(0.0001205*f_ctr**3.0)

def BW_smear(DMstep, BW, f_ctr):
    """
    BW_smear(DMstep, BW, f_ctr):
        Return the smearing in ms caused by a search using a DM stepsize of
        'DMstep' over a bandwidth of 'BW' MHz centered at 'f_ctr' MHz.
    """
    maxDMerror = 0.5*DMstep
    return dm_smear(maxDMerror, BW, f_ctr)

def guess_DMstep(DM, dt, BW, f_ctr):
    """
    guess_DMstep(DM, dt, BW, f_ctr):
        Choose a reasonable DMstep by setting the maximum smearing across the
        'BW' to equal the sampling time 'dt'.
    """
    return dt*0.0001205*f_ctr**3.0/(0.5*BW)

def subband_smear(subDMstep, numsub, BW, f_ctr):
    """
    subband_smear(subDMstep, numsub, BW, f_ctr):
        Return the smearing in ms caused by a search using a subband
        DM stepsize of 'subDMstep' over a total bandwidth of 'BW' MHz
        centered at 'f_ctr' MHz, and having numsub subbands.
    """
    if (numsub==0): return 0.0
    subBW = BW/numsub
    maxsubDMerror = 0.5*subDMstep
    return dm_smear(maxsubDMerror, subBW, f_ctr)

def total_smear(DM, DMstep, dt, f_ctr, BW, numchan, subDMstep, numsub=0):
    """
    total_smear(DM, DMstep, dt, f_ctr, BW, numchan, subDMstep, numsub=0):
        Return the total smearing in ms due to the sampling rate,
        the smearing over each channel, the smearing over each subband
        (if numsub > 0) and the smearing over the full BW assuming the
        worst-case DM error.
    """
    return sqrt(2 * (1000.0*dt)**2.0 +
                dm_smear(DM, BW/numchan, f_ctr)**2.0 +
                subband_smear(subDMstep, numsub, BW, f_ctr)**2.0 + 
                BW_smear(DMstep, BW, f_ctr)**2.0)

def dm_steps(loDM, hiDM, obs, numsub=0, ok_smearing=0.0):
    """
    dm_steps(loDM, hiDM, obs, numsub=0, ok_smearing=0.0):
        Return the optimal DM stepsizes (and subband DM stepsizes if
        numsub>0) to keep the total smearing below 'ok_smearing' (in ms),
        for the DMs between loDM and hiDM.  If 'ok_smearing'=0.0, then
        use the best values based only on the data.
    """
    # Allowable DM stepsizes
    allow_dDMs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0,
                  2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0]
    # Allowable number of downsampling factors
    allow_downsamps = [pow(2, i) for i in range(0, 20)]

    # Initial values
    index_downsamps = index_dDMs = 0
    downsamp = allow_downsamps[index_downsamps]
    dDM = allow_dDMs[index_dDMs]
    dtms = 1000.0*obs.dt
    
    # Fudge factor that "softens" the boundary defining
    # if 2 time scales are equal or not
    ff = 1.2
    
    # This is the array that will hold the de-dispersion plans
    methods = []

    # Minimum possible smearing
    min_tot_smearing = total_smear(loDM+0.5*dDM, dDM, obs.dt, obs.f_ctr,
                                   obs.BW, obs.numchan, allow_dDMs[0], 0)
    # Minimum channel smearing
    min_chan_smearing = dm_smear(loDM+0.5*dDM, obs.chanwidth, obs.f_ctr)
    # Minimum smearing across the obs.BW
    min_BW_smearing = BW_smear(dDM, obs.BW, obs.f_ctr)

    ok_smearing = max([ok_smearing, min_chan_smearing, min_BW_smearing, dtms])

    # See if the data is too high time resolution for our needs
    if (ff*min_chan_smearing > dtms or
        ok_smearing > dtms):
        if (ok_smearing > ff*min_chan_smearing):
            okval = ok_smearing
        else:
            okval = ff*min_chan_smearing

        while (dtms*allow_downsamps[index_downsamps+1] < okval):
            index_downsamps += 1    
        downsamp = allow_downsamps[index_downsamps]

    # Calculate the appropriate initial dDM 
    dDM = guess_DMstep(loDM, obs.dt*downsamp, obs.BW, obs.f_ctr)
    while (allow_dDMs[index_dDMs+1] < ff*dDM):
        index_dDMs += 1

    # Create the first method
    methods = [dedisp_method(obs, downsamp, loDM, hiDM,
                             allow_dDMs[index_dDMs], numsub=numsub)]
    numDMs = [methods[-1].numDMs]
   
    # Calculate the next methods
    while(methods[-1].hiDM < hiDM):

        # Determine the new downsample factor
        index_downsamps += 1
        downsamp = allow_downsamps[index_downsamps]
        eff_dt = dtms*downsamp

        # Determine the new DM step
        while (BW_smear(allow_dDMs[index_dDMs+1], obs.BW, obs.f_ctr) < ff*eff_dt):
            index_dDMs += 1
        dDM = allow_dDMs[index_dDMs]

        # Get the next method
        methods.append(dedisp_method(obs, downsamp, methods[-1].hiDM,
                                     hiDM, dDM, numsub=numsub))
        numDMs.append(methods[-1].numDMs)
            
    # Calculate the DMs to search and the smearing at each
    total_numDMs = sum(numDMs)
    DMs = zeros(total_numDMs, dtype='d')
    total_smears = zeros(total_numDMs, dtype='d')

    # Calculate the DMs and optimal smearing for all the DMs
    for ii, offset in enumerate(add.accumulate([0]+numDMs[:-1])):
        DMs[offset:offset+numDMs[ii]] = methods[ii].DMs
        total_smears[offset:offset+numDMs[ii]] = methods[ii].total_smear(methods[ii].DMs)

    # Calculate the predicted amount of time that will be spent in searching
    # this batch of DMs as a fraction of the total
    work_fracts = [meth.numDMs/float(meth.downsamp) for meth in methods]
    work_fracts = asarray(work_fracts)/sum(work_fracts)

    # The optimal smearing
    tot_smear = total_smear(DMs, allow_dDMs[0], obs.dt, obs.f_ctr,
                            obs.BW, obs.numchan, allow_dDMs[0], 0)

    if (numsub):
        print "\n  Low DM    High DM     dDM  DownSamp  dsubDM   #DMs  DMs/call  calls  WorkFract"
    else:
        print "\n  Low DM    High DM     dDM  DownSamp   #DMs  WorkFract"
    for method, fract in zip(methods, work_fracts):
        print method, "  %.4g" % fract
    print "\n\n"

#    root = ET.Element("observation")

#    freq = ET.SubElement(root, "frequencies")
#    freq.set("center", str(0))
#    freq.set("offset", str(0))

#    dm = ET.SubElement(root, "dm")

#    channels = ET.SubElement(root, "channels")
#    channels.set("number", str(0))
#    channels.set("subbands", str(4))

#    timing = ET.SubElement(root, "timing")
#    timing.set("tsamp", str(0))

#    passes = ET.SubElement(root, "passes")
#    for item in methods:
#        currpass = ET.SubElement(passes, "pass")
#        ET.SubElement(currpass, "lowDm").text = str(item.loDM)
#        ET.SubElement(currpass, "highDm").text = str(item.hiDM)
#        ET.SubElement(currpass, "deltaDm").text = str(item.dDM)
#        ET.SubElement(currpass, "downsample").text = str(item.downsamp)
#        ET.SubElement(currpass, "subDm").text = str(item.dsubDM)
#        ET.SubElement(currpass, "numDms").text = str(item.numDMs)
#        ET.SubElement(currpass, "dmsPerCall").text = str(item.DMs_per_prepsub)
#        ET.SubElement(currpass, "ncalls").text = str(item.numprepsub)

#    # wrap it in an ElementTree instance, and save as XML
#    tree = ET.ElementTree(root)
#    tree.write("observation.xml")

    return methods  

def calculateParams(loDM, hiDM, fctr, BW, numchan, numsubbands, dt, ok_smearing):

    # The following is an instance of an "observation" class
    obs = observation(dt, fctr, BW, numchan)

    return dm_steps(loDM, hiDM, obs, numsub=numsubbands,
             ok_smearing=ok_smearing)

    
def usage():
    print """
    usage:  DDplan.py [options]
      [-h, --help]                    : Display this help
      [-o outfile, --outfile=outfile] : Output .eps plot file (default is xwin)
      [-l loDM, --loDM=loDM]          : Low DM to search   (default = 0 pc cm-3)
      [-d hiDM, --hiDM=HIDM]          : High DM to search  (default = 1000 pc cm-3)
      [-f fctr, --fctr=fctr]          : Center frequency   (default = 1400MHz)
      [-b BW, --bw=bandwidth]         : Bandwidth in MHz   (default = 300MHz)
      [-n #chan, --numchan=#chan]     : Number of channels (default = 1024)
      [-t dt, --dt=dt]                : Sample time (s)    (default = 0.000064 s)
      [-s subbands, --subbands=nsub]  : Number of subbands (default = #chan) 
      [-r resolution, --res=res]      : Acceptable time resolution (ms)
      The program generates a good plan for de-dispersing raw data.  It
      trades a small amount of sensitivity in order to save computation costs.

    """    

if __name__=='__main__':
    import getopt, sys

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:l:d:f:b:n:t:s:r:",
                                   ["help", "output=", "loDM=", "hiDM=",
                                    "fctr=", "bw=", "numchan=", "dt=",
                                    "subbands=", "res="])

    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)
    if len(sys.argv)==1:
        usage()
        sys.exit(2)
    # The defaults are close to the future ALFA survey
    loDM, hiDM = 0.0, 1000.0
    fctr = 1400.0
    BW = 300.0
    numchan = 1024
    numsubbands = 0
    dt = 0.000064
    ok_smearing = 0.0

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-l", "--loDM"):
            loDM = float(a)
        if o in ("-d", "--hiDM"):
            hiDM = float(a)
        if o in ("-f", "--fctr"):
            fctr = float(a)
        if o in ("-b", "--bw"):
            BW = float(a)
        if o in ("-n", "--numchan"):
            numchan = int(a)
        if o in ("-t", "--dt"):
            dt = float(a)
        if o in ("-s", "--subbands"):
            numsubbands = int(a)
        if o in ("-r", "--res"):
            ok_smearing = float(a)

    obs = observation(dt, fctr, BW, numchan)
    dm_steps(loDM, hiDM, obs, numsub=numsubbands,
             ok_smearing=ok_smearing)
    
