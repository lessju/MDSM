#! /usr/bin/env python
""" 
Script for initializing S Engine of the Medicina seng.ger
"""
import time, sys, numpy, os, katcp, socket, struct
from corr import katcp_wrapper, log_handlers
import subprocess

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    print "Unexpected error:", sys.exc_info()
    try:
        seng.disconnect_all()
    except: pass
    exit()

def exit_clean():
    try:
        seng.disconnect_all()
    except: pass
    exit()

def gen_ip(ip_str):
    ip_v = ip_str.split('.')
    ip = (int(ip_v[0]) << 24) + (int(ip_v[1]) << 16) + (int(ip_v[2]) << 8) + (int(ip_v[3])) 
    return ip

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('seng_init.py [options] CONFIG_FILE')
    p.set_description(__doc__)
    p.add_option('-p', '--skip_prog', dest='prog_fpga',action='store_false', default=True, 
        help='Skip FPGA programming (assumes already programmed).  Default: program the FPGAs')
    p.add_option('-n', '--nbeams', dest='nbeams',type='int', default=1, 
        help='Set the number of beams to be transmitted (up to 8). Default is 1')
    p.add_option('-r', '--roach_ip', dest='roach_ip',type='string', default='hammy', 
        help='Roach to use (ip or hostname). Default is hammy')
    p.add_option('--src_ip', dest='src_ip',type='string', default='10.0.0.100', 
        help='tgbe source ip. Default is 10.0.0.100')
    p.add_option('--dest_port', dest='dest_port',type='int', default=10000, 
        help='tgbe destination port. Default is 10000')
    p.add_option('--dest_ip', dest='dest_ip',type='string', default='10.0.0.145', 
        help='tgbe destination ip. Default is 10.0.0.145')
    p.add_option('-b', '--boffile', dest='boffile',type='string', default='spead_tx_test.bof', 
        help='Boffile to load. Default is spead_tx_test.bof')
    p.add_option('-v', '--verbose', dest='verbose',action='store_true', default=False, 
        help='Be verbose about errors.')

    opts, args = p.parse_args(sys.argv[1:])



lh=log_handlers.DebugLogHandler()

try:
    seng = katcp_wrapper.FpgaClient(opts.roach_ip, 7147)
    time.sleep(0.2)

    if opts.prog_fpga:
        print 'programming roach %s with boffile %s' %(opts.roach_ip, opts.boffile)
        seng.progdev(opts.boffile)
    else:
        print 'skipping programming'

    print 'Estimating board clock:'
    brd_clk = seng.est_brd_clk()
    print brd_clk

    print '\n======================'
    print 'Initial configuration:'
    print '======================'

    print 'setting destination IP to %s' %opts.dest_ip
    seng.write_int('gbe_dest_ip', gen_ip(opts.dest_ip))
    
    print 'setting destination port to %d' %opts.dest_port
    seng.write_int('gbe_dest_port', opts.dest_port)

    print 'setting number of beams to %d' %opts.nbeams
    seng.write_int('n_beams', opts.nbeams-1)

    print 'Configuring tengbe core %s:%d' %(opts.src_ip, opts.dest_port)
    src_ip = gen_ip(opts.src_ip)
    src_mac = (0<<40)+(96<<32)+src_ip
    seng.tap_start('tge0', 'ten_Gbe_v2', src_mac, src_ip, opts.dest_port)

    print 'resetting everything'
    seng.write_int('ctrl_sw',3)
    seng.write_int('ctrl_sw',0)

    print 'Checking packet output'
    n0 = seng.read_uint('gbe_packet_cnt')
    time.sleep(1)
    n1 = seng.read_uint('gbe_packet_cnt')
    pkt_ps = n1-n0

    print '%d packets transmitted in the last second (%.2f Gb/s)' %(pkt_ps, pkt_ps*4096*8/1e9)

    print 'data rate is: %.2f Gb/s' %(32 * brd_clk * 1e6 * float(opts.nbeams)/8.0 /1e9)
    print 'data rate (including headers) is: %.2f Gb/s' %(32 * brd_clk * 1e6 * float(opts.nbeams)/8.0 /1e9 * ((512.0+6.0)/512.))


except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()
