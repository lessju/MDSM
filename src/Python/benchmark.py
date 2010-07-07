from process_log import process_logfile
import subprocess as sp
import os

configs = [('128', '131072', '4000'),  ('256', '131072', '4000'), ('500', '131072', '4000'), ('512', '131072', '4000'),
           ('1000', '131072', '4000'), ('1024', '131072', '4000'), ('2048', '131072', '4000'), ('2500', '131072', '4000'),
           ('512', str(16 * 1024), '4000'), ('512', str(32 * 1024), '4000'), ('512', str(64 * 1024), '4000'), ('512', str(128 * 1024), '4000'),
           ('512', str(256 * 1024), '4000'), ('512', '131072', '1000'), ('512', '131072', '2000'), 
           ('512', '131072', '3000'), ('512', '131072', '5000')]
           
for config in configs:
    if not os.path.exists("Timing/%s_%s_%s_output.txt" % config):
        p1 = sp.Popen([os.path.join(os.path.abspath('.'), 'main'), "-nchans", config[0], '-nsamp', 
                                                                    config[1], '-tdms', config[2]], stdout = sp.PIPE)
        output = p1.communicate()[0]
        f = open("Timing/%s_%s_%s_output.txt" % config, 'w')
        f.write(output)
        f.close()
        process_logfile("Timing/%s_%s_%s.txt" % config, data = output)

