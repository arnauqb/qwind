from qwind import wind, utils
import numpy as np
import os
import sys
import shutil

N_CPUS = 16
#M_range = np.geomspace(1e7,1e9,2)
M_range = [1e6, 1e7, 1e8, 1e9]
#mdot_range = np.geomspace(0.05,1,2)
mdot_range = [0.1, 0.3, 0.5]
main_folder = "grid_results_nomura"

try:
    os.mkdir(main_folder)
except:
#    answer = input("delete folder?")
#    if(answer == 'y'):
    shutil.rmtree(main_folder)
    os.mkdir(main_folder)

for M in M_range:
    for mdot in mdot_range:
        fname = "M_%.2e_mdot_%.2f"%(M,mdot)
        qw = wind.Qwind(M=M, mdot = mdot, n_cpus = N_CPUS, nr = 16)
        qw.start_lines(niter = 50000)
        utils.save_results(qw, os.path.join(main_folder, fname))


