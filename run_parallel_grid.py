from qwind import wind, utils
import numpy as np
import os
import sys
import shutil
from argparse import ArgumentParser
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="output dir", metavar="FILE")
parser.add_argument("-r", "--rho", dest="rho_shielding", default=2e8,
                    help="density shielding")
parser.add_argument("-j", "--ncpus", dest="ncpus", default=4,
                    help="Number of CPUs to use.")
parser.add_argument("-v", "--vel", dest="v_z", default=1e7,
                    help="Initial vertical velocity.")

def run_bh(M, mdot):
    print("Starting M=%e, mdot = %f"%(M, mdot))
    sys.stdout.flush()
    fname = "M_%.2e_mdot_%.2f"%(M,mdot)
    qw = wind.Qwind(M=M,
            mdot = mdot,
            n_cpus = 1,
            nr = 32,
            radiation_mode="SimpleSED",
            modes = ["interp_fm"],
            rho_shielding = rho_shielding)
    qw.start_lines(niter = 50000,
            v_z_0 = v_z_0,
            rho = rho_shielding)
    utils.save_results(qw, os.path.join(main_folder, fname))
    print("Finished M=%e, mdot = %f"%(M, mdot))
    sys.stdout.flush()

args = parser.parse_args()
main_folder = str(args.filename) #"grid_rho_1e7"
rho_shielding = float(args.rho_shielding)
v_z_0 = float(args.v_z)
N_CPUS = int(args.ncpus) #16
print(main_folder, rho_shielding, N_CPUS, v_z_0)
M_range = np.geomspace(1e7,1e10,11)
mdot_range = np.geomspace(0.1,1,10)
params = []

for M in M_range:
    for mdot in mdot_range:
        params.append((M,mdot))
try:
    os.mkdir(main_folder)
except:
    pass

with Pool(N_CPUS) as pool:
    pool.starmap(run_bh, params)
