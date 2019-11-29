from qwind import constants
import shutil
import os
import pandas as pd
import pickle
import numpy as np
import json


def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


def save_results(wind, folder_name="Results"):
    """
    Saves results to filename.
    """
    try:
        os.mkdir(folder_name)
    except:
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)

    metadata_file = os.path.join(folder_name, "metadata.txt")
    with open(metadata_file, "w") as f:
        f.write("M: \t %.2e\n" % wind.M)
        f.write("Mdot: \t %.2e\n" % wind.mdot)
        f.write("a: \t %.2e\n" % wind.spin)
        f.write("rho_shielding: \t %e\n" % wind.rho_shielding)
        f.write("r_in: \t %f\n" % wind.lines_r_min)
        f.write("r_out: \t %f\n" % wind.lines_r_max)
        f.write("f_uv: \t %f\n" % wind.radiation.uv_fraction)
        f.write("f_x: \t %f\n" % wind.radiation.xray_fraction)

    for i, line in enumerate(wind.lines):
        line_name = "line_%02d" % i
        lines_file = os.path.join(folder_name, line_name + ".csv")
        a_hist = np.array(line.a_hist)
        data = {
            'R': line.r_hist,
            #'P': line.phi_hist,
            'Z': line.z_hist,
            #'X': line.x_hist,
            'V_R': line.v_r_hist,
            #'V_PHI': line.v_phi_hist,
            'V_Z': line.v_z_hist,
            'V_T': line.v_T_hist,
            #'a_r': a_hist[:,0],
            #'a_z': a_hist[:,-1],
            'rho': line.rho_hist,
            'xi': line.xi_hist,
            'fm': line.fm_hist,
            'tau_dr': line.tau_dr_hist,
            'tau_uv': line.tau_uv_hist,
            'tau_x': line.tau_x_hist,
            'dv_dr': line.dv_dr_hist,
            'escaped': line.escaped * np.ones(len(line.rho_hist)),
            #'V_esc': line.v_esc_hist,
        }
        df = pd.DataFrame.from_dict(data)
        df.to_csv(lines_file, index=False)
    properties = {
        "mdot_w_gs" : wind.mdot_w,
        "mdot_w_msunyr" : wind.mdot_w / constants.M_SUN * constants.YEAR_TO_SEC,
        "kin_lumin" : wind.kinetic_luminosity,
        "angle" : wind.angle,
        "terminal_velocity" : wind.v_terminal,
    }
    with open(os.path.join(folder_name, "wind_properties.csv"), "w") as f:
        f.write(json.dumps(properties))
    return 1
