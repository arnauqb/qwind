"""
Module to handle plotting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import seaborn as sns
#sns.set()
sns.set_context("talk")
from glob import glob
import os
from qwind import constants


def luminosity(m, mdot):
    Rg = constants.G * m * constants.Ms / constants.c**2
    return mdot * Rg * constants.emissivity_constant

def accretion_rate(m, mdot):
    lumin = luminosity(m,mdot)
    acc = lumin / (0.06 * constants.c**2)
    return acc

def pcolormesh_sensible(x_range, y_range, data, ax, log = True):
    
    if(log):
        x_range_log = np.log10(x_range)
        y_range_log = np.log10(y_range)
    else:
        x_range_log = x_range
        y_range_log = y_range
        
    dx = x_range_log[1] - x_range_log[0]
    #for i in range(1,len(x_range)):
        #assert dx == x_range_log[i] - x_range_log[i-1], "x array must be equally spaced"
    dy = y_range_log[1] - y_range_log[0]
    #for i in range(1,len(y_range_log)):
        #assert dy == y_range_log[i] - y_range_log[i-1], "y array must be equally spaced"
        
    if(log):
        x_range_plot = np.geomspace(10**(np.log10(x_range[0]) - dx/2.), 10**(np.log10(x_range[-1]) + dx/2.), len(x_range) + 1)
        y_range_plot = np.geomspace(10**(np.log10(y_range[0]) - dy/2.), 10**(np.log10(y_range[-1]) + dy/2.), len(y_range) + 1)
    else:
        x_range_plot = np.linspace(x_range[0] - dx/2., x_range[-1] + dx/2., len(x_range) + 1)
        y_range_plot = np.linspace(y_range[0] - dy/2., y_range[-1] + dy/2., len(y_range) + 1)
     
    cmap = ax.pcolormesh(x_range_plot, y_range_plot, np.transpose(data))
    if(log):
        ax.set_xscale('log')
        ax.set_yscale('log')
    return cmap
    
def read_data(grid_folder):
    """
    Parses data into Pandas dataframe.
    """
    files = glob(os.path.join(grid_folder, "*"))
    mdot_wind_list = []
    for file in files:
        fname = file.split("/")[-1]
        M = float(fname.split("_")[1])
        mdot = float(fname.split("_")[-1])
        mass_loss_file = os.path.join(file, "mass_loss.csv")
        with open(mass_loss_file, "r") as f:
            lines = f.readlines()
            mloss = float(lines[0].split(' ')[3])   
        mloss_norm = mloss / accretion_rate(M, mdot)
        mloss_msun = mloss / constants.Ms * constants.year
        data = {
            'M' : M,
            'mdot' : mdot,
            'mloss' : mloss,
            'mloss_norm' : mloss_norm,
            "mloss_msun" : mloss_msun,
        }
        mdot_wind_list.append(data)
    data_pd = pd.DataFrame.from_dict(mdot_wind_list)
    data_pd = data_pd.sort_values(['M','mdot'])

    return data_pd

def plot_mloss_grid(grid_folder, title = "Wind mass loss."):
    """
    Reads data and creates mloss grid.
    """
    data = read_data(grid_folder)
    #print(data)
    M_range = data.M.unique()
    mdot_range = data.mdot.unique()
    data_grid = np.array(data.mloss_norm).reshape(len(M_range), len(mdot_range))
    fig, ax = plt.subplots(figsize = (15,10))
    cmap = pcolormesh_sensible(M_range, mdot_range, data_grid, ax, log = True)
    for i in range(len(M_range)):
        for j in range(len(mdot_range)):
            M = M_range[i]
            mdot = mdot_range[j]
            mloss_sun = data[(data.M == M) & (data.mdot == mdot)]["mloss_msun"].values[0]
            mloss_sun = "%.3f"%mloss_sun
            text = ax.text(M, mdot, mloss_sun, ha="center", va="center", color="w")
            
    cbar = plt.colorbar(cmap, ax = ax)
    cbar.ax.set_ylabel(r"$\dot M_\mathrm{wind} \; / \; \dot M$ ")
    ax.set_xlabel(r"$M_{BH}$")
    ax.set_ylabel(r"$\dot m$")
    plt.tight_layout()
    ax.set_title(title)   
    return fig, ax 
    
def plot_cross_grid(grid_folder, title = "Wind grid"):
    """
    Reads data and creates mloss grid.
    """
    data = read_data(grid_folder)
    mask = data["mloss"] > 0.01
    data_has_wind = data[mask]
    data_no_wind = data[~mask]
    
    fig, ax = plt.subplots(figsize = (8,6))
    ax.scatter(data_has_wind.M, np.log10(data_has_wind.mdot), color = 'b', marker = "+")
    ax.scatter(data_no_wind.M, np.log10(data_no_wind.mdot), color = 'r', marker = "_")
    
    ax.set_xscale('log')
    ax.set_xlim(7e6,1.5e9)
    ax.set_xlabel(r"$M_{BH}$")
    ax.set_ylabel(r"$\dot m$")
    ax.set_ylim( -1.1, 0.1)
    plt.tight_layout()
    ax.set_title(title)
    labels = [float(item.get_text().replace("−", "-")) for item in ax.get_yticklabels()]
    labels_new = ["%.2f"%(10**label) for label in labels]
    b = ax.set_yticklabels(labels_new)
    return fig, ax

def plot_final_velocity(bh_folder):
    
    lines = glob(os.path.join(bh_folder, "lin*"))
    fig, ax = plt.subplots(figsize=(9,6))
    v_s = []
    r_s = []
    for line_f in lines:
        line = pd.read_csv(line_f)
        v_s.append(line.v_T_hist[-1] / line.v_esc_hist[-1])
        r_s.append(line.v_r_hist[0])
    ax.plot(r_s, v_s)
    plt.show()
        
    
    