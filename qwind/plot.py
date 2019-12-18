import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm as colormaps
from qwind import grid as grid_module

class Plotter:

    def __init__(self, wind):

        self.wind = wind

    def plot_wind(self, r_max = 2000, z_max=500):
        fig, ax = plt.subplots()
        for line in self.wind.lines:
            ax.plot(line.r_hist, line.z_hist)
        ax.set_xlim(0,r_max)
        ax.set_ylim(0,z_max)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        return fig, ax 

    def plot_wind_property(self, y, x="t_hist", x_lim=None, y_lim=None):
        """
        Plot y wind property. x_lim and y_lim are list or tuples.
        """
        fig, ax = plt.subplots()
        for line in self.wind.lines:
            x_plot = getattr(line, x)
            y_plot = getattr(line, y)
            ax.plot(x_plot, y_plot)
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        return fig, ax 

    def plot_density_grid(self):
        grid=self.wind.radiation.density_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(grid_module.GRID_R_RANGE, grid_module.GRID_Z_RANGE, grid.grid.T, norm=LogNorm(), cmap=colormaps.matter)
        #cs = ax.contour(grid_module.GRID_R_RANGE, grid_module.GRID_Z_RANGE, grid.grid.T, levels=3)
        #ax.clabel(cs, inline=1)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        plt.colorbar(cm, ax=ax)
        return fig, ax

    def plot_ionization_grid(self):
        grid=self.wind.radiation.ionization_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(grid_module.GRID_R_RANGE, grid_module.GRID_Z_RANGE, grid.grid.T, norm=LogNorm(vmin=1e-5, vmax=1e6), cmap=colormaps.ice)
        plt.colorbar(cm, ax=ax)
        #cs = ax.contour(grid_module.GRID_R_RANGE, grid_module.GRID_Z_RANGE, grid.grid.T, levels=[1e5])
        #ax.clabel(cs, inline=1)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        return fig, ax

    def plot_tau_x_grid(self):
        grid = self.wind.radiation.tau_x_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(grid_module.GRID_R_RANGE, grid_module.GRID_Z_RANGE, grid.grid.T, norm=LogNorm(), cmap=colormaps.balance)
        #cs = ax.contour(grid_module.GRID_R_RANGE, grid_module.GRID_Z_RANGE, grid.grid.T, levels=[1e-2, 1e-1, 1e0, 1e1])
        #ax.clabel(cs, inline=1)
        plt.colorbar(cm, ax=ax)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        return fig, ax

    def plot_all_grids(self):
        cmaps = [colormaps.matter, colormaps.ice, colormaps.tempo]
        fig, ax = plt.subplots(1,4, figsize=(20,4))
        grids = [self.wind.radiation.density_grid,
                self.wind.radiation.ionization_grid,
                self.wind.radiation.tau_x_grid]
        for line in self.wind.lines:
            ax[0].plot(line.r_hist, line.z_hist)
        ax[0].set_xlim(0,3000)
        ax[0].set_ylim(0,3000)
        ax[0].set_xlabel("R [Rg]")
        ax[0].set_xlabel("z [Rg]")

        for i, grid in enumerate(grids):
            i = i + 1
            cm =ax[i].pcolormesh(grid_module.GRID_R_RANGE, grid_module.GRID_Z_RANGE, grid.grid.T, norm=LogNorm(), cmap = cmaps[i-1])
            plt.colorbar(cm, ax=ax[i]) 
            ax[i].set_xlabel("R [Rg]")
            ax[i].set_xlabel("z [Rg]")
            ax[i].set_xlim(grid_module.GRID_R_RANGE[0], grid_module.GRID_R_RANGE[-1])
            ax[i].set_ylim(grid_module.GRID_Z_RANGE[0], grid_module.GRID_Z_RANGE[-1])



