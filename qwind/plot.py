import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm as colormaps

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
        grid=self.wind.density_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(grid.grid_r_range, grid.grid_z_range, grid.grid.T, norm=LogNorm(), cmap=colormaps.matter)
        #cs = ax.contour(grid.grid_r_range, grid.grid_z_range, grid.grid.T, levels=3)
        #ax.clabel(cs, inline=1)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        plt.colorbar(cm, ax=ax)
        return fig, ax

    def plot_ionization_grid(self):
        grid=self.wind.ionization_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(grid.grid_r_range, grid.grid_z_range, grid.grid.T, norm=LogNorm(vmin=1e-5, vmax=1e6), cmap=colormaps.ice)
        plt.colorbar(cm, ax=ax)
        #cs = ax.contour(grid.grid_r_range, grid.grid_z_range, grid.grid.T, levels=[1e5])
        #ax.clabel(cs, inline=1)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        return fig, ax

    def plot_tau_x_grid(self):
        grid = self.wind.tau_x_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(grid.grid_r_range, grid.grid_z_range, grid.grid.T, norm=LogNorm(), cmap=colormaps.tempo)
        #cs = ax.contour(grid.grid_r_range, grid.grid_z_range, grid.grid.T, levels=[1e-2, 1e-1, 1e0, 1e1])
        #ax.clabel(cs, inline=1)
        plt.colorbar(cm, ax=ax)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        return fig, ax

    def plot_all_grids(self):
        self.plot_density_grid()
        self.plot_ionization_grid()
        self.plot_tau_x_grid()


