import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm as colormaps
import matplotlib

class Plotter:

    def __init__(self, grid):

        self.grid = grid
#        matplotlib.use("Qt5Agg")

    def plot_wind(self, r_max = 2000, z_max=500):
        fig, ax = plt.subplots()
        for line in self.grid.wind.lines:
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
        for line in self.grid.wind.lines:
            x_plot = getattr(line, x)
            y_plot = getattr(line, y)
            ax.plot(x_plot, y_plot)
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        return fig, ax 

    def plot_density_grid(self):
        grid=self.grid.wind.radiation.density_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(self.grid.grid_r_range, self.grid.grid_z_range, grid.values.T, norm=LogNorm(), cmap=colormaps.matter)
        #cs = ax.contour(self.grid.grid_r_range, self.grid.grid_z_range, grid.grid.T, levels=3)
        #ax.clabel(cs, inline=1)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        plt.colorbar(cm, ax=ax)
        return fig, ax

    def plot_ionization_grid(self):
        grid=self.grid.wind.radiation.ionization_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(self.grid.grid_r_range, self.grid.grid_z_range, grid.values.T, norm=LogNorm(vmin=1e-5, vmax=1e6), cmap=colormaps.ice)
        plt.colorbar(cm, ax=ax)
        #cs = ax.contour(self.grid.grid_r_range, self.grid.grid_z_range, grid.grid.T, levels=[1e5])
        #ax.clabel(cs, inline=1)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        return fig, ax

    def plot_tau_x_grid(self):
        grid = self.grid.wind.radiation.tau_x_grid
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(self.grid.grid_r_range, self.grid.grid_z_range, grid.values.T, norm=LogNorm(), cmap=colormaps.balance)
        #cs = ax.contour(self.grid.grid_r_range, self.grid.grid_z_range, grid.grid.T, levels=[1e-2, 1e-1, 1e0, 1e1])
        #ax.clabel(cs, inline=1)
        plt.colorbar(cm, ax=ax)
        ax.set_xlabel("R [Rg]")
        ax.set_xlabel("z [Rg]")
        return fig, ax

    def plot_all_grids(self, fig=None, ax=None):
        cmaps = [colormaps.matter, colormaps.ice, colormaps.balance]
        grid_plots = []
        grids = [self.grid.density_grid.values,
                self.grid.ionization_grid.values,
                self.grid.tau_x_grid.values]
        #if fig is None or ax is None:
        fig, ax = plt.subplots(1,4, figsize=(20,5), sharey=True, sharex=True)
        for line in self.grid.wind.lines:
            ax[0].plot(line.r_hist, line.z_hist)
        ax[0].set_xlim(0,3000)
        ax[0].set_ylim(0,3000)
        ax[0].set_xlabel("R [Rg]")
        ax[0].set_xlabel("z [Rg]")

        for i, grid in enumerate(grids):
            i = i + 1
            if i == 2:
                cm =ax[i].pcolormesh(self.grid.grid_r_range,
                        self.grid.grid_z_range,
                        grid.T,
                        norm=LogNorm(1e-5, 1e7),
                        cmap=cmaps[i-1])
            elif i == 3:
                cm =ax[i].pcolormesh(self.grid.grid_r_range,
                        self.grid.grid_z_range,
                        grid.T,
                        norm=LogNorm(1e-2, 100),
                        cmap=cmaps[i-1])
            else:
                cm =ax[i].pcolormesh(self.grid.grid_r_range,
                        self.grid.grid_z_range,
                        grid.T,
                        norm=LogNorm(1e-5, 1e7),
                        cmap=cmaps[i-1])
            plt.colorbar(cm, ax=ax[i]) 
            ax[i].set_xlabel("R [Rg]")
            ax[i].set_xlabel("z [Rg]")
            ax[i].set_xlim(self.grid.grid_r_range[0], self.grid.grid_r_range[-1])
            ax[i].set_ylim(self.grid.grid_z_range[0], self.grid.grid_z_range[-1])
            grid_plots.append(cm)
        return fig, ax
        #else:
        #    for i, grid_plot in enumerate(grids):
        #        for line in self.wind.lines:
        #            ax[0].plot(line.r_hist, line.z_hist)
        #        ax[i+1].pcolormesh(self.grid.grid_r_range, self.grid.grid_z_range, grid_plot.grid.T, norm=LogNorm(), cmap = cmaps[i])
        #        fig.canvas.draw()
        #        fig.canvas.flush_events()




