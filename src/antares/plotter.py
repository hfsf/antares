# -*- coding: utf-8 -*-

"""
Plotter Module.

Provides high-level plotting utilities for the ANTARES framework.
Interacts directly with the Results object to generate publication-quality
visualizations using matplotlib, pandas, and seaborn integration.
Supports 0D dynamics, 1D spatial profiles, 2D heatmaps, and 3D cross-section slicing.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import antares.core.GLOBAL_CFG as cfg


class Plotter:
    """
    Definition of Plotter class. Handles simulation data visualization
    with professional aesthetics, automatic steady-state detection, and 
    multidimensional array slicing.
    """

    def __init__(self, results_obj):
        """
        Initialize the Plotter with a Results object.

        :param Results results_obj: The results container object holding the 
                                    simulation history and metadata.
        """
        self.results = results_obj

    def _apply_aesthetics(self):
        """Applies the global plotting configurations defined in GLOBAL_CFG.py."""
        use_seaborn = getattr(cfg, "USE_SEABORN_STYLE", True)
        if use_seaborn:
            sns.set_theme(
                style=getattr(cfg, "SEABORN_THEME", "whitegrid"),
                palette=getattr(cfg, "SEABORN_PALETTE", "colorblind"),
                context=getattr(cfg, "SEABORN_CONTEXT", "notebook"),
            )
        else:
            sns.reset_orig()

    def _resolve_time_index(self, time=None, time_index=-1):
        """Helper to safely extract the index of the requested simulation time."""
        time_grid = self.results.history.index.values
        if time is not None:
            if time < np.min(time_grid) or time > np.max(time_grid):
                raise ValueError(f"Time {time} is out of bounds [{np.min(time_grid)}, {np.max(time_grid)}].")
            return int(np.argmin(np.abs(time_grid - time)))
        return time_index

    def plot(
        self,
        variables=None,
        variable=None,
        domain=None,
        coordinates=None,
        title=None,
        xlabel=None,
        ylabel=None,
        legend_labels=None,
        show=True,
        save_path=None,
        **kwargs,
    ):
        """
        Plots temporal dynamics. Safely combines lumped variables (0D) and 
        phenomenological coordinate extraction for distributed domains (1D, 2D, 3D).
        Auto-detects Steady-State results to plot markers instead of invisible lines.
        """
        self._apply_aesthetics()

        vars_to_plot = []
        auto_legends = {}

        if variables is not None:
            vars_to_plot.extend(variables)

        if variable is not None and domain is not None and coordinates is not None:
            if not getattr(variable, "is_distributed", False):
                raise TypeError(f"Variable '{variable.name}' must be distributed.")

            for coord in coordinates:
                if hasattr(domain, 'x') and hasattr(domain, 'y') and hasattr(domain, 'z'):
                    cx, cy, cz = coord
                    ix = int(np.argmin(np.abs(domain.x.grid - cx)))
                    iy = int(np.argmin(np.abs(domain.y.grid - cy)))
                    iz = int(np.argmin(np.abs(domain.z.grid - cz)))
                    ax, ay, az = domain.x.grid[ix], domain.y.grid[iy], domain.z.grid[iz]
                    node_name = variable.discrete_nodes[ix, iy, iz].name
                    label = f"{domain.name}(X={ax:.2f}, Y={ay:.2f}, Z={az:.2f})"
                
                elif hasattr(domain, 'x') and hasattr(domain, 'y'):
                    cx, cy = coord
                    ix = int(np.argmin(np.abs(domain.x.grid - cx)))
                    iy = int(np.argmin(np.abs(domain.y.grid - cy)))
                    ax, ay = domain.x.grid[ix], domain.y.grid[iy]
                    node_name = variable.discrete_nodes[ix, iy].name
                    label = f"{domain.name}(X={ax:.2f}, Y={ay:.2f})"
                
                elif hasattr(domain, 'grid'):
                    ix = int(np.argmin(np.abs(domain.grid - coord)))
                    ax = domain.grid[ix]
                    node_name = variable.discrete_nodes[ix].name
                    label = f"{domain.name}={ax:.2f} {domain.unit.name}"
                else:
                    raise TypeError("Unsupported domain type for coordinate extraction.")

                vars_to_plot.append(node_name)
                auto_legends[node_name] = label

        if legend_labels:
            auto_legends.update(legend_labels)

        if not vars_to_plot:
            raise ValueError("Provide either 'variables' or ('variable', 'domain', 'coordinates').")

        valid_vars = [v for v in vars_to_plot if v in self.results.history.columns]
        if not valid_vars:
            return None

        df_to_plot = self.results.history[valid_vars]
        if auto_legends:
            df_to_plot = df_to_plot.rename(columns=auto_legends)

        # AUTO STEADY-STATE DETECTION: If only 1 time point, force markers
        if df_to_plot.shape[0] == 1:
            kwargs.setdefault("marker", getattr(cfg, "PLOT_MARKER", "o"))
            kwargs.setdefault("linestyle", "none")
            kwargs.setdefault("markersize", getattr(cfg, "PLOT_MARKERSIZE", 8))

        figsize = kwargs.pop("figsize", getattr(cfg, "PLOT_FIGSIZE", (10, 6)))
        linewidth = kwargs.pop("linewidth", getattr(cfg, "PLOT_LINEWIDTH", 2.5))

        ax = df_to_plot.plot(figsize=figsize, linewidth=linewidth, **kwargs)

        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.05
        margin = margin if margin > 0 else (y_max * 0.05 if y_max != 0 else 0.1)
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_title(title if title else f"Simulation Results: {self.results.name}", fontsize=14, fontweight="bold", pad=15)
        
        indep_var = self.results.history.index.name
        ax.set_xlabel(xlabel if xlabel else f"{indep_var if indep_var else 'Time'} ({self.results.time_units})", fontsize=12)
        if ylabel: ax.set_ylabel(ylabel, fontsize=12)

        if not getattr(cfg, "USE_SEABORN_STYLE", True):
            ax.grid(True, linestyle="--", alpha=0.7)

        ax.legend(fontsize=11, loc="best", frameon=True)
        plt.tight_layout()

        if save_path: plt.savefig(save_path, dpi=getattr(cfg, "PLOT_DPI", 300), bbox_inches="tight")
        if show: plt.show()
        return ax

    def plot_spatial(
        self, variable, domain, time=None, time_index=-1, title=None, 
        xlabel=None, ylabel=None, show=True, save_path=None, **kwargs
    ):
        """Generates a spatial profile plot for a 1D distributed variable."""
        self._apply_aesthetics()

        if not getattr(variable, "is_distributed", False):
            raise TypeError(f"Variable '{variable.name}' is not distributed.")
        if hasattr(domain, 'y'):
            raise TypeError("plot_spatial is for 1D domains. Use plot_heatmap_2d for 2D domains.")

        time_grid = self.results.history.index.values
        indices_to_plot = []

        if time is not None:
            time_list = [time] if not isinstance(time, (list, tuple, np.ndarray)) else time
            for t in time_list:
                indices_to_plot.append(self._resolve_time_index(time=t))
        else:
            idx_list = [time_index] if not isinstance(time_index, (list, tuple, np.ndarray)) else time_index
            indices_to_plot.extend(idx_list)

        node_names = [node.name for node in variable.discrete_nodes]
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", getattr(cfg, "PLOT_FIGSIZE", (10, 6))))

        for idx in indices_to_plot:
            state_data = self.results.history.iloc[idx][node_names].values
            ax.plot(
                domain.grid, state_data, 
                linewidth=kwargs.pop("linewidth", getattr(cfg, "PLOT_LINEWIDTH", 2.5)),
                marker=kwargs.pop("marker", getattr(cfg, "PLOT_MARKER", "o") if len(time_grid) == 1 else None),
                label=f"t = {time_grid[idx]:.1f} {self.results.time_units}",
                **kwargs
            )

        ax.set_title(title if title else f"Spatial Profile: {variable.description}", fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel if xlabel else f"Position {domain.name} ({domain.unit.name})", fontsize=12)
        ax.set_ylabel(ylabel if ylabel else f"{variable.description} ({variable.units.name})", fontsize=12)
        ax.legend(fontsize=11, loc="best", frameon=True)
        plt.tight_layout()

        if save_path: plt.savefig(save_path, dpi=getattr(cfg, "PLOT_DPI", 300), bbox_inches="tight")
        if show: plt.show()
        return ax

    def plot_heatmap_2d(
        self, variable, domain, time=None, time_index=-1, title=None, 
        xlabel=None, ylabel=None, cmap=None, show=True, save_path=None, **kwargs
    ):
        """
        Generates a 2D Contour/Heatmap plot for a distributed variable across a 2D Domain.

        :param Variable variable: The 2D distributed state variable.
        :param Domain2D domain: The spatial domain.
        :param float time: Specific simulation time to plot.
        :param str cmap: Colormap (defaults to GLOBAL_CFG settings based on unit).
        """
        self._apply_aesthetics()

        if not hasattr(domain, 'x') or not hasattr(domain, 'y') or hasattr(domain, 'z'):
            raise TypeError("plot_heatmap_2d requires a Domain2D object.")

        idx = self._resolve_time_index(time, time_index)
        t_val = self.results.history.index.values[idx]

        Nx, Ny = domain.x.n_points, domain.y.n_points
        X, Y = domain.X_grid, domain.Y_grid
        matrix = np.zeros((Nx, Ny))

        # Vectorized extraction
        for i in range(Nx):
            for j in range(Ny):
                matrix[i, j] = self.results.history.iloc[idx][variable.discrete_nodes[i, j].name]

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", getattr(cfg, "PLOT_FIGSIZE", (9, 7))))

        # Auto-select colormap based on variable unit semantics if not provided
        if cmap is None:
            if variable.units.name in ["K", "degC", "J"]: cmap = getattr(cfg, "PLOT_COLORMAP_HEAT", "inferno")
            else: cmap = getattr(cfg, "PLOT_COLORMAP_MASS", "viridis")

        levels = kwargs.pop("levels", getattr(cfg, "PLOT_CONTOUR_LEVELS", 100))
        contour = ax.contourf(X, Y, matrix, levels=levels, cmap=cmap, **kwargs)

        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(f"{variable.description} ({variable.units.name})", fontsize=11)

        final_title = title if title else f"2D Heatmap: {variable.description} (t={t_val} {self.results.time_units})"
        ax.set_title(final_title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel(xlabel if xlabel else f"{domain.x.name} ({domain.x.unit.name})", fontsize=12)
        ax.set_ylabel(ylabel if ylabel else f"{domain.y.name} ({domain.y.unit.name})", fontsize=12)

        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        if save_path: plt.savefig(save_path, dpi=getattr(cfg, "PLOT_DPI", 300), bbox_inches="tight")
        if show: plt.show()
        return ax

    def plot_slice_3d(
        self, variable, domain, slice_axis='z', slice_coord=None, time=None, time_index=-1, 
        title=None, xlabel=None, ylabel=None, cmap=None, show=True, save_path=None, **kwargs
    ):
        """
        Takes a 2D cross-section slice of a 3D volume and renders it as a Heatmap.

        :param Variable variable: The 3D distributed state variable.
        :param Domain3D domain: The 3D spatial domain.
        :param str slice_axis: The axis normal to the slice plane ('x', 'y', or 'z').
        :param float slice_coord: The physical coordinate to slice at. Defaults to the center.
        """
        self._apply_aesthetics()

        if not hasattr(domain, 'z'):
            raise TypeError("plot_slice_3d requires a Domain3D object.")

        idx = self._resolve_time_index(time, time_index)
        t_val = self.results.history.index.values[idx]

        # Determine slicing logic
        axis = slice_axis.lower()
        if axis not in ['x', 'y', 'z']: raise ValueError("slice_axis must be 'x', 'y', or 'z'.")

        target_dom = getattr(domain, axis)
        if slice_coord is None: slice_coord = target_dom.grid[target_dom.n_points // 2]
        
        slice_idx = int(np.argmin(np.abs(target_dom.grid - slice_coord)))
        actual_coord = target_dom.grid[slice_idx]

        # Determine the resulting 2D grids
        if axis == 'x':
            X, Y = domain.y.grid, domain.z.grid
            Nx, Ny = domain.y.n_points, domain.z.n_points
            labels = (domain.y.name, domain.z.name)
        elif axis == 'y':
            X, Y = domain.x.grid, domain.z.grid
            Nx, Ny = domain.x.n_points, domain.z.n_points
            labels = (domain.x.name, domain.z.name)
        else: # z
            X, Y = domain.x.grid, domain.y.grid
            Nx, Ny = domain.x.n_points, domain.y.n_points
            labels = (domain.x.name, domain.y.name)

        X_mesh, Y_mesh = np.meshgrid(X, Y, indexing='ij')
        matrix = np.zeros((Nx, Ny))

        for i in range(Nx):
            for j in range(Ny):
                if axis == 'x': node = variable.discrete_nodes[slice_idx, i, j]
                elif axis == 'y': node = variable.discrete_nodes[i, slice_idx, j]
                else: node = variable.discrete_nodes[i, j, slice_idx]
                matrix[i, j] = self.results.history.iloc[idx][node.name]

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", getattr(cfg, "PLOT_FIGSIZE", (9, 7))))

        if cmap is None:
            if variable.units.name in ["K", "degC", "J"]: cmap = getattr(cfg, "PLOT_COLORMAP_HEAT", "inferno")
            else: cmap = getattr(cfg, "PLOT_COLORMAP_MASS", "viridis")

        levels = kwargs.pop("levels", getattr(cfg, "PLOT_CONTOUR_LEVELS", 100))
        contour = ax.contourf(X_mesh, Y_mesh, matrix, levels=levels, cmap=cmap, **kwargs)

        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(f"{variable.description} ({variable.units.name})", fontsize=11)

        final_title = title if title else f"3D Slice [{axis.upper()}={actual_coord:.2f}]: {variable.description} (t={t_val})"
        ax.set_title(final_title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel(xlabel if xlabel else labels[0], fontsize=12)
        ax.set_ylabel(ylabel if ylabel else labels[1], fontsize=12)

        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        if save_path: plt.savefig(save_path, dpi=getattr(cfg, "PLOT_DPI", 300), bbox_inches="tight")
        if show: plt.show()
        return ax