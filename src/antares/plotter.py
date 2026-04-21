# -*- coding: utf-8 -*-

"""
Plotter Module.

Provides high-level plotting utilities for the ANTARES framework.
Interacts directly with the Results object to generate publication-quality
visualizations using matplotlib, pandas, and seaborn integration.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import antares.core.GLOBAL_CFG as cfg


class Plotter:
    """
    Definition of Plotter class. Handles simulation data visualization
    with professional aesthetics and customizable parameters.
    """

    def __init__(self, results_obj):
        """
        Initialize the Plotter with a Results object.

        :param Results results_obj: The results container object holding the 
                                    simulation history and metadata.
        """
        self.results = results_obj

    def _apply_aesthetics(self):
        """
        Applies the global plotting configurations defined in GLOBAL_CFG.py.
        """
        use_seaborn = getattr(cfg, "USE_SEABORN_STYLE", True)
        if use_seaborn:
            sns.set_theme(
                style=getattr(cfg, "SEABORN_THEME", "whitegrid"),
                palette=getattr(cfg, "SEABORN_PALETTE", "colorblind"),
                context=getattr(cfg, "SEABORN_CONTEXT", "notebook"),
            )
        else:
            sns.reset_orig()

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

        :param list variables: List of lumped variable names to plot (e.g., [tanque.h.name]).
        :param Variable variable: The distributed state variable to extract points from.
        :param Domain domain: The spatial domain (Domain1D, Domain2D, or Domain3D).
        :param list coordinates: List of floats (1D) or tuples (2D/3D) representing physical coordinates.
        :param str title: Optional title for the figure.
        :param str xlabel: Custom label for the X-axis.
        :param str ylabel: Custom label for the Y-axis.
        :param dict legend_labels: Dictionary to rename specific variable lines in the legend.
        :param bool show: Whether to trigger the interactive plot window.
        :param str save_path: If provided, saves the resulting figure.
        :return: The matplotlib Axes object.
        :rtype: matplotlib.axes.Axes
        """
        self._apply_aesthetics()

        vars_to_plot = []
        auto_legends = {}

        # 1. Adds Lumped/Direct Variables if provided
        if variables is not None:
            vars_to_plot.extend(variables)

        # 2. Automatically extracts N-Dimensional nodes based on physical coordinates
        if variable is not None and domain is not None and coordinates is not None:
            if not getattr(variable, "is_distributed", False):
                raise TypeError(f"Variable '{variable.name}' must be distributed to plot by coordinates.")

            for coord in coordinates:
                # 3D Domain Extraction
                if hasattr(domain, 'x') and hasattr(domain, 'y') and hasattr(domain, 'z'):
                    cx, cy, cz = coord
                    ix = int(np.argmin(np.abs(domain.x.grid - cx)))
                    iy = int(np.argmin(np.abs(domain.y.grid - cy)))
                    iz = int(np.argmin(np.abs(domain.z.grid - cz)))
                    ax, ay, az = domain.x.grid[ix], domain.y.grid[iy], domain.z.grid[iz]
                    node_name = variable.discrete_nodes[ix, iy, iz].name
                    label = f"{domain.name}(X={ax:.2f}, Y={ay:.2f}, Z={az:.2f})"
                
                # 2D Domain Extraction
                elif hasattr(domain, 'x') and hasattr(domain, 'y'):
                    cx, cy = coord
                    ix = int(np.argmin(np.abs(domain.x.grid - cx)))
                    iy = int(np.argmin(np.abs(domain.y.grid - cy)))
                    ax, ay = domain.x.grid[ix], domain.y.grid[iy]
                    node_name = variable.discrete_nodes[ix, iy].name
                    label = f"{domain.name}(X={ax:.2f}, Y={ay:.2f})"
                
                # 1D Domain Extraction
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
            raise ValueError("Plotter Error: Provide either 'variables' or ('variable', 'domain', 'coordinates').")

        # VALIDATION & RENDERING
        valid_vars = []
        for var in vars_to_plot:
            if var not in self.results.history.columns:
                if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                    warnings.warn(f"Variable '{var}' not found in simulation results. Skipping.")
            else:
                valid_vars.append(var)

        if not valid_vars:
            return None

        figsize = kwargs.pop("figsize", getattr(cfg, "PLOT_FIGSIZE", (10, 6)))
        linewidth = kwargs.pop("linewidth", getattr(cfg, "PLOT_LINEWIDTH", 2.5))

        df_to_plot = self.results.history[valid_vars]
        if auto_legends:
            df_to_plot = df_to_plot.rename(columns=auto_legends)

        ax = df_to_plot.plot(figsize=figsize, linewidth=linewidth, **kwargs)

        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.05
        margin = margin if margin > 0 else (y_max * 0.05 if y_max != 0 else 0.1)
        ax.set_ylim(y_min - margin, y_max + margin)

        final_title = title if title else f"Simulation Results: {self.results.name}"
        ax.set_title(final_title, fontsize=14, fontweight="bold", pad=15)

        if xlabel is not None:
            final_xlabel = xlabel
        else:
            indep_var_name = self.results.history.index.name
            final_xlabel = f"{indep_var_name} ({self.results.time_units})" if indep_var_name else f"[{self.results.time_units}]"

        ax.set_xlabel(final_xlabel, fontsize=12, labelpad=10)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

        if not getattr(cfg, "USE_SEABORN_STYLE", True):
            ax.grid(True, linestyle="--", alpha=0.7)

        ax.legend(fontsize=11, loc="best", frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=getattr(cfg, "PLOT_DPI", 300), bbox_inches="tight")

        if show:
            plt.show()

        return ax

    def plot_spatial(
        self,
        variable,
        domain,
        time=None,
        time_index=-1,
        title=None,
        xlabel=None,
        ylabel=None,
        show=True,
        save_path=None,
        **kwargs,
    ):
        """
        Generates a spatial profile plot for a distributed variable along a given domain.
        Supports single or multiple time instances (superposition).

        :param Variable variable: The distributed state variable to plot.
        :param Domain domain: The spatial domain.
        :param float/list time: Exact simulation time(s) to plot (Overrides time_index).
        :param int/list time_index: Specific time step array index(es). Defaults to -1.
        :return: The matplotlib Axes object.
        """
        self._apply_aesthetics()

        if not getattr(variable, "is_distributed", False):
            raise TypeError(f"Variable '{variable.name}' is not distributed.")

        time_grid = self.results.history.index.values

        # 1. Resolve Time Indices (Handling Lists)
        indices_to_plot = []
        if time is not None:
            time_list = [time] if not isinstance(time, (list, tuple, np.ndarray)) else time
            for t in time_list:
                if t < np.min(time_grid) or t > np.max(time_grid):
                    raise ValueError(f"Time {t} is out of bounds [{np.min(time_grid)}, {np.max(time_grid)}].")
                idx = int(np.argmin(np.abs(time_grid - t)))
                indices_to_plot.append(idx)
        else:
            idx_list = [time_index] if not isinstance(time_index, (list, tuple, np.ndarray)) else time_index
            indices_to_plot.extend(idx_list)

        node_names = [node.name for node in variable.discrete_nodes]

        # 2. Setup Figure
        figsize = kwargs.pop("figsize", getattr(cfg, "PLOT_FIGSIZE", (10, 6)))
        linewidth = kwargs.pop("linewidth", getattr(cfg, "PLOT_LINEWIDTH", 2.5))
        marker = kwargs.pop("marker", getattr(cfg, "PLOT_MARKER", "o"))
        markersize = kwargs.pop("markersize", getattr(cfg, "PLOT_MARKERSIZE", 4))
        color = kwargs.pop("color", None) # Let seaborn cycle colors if multiple lines
        
        fig, ax = plt.subplots(figsize=figsize)

        # 3. Plot Iteratively (Superposition)
        for idx in indices_to_plot:
            state_data = self.results.history.iloc[idx][node_names].values
            t_val = time_grid[idx]
            label = f"t = {t_val:.1f} {self.results.time_units}"

            plot_kwargs = {"linewidth": linewidth, "marker": marker, "markersize": markersize, "label": label}
            if color and len(indices_to_plot) == 1:
                plot_kwargs["color"] = color

            ax.plot(domain.grid, state_data, **plot_kwargs, **kwargs)

        # 4. Labels & Titles
        final_title = title if title else f"Spatial Profile: {variable.description}"
        ax.set_title(final_title, fontsize=14, fontweight="bold", pad=15)

        ax.set_xlabel(xlabel if xlabel else f"Position {domain.name} ({domain.unit.name})", fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else f"{variable.description} ({variable.units.name})", fontsize=12, labelpad=10)

        if not getattr(cfg, "USE_SEABORN_STYLE", True):
            ax.grid(True, linestyle="--", alpha=0.7)

        ax.legend(fontsize=11, loc="best", frameon=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=getattr(cfg, "PLOT_DPI", 300), bbox_inches="tight")

        if show:
            plt.show()

        return ax

    def plot_surface(
        self,
        variable,
        domain,
        title=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        cmap="viridis",
        show=True,
        save_path=None,
    ):
        """
        Generates a 3D Surface Plot for a 1D distributed variable over time.
        (X = Time, Y = Space, Z = State Variable).

        :param Variable variable: The distributed state variable.
        :param Domain domain: The spatial domain (1D).
        :param str cmap: Matplotlib colormap string (e.g., 'viridis', 'plasma', 'inferno').
        :return: The matplotlib Axes3D object.
        """
        self._apply_aesthetics()

        if not getattr(variable, "is_distributed", False):
            raise TypeError(f"Variable '{variable.name}' is not distributed.")

        # 1. Extract Data Grids
        time_grid = self.results.history.index.values
        space_grid = domain.grid
        node_names = [node.name for node in variable.discrete_nodes]

        # Ensure we have all nodes
        for name in node_names:
            if name not in self.results.history.columns:
                raise KeyError(f"Node '{name}' missing from results.")

        # Extract the full 2D matrix (Rows = Time, Cols = Space)
        data_matrix = self.results.history[node_names].values

        # 2. Create Meshgrids
        T_mesh, S_mesh = np.meshgrid(time_grid, space_grid, indexing='ij')

        # 3. Render 3D Surface
        figsize = getattr(cfg, "PLOT_FIGSIZE", (10, 6))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            T_mesh, S_mesh, data_matrix, cmap=cmap, edgecolor='none', alpha=0.9
        )

        # 4. Aesthetics
        final_title = title if title else f"Spatiotemporal Surface: {variable.description}"
        ax.set_title(final_title, fontsize=14, fontweight="bold", pad=20)

        t_unit = self.results.time_units
        ax.set_xlabel(xlabel if xlabel else f"Time ({t_unit})", fontsize=11, labelpad=10)
        ax.set_ylabel(ylabel if ylabel else f"Position {domain.name} ({domain.unit.name})", fontsize=11, labelpad=10)
        ax.set_zlabel(zlabel if zlabel else f"{variable.description} ({variable.units.name})", fontsize=11, labelpad=10)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=0.8, pad=0.1, label=variable.units.name)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=getattr(cfg, "PLOT_DPI", 300), bbox_inches="tight")

        if show:
            plt.show()

        return ax