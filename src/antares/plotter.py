# -*- coding: utf-8 -*-

"""
Define Plotter class.
Provides high-level plotting utilities for the ANTARES framework.
Interacts directly with the Results object to generate publication-quality
visualizations using matplotlib, pandas, and seaborn integration.
"""

import warnings

import matplotlib.pyplot as plt
import seaborn as sns

import antares.core.GLOBAL_CFG as cfg


class Plotter:
    """
    Definition of Plotter class. Handles simulation data visualization
    with professional aesthetics.
    """

    def __init__(self, results_obj):
        """
        Initialize the Plotter with a Results object.

        :param Results results_obj:
            The results container object holding the simulation history
            and metadata.
        """
        self.results = results_obj

    def _apply_aesthetics(self):
        """
        Applies the global plotting configurations defined in GLOBAL_CFG.py.
        Uses defensively getattr to avoid crashing if older configs are used.
        """
        use_seaborn = getattr(cfg, "USE_SEABORN_STYLE", True)

        if use_seaborn:
            # Aplica a magia do Seaborn com os parâmetros globais
            sns.set_theme(
                style=getattr(cfg, "SEABORN_THEME", "whitegrid"),
                palette=getattr(cfg, "SEABORN_PALETTE", "colorblind"),
                context=getattr(cfg, "SEABORN_CONTEXT", "notebook"),
            )
        else:
            # Reseta para os padrões originais do Matplotlib
            sns.reset_orig()

    def plot(
        self,
        variables,
        title=None,
        xlabel=None,
        ylabel=None,
        legend_labels=None,
        show=True,
        save_path=None,
    ):
        """
        Plots a list of variables against the simulation time index.

        :param list variables: List of strings with variable names.
        :param str title: Optional title for the figure.
        :param str xlabel: Custom label for the X-axis. If None, infers from data.
        :param str ylabel: Custom label for the Y-axis.
        :param dict legend_labels: Dictionary to rename variables in the legend.
        :param bool show: Whether to trigger the interactive plot window.
        :param str save_path: If provided, saves the resulting figure.
        :return ax: The matplotlib Axes object.
        """
        # 1. Aplicar a estética ANTES de criar a figura
        self._apply_aesthetics()

        vars_to_plot = [v for v in variables]

        # Validation: Check if requested variables exist
        for var in variables:
            if var not in self.results.history.columns:
                if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                    warnings.warn(
                        f"Variable '{var}' not found in simulation results. Skipping."
                    )
                vars_to_plot.remove(var)

        if not vars_to_plot:
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print("Plotter Error: No valid variables were found to plot.")
            return None

        # Configurações globais
        figsize = getattr(cfg, "PLOT_FIGSIZE", (10, 6))
        linewidth = getattr(cfg, "PLOT_LINEWIDTH", 2.5)

        # 2. Preparação de Dados e Renomeação Dinâmica
        df_to_plot = self.results.history[vars_to_plot]
        if legend_labels:
            df_to_plot = df_to_plot.rename(columns=legend_labels)

        # 3. Renderização
        ax = df_to_plot.plot(figsize=figsize, linewidth=linewidth)

        # Adiciona 5% de margem no eixo Y
        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.05
        # Previne erro se a linha for perfeitamente horizontal e a margem for 0
        margin = margin if margin > 0 else (y_max * 0.05 if y_max != 0 else 0.1)
        ax.set_ylim(y_min - margin, y_max + margin)

        # 4. Estilização Visual Adicional
        final_title = title if title else f"Simulation Results: {self.results.name}"
        ax.set_title(final_title, fontsize=14, fontweight="bold", pad=15)

        # Determinação Inteligente do Eixo X
        if xlabel is not None:
            final_xlabel = xlabel
        else:
            indep_var_name = self.results.history.index.name
            if indep_var_name:
                final_xlabel = f"{indep_var_name} ({self.results.time_units})"
            else:
                final_xlabel = f"[{self.results.time_units}]"

        ax.set_xlabel(final_xlabel, fontsize=12, labelpad=10)

        # Só exibe a label Y se ela for fornecida
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

        # Fallback de grelha se o Seaborn estiver desligado
        if not getattr(cfg, "USE_SEABORN_STYLE", True):
            ax.grid(True, linestyle="--", alpha=0.7)

        ax.legend(fontsize=11, loc="best", frameon=True)
        plt.tight_layout()

        # 5. Exportação
        if save_path:
            dpi = getattr(cfg, "PLOT_DPI", 300)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print(f"Figure successfully saved to: {save_path}")

        # 6. Visualização
        if show:
            plt.show()

        return ax
