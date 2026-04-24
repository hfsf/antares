# -*- coding: utf-8 -*-

"""
Results Module (V5 Native CasADi Architecture).

Defines the Results class for the ANTARES framework.
It acts as a passive data container that receives the full numerical results
matrices from the Simulator (CasADi C++ backend) and structures them into a
pandas DataFrame for streamlined plotting, analysis, and exporting.
"""

import numpy as np
import pandas as pd

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError


class Results:
    """
    Results container class definition.
    Handles post-simulation data aggregation, structuring time-series data
    into a cohesive Pandas DataFrame.
    """

    def __init__(self, name, description="", time_units=None):
        """
        Instantiates the Results container object.

        :param str name: Internal name of the simulation run.
        :param str description: Physical or contextual description of the run.
        :param str time_units: Units of the independent variable (typically time).
                               Falls back to GLOBAL_CFG.DEFAULT_TIME_UNIT if None.
        """
        self.name = name
        self.description = description

        # Fallback to the global configuration if no specific unit is provided
        self.time_units = (
            time_units
            if time_units is not None
            else getattr(cfg, "DEFAULT_TIME_UNIT", "s")
        )

        # The core of the post-processing phase: a Pandas DataFrame
        self.history = pd.DataFrame()

    def __getitem__(self, var_name):
        """
        Allows direct data access using bracket notation (e.g., data['temperature']).

        :param str var_name: The exact string name of the tracked variable.
        :return: A 1D numpy array containing the time series of the requested variable.
        :rtype: numpy.ndarray
        :raises UnexpectedValueError: If the variable name does not exist in the history.
        """
        if var_name in self.history.columns:
            return self.history[var_name].values

        raise UnexpectedValueError(
            f"Variable '{var_name}' not found in the simulation results."
        )

    def load_from_simulator(self, t_span, x_res, z_res, x_names, z_names):
        """
        Receives the raw output matrices from the dynamic integrators or rootfinders
        and maps them to the correct variable names inside the DataFrame.

        :param array-like t_span: The evaluated time grid (1D array).
        :param numpy.ndarray x_res: Evaluated differential states matrix (Time x Vars).
        :param numpy.ndarray z_res: Evaluated algebraic states matrix (Time x Vars).
        :param list x_names: Ordered list of string names for the differential variables.
        :param list z_names: Ordered list of string names for the algebraic variables.
        """
        # Create the base DataFrame with the time vector
        df = pd.DataFrame({"time": t_span})

        # Append differential variables (x)
        if x_res is not None and len(x_names) > 0:
            df_x = pd.DataFrame(x_res, columns=x_names)
            df = pd.concat([df, df_x], axis=1)

        # Append algebraic variables (z)
        if z_res is not None and len(z_names) > 0:
            df_z = pd.DataFrame(z_res, columns=z_names)
            df = pd.concat([df, df_z], axis=1)

        # Set time as the DataFrame index (greatly facilitates native pandas plotting)
        df.set_index("time", inplace=True)
        self.history = df

    def get_variable(self, var_name):
        """
        Retrieves the time series of a specific variable as a numpy array.
        Provides an explicit method alternative to the bracket notation (__getitem__).

        :param str var_name: The exact string name of the tracked variable.
        :return: A 1D numpy array containing the time series.
        :rtype: numpy.ndarray
        :raises UnexpectedValueError: If the variable name does not exist in the history.
        """
        if var_name in self.history.columns:
            return self.history[var_name].values

        raise UnexpectedValueError(
            f"Variable '{var_name}' not found in the simulation results."
        )

    def export_to_csv(self, filename):
        """
        Exports the entire structured simulation history to a CSV file.

        :param str filename: Target system path and name for the output CSV file.
        """
        self.history.to_csv(filename)

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"Results successfully exported to: {filename}")
