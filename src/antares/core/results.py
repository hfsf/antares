# -*- coding: utf-8 -*-

"""
Define the Results class for the ANTARES framework.
It acts as a passive data container that receives the full results matrix
from the Simulator (CasADi) and structures it into a pandas DataFrame
for easy plotting and exporting.
"""

import pandas as pd

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError


class Results:
    """
    Definition of the Results container class. Handles post-simulation data.
    """

    def __init__(self, name, description="", time_units=None):
        """
        Instantiates the results container object.

        :param str name: Name of the simulation run.
        :param str description: Description of the run.
        :param str time_units: Units of the independent variable (usually time).
                               Defaults to the global DEFAULT_TIME_UNIT.
        """
        self.name = name
        self.description = description

        # Fallback to the global configuration if no specific unit is provided
        self.time_units = (
            time_units if time_units is not None else cfg.DEFAULT_TIME_UNIT
        )

        # The core of the post-processing phase: a Pandas DataFrame
        self.history = pd.DataFrame()

    def __getitem__(self, var_name):
        """
        Allows direct data access using bracket notation.
        Example: data['temperature_T1']

        :param str var_name: Name of the variable to retrieve.
        :return: A numpy array containing the time series of the requested variable.
        :rtype: numpy.ndarray
        """
        if var_name in self.history.columns:
            return self.history[var_name].values
        else:
            raise UnexpectedValueError(
                f"Variable '{var_name}' not found in the simulation results."
            )

    def load_from_simulator(self, t_span, x_res, z_res, x_names, z_names):
        """
        Receives the raw output matrices from the Simulator and maps them
        to the correct variable names inside the DataFrame.

        :param array-like t_span: Time grid (1D array).
        :param numpy.ndarray x_res: Differential states matrix (Time x Vars).
        :param numpy.ndarray z_res: Algebraic states matrix (Time x Vars).
        :param list x_names: Names of the differential variables.
        :param list z_names: Names of the algebraic variables.
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

        # Set time as the DataFrame index (greatly facilitates native plotting)
        df.set_index("time", inplace=True)
        self.history = df

    def get_variable(self, var_name):
        """
        Retrieves the time series of a specific variable as a numpy array.
        Alternative to the bracket notation (__getitem__).

        :param str var_name: Name of the variable.
        :return: Time series array.
        :rtype: numpy.ndarray
        """
        if var_name in self.history.columns:
            return self.history[var_name].values
        else:
            raise UnexpectedValueError(
                f"Variable '{var_name}' not found in the simulation results."
            )

    def export_to_csv(self, filename):
        """
        Exports the entire simulation history to a CSV file.

        :param str filename: Target path/name for the CSV file.
        """
        self.history.to_csv(filename)

        if cfg.VERBOSITY_LEVEL >= 1:
            print(f"Results successfully exported to: {filename}")
