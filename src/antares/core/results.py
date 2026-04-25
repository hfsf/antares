# -*- coding: utf-8 -*-

"""
Results Module (V5 Native CasADi Architecture).

Defines the Results class for the ANTARES framework.
It acts as a passive data container that receives the full numerical results
matrices from the Simulator (CasADi C++ backend) and structures them into a
pandas DataFrame for streamlined plotting, analysis, and exporting.
In the V5 architecture, this module actively reconstructs spatial tensors 
by hunting down vector shards (e.g., _x_0, _DomainName_0, [0]) expanded by the 
CasADi compiler and stitching them back into their precise topological coordinates.
"""

import numpy as np
import pandas as pd

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError


class Results:
    """
    Results container class definition.
    Handles post-simulation data aggregation, structuring time-series data
    into a cohesive Pandas DataFrame and dynamically reassembling complex 
    spatial domains (1D, 2D, 3D).
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

        self.time_units = (
            time_units
            if time_units is not None
            else getattr(cfg, "DEFAULT_TIME_UNIT", "s")
        )

        self.history = pd.DataFrame()

    def __getitem__(self, var_identifier):
        """
        Allows direct data access using bracket notation (e.g., data['temperature'] or data[model.T]).
        Delegates the logic directly to get_variable() to ensure proper spatial reconstruction.

        :param var_identifier: The exact string name or the Variable object itself.
        :type var_identifier: str or Variable
        :return: A numpy array containing the time series of the requested variable.
        :rtype: numpy.ndarray
        :raises UnexpectedValueError: If the variable name does not exist in the history.
        """
        return self.get_variable(var_identifier)

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
        df = pd.DataFrame({"time": t_span})

        if x_res is not None and len(x_names) > 0:
            df_x = pd.DataFrame(x_res, columns=x_names)
            df = pd.concat([df, df_x], axis=1)

        if z_res is not None and len(z_names) > 0:
            df_z = pd.DataFrame(z_res, columns=z_names)
            df = pd.concat([df, df_z], axis=1)

        df.set_index("time", inplace=True)
        self.history = df

    def get_variable(self, var_identifier):
        """
        Retrieves the time series of a specific variable as a numpy array.
        In the V5 Architecture, it automatically detects if the variable was 
        shattered into indexed CasADi arrays and stitches them back into a 
        unified N-Dimensional spatial tensor.

        :param var_identifier: The exact string name or the Variable object.
        :type var_identifier: str or Variable
        :return: A 1D (lumped) or 2D (distributed, Time x Nodes) numpy array.
        :rtype: numpy.ndarray
        :raises UnexpectedValueError: If the variable shards cannot be located.
        """
        var_name = getattr(var_identifier, "name", str(var_identifier))

        # 1. Exact Match (Lumped or already stitched by simulator)
        if var_name in self.history.columns:
            val = self.history[var_name].values
            if len(val) > 0 and isinstance(val[0], (list, np.ndarray)):
                return np.stack(val)
            return val

        # Filter all columns that contain the base variable name
        related_cols = [str(c) for c in self.history.columns if var_name in str(c)]
        is_dist = hasattr(var_identifier, "is_distributed") and getattr(var_identifier, "is_distributed")

        # 2. Distributed Variable Topological Stitching
        if is_dist:
            time_steps = len(self.history)
            n_points = getattr(var_identifier, "n_points", 0)
            
            if n_points == 0:
                raise UnexpectedValueError(f"Distributed variable '{var_name}' lacks geometric n_points.")

            stitched = np.zeros((time_steps, n_points))
            idx_diff = getattr(var_identifier, "idx_diff", [])
            idx_alg = getattr(var_identifier, "idx_alg", [])

            # 2.1 - Check if the simulator preserved whole arrays as _x and _z
            if f"{var_name}_x" in related_cols or f"{var_name}_z" in related_cols:
                has_x = f"{var_name}_x" in related_cols
                has_z = f"{var_name}_z" in related_cols

                if has_x:
                    x_vals = self.history[f"{var_name}_x"].values
                    x_vals = np.stack(x_vals) if len(x_vals) > 0 and isinstance(x_vals[0], (list, np.ndarray)) else x_vals.reshape(-1, 1)
                    stitched[:, idx_diff] = x_vals

                if has_z:
                    z_vals = self.history[f"{var_name}_z"].values
                    z_vals = np.stack(z_vals) if len(z_vals) > 0 and isinstance(z_vals[0], (list, np.ndarray)) else z_vals.reshape(-1, 1)
                    stitched[:, idx_alg] = z_vals

                return stitched

            # 2.2 - Deep Search: Hunt for shattered nodes with dynamic domain names
            found_any = False
            
            # Combine all flat indices needed to reconstruct the full spatial domain
            all_indices = list(idx_diff) + list(idx_alg)

            for flat_i in all_indices:
                matched_col = None
                for col in related_cols:
                    # Matches any variation generated by the Transpiler/Simulator
                    # e.g.: C_Modelo_cilindro_0, C_Modelo_cilindro_Raio_0, C_Modelo_cilindro[0]
                    if col == f"{var_name}_{flat_i}" or \
                       col == f"{var_name}_x_{flat_i}" or \
                       col == f"{var_name}_z_{flat_i}" or \
                       col == f"{var_name}[{flat_i}]" or \
                       (col.startswith(f"{var_name}_") and col.endswith(f"_{flat_i}")):
                        matched_col = col
                        break
                
                if matched_col:
                    stitched[:, flat_i] = self.history[matched_col].values
                    found_any = True

            if found_any:
                return stitched

        # 3. Fallback for Lumped Variables named with suffixes
        if f"{var_name}_x" in related_cols:
            return self.history[f"{var_name}_x"].values
        if f"{var_name}_z" in related_cols:
            return self.history[f"{var_name}_z"].values

        # THE ULTIMATE DIAGNOSTIC NET
        raise UnexpectedValueError(
            f"Variable '{var_name}' could not be reconstructed topologically.\n"
            f"-> Related columns found: {related_cols}\n"
            f"-> ALL available columns exported by Simulator: {list(self.history.columns)}"
        )

    def export_to_csv(self, filename):
        """
        Exports the entire structured simulation history to a CSV file.

        :param str filename: Target system path and name for the output CSV file.
        """
        self.history.to_csv(filename)

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"Results successfully exported to: {filename}")