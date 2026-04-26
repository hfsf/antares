# -*- coding: utf-8 -*-

"""
Thermodynamic Package Module (V5 Native CasADi Architecture).

This module implements the SOLID principles by decoupling the thermodynamic
data retrieval and mathematical graph generation from the topological flow
entities (Streams).

It defines 'Property Packages' that act as containers. These packages handle:
1. Interfacing with external libraries (e.g., `thermo`) during instantiation
   to extract critical constants or generate grids.
2. Generating the native CasADi (C++) symbolic expressions for thermodynamic
   properties (like Enthalpy) to be injected into the main system matrix.
   Note: Empirical calculations temporarily bypass the Dimensional Guardian
   by operating directly on CasADi MX variables, and are subsequently re-wrapped
   with standard physical units.
"""

from abc import ABC, abstractmethod

import casadi as ca
import numpy as np

from antares.core.expression_evaluation import EquationNode
from antares.core.unit import Unit

try:
    import thermo
    from thermo.chemical import Chemical

    HAS_THERMO = True
except ImportError:
    HAS_THERMO = False


class PropertyPackage(ABC):
    """
    Abstract Base Class for all Thermodynamic Property Packages.
    Enforces the implementation of graph-generating methods for physical properties.
    """

    def __init__(self, components):
        """
        :param list components: List of string names for chemical components.
        """
        self.components = components

    @abstractmethod
    def get_enthalpy_expression(self, T_sym, P_sym, z_sym_dict):
        """
        Abstract method to generate the CasADi symbolic expression for mixture enthalpy.

        :param EquationNode T_sym: Node for Temperature (K).
        :param EquationNode P_sym: Node for Pressure (bar).
        :param dict z_sym_dict: Dictionary mapping component names to their molar fractions.
        :return: EquationNode representing Molar Enthalpy (J/mol).
        :rtype: EquationNode
        """
        pass


class PureFluidLUT(PropertyPackage):
    """
    Lookup Table (LUT) Property Package for Pure Fluids.

    Ideal for utility fluids (water, steam, refrigerants). It interfaces with
    the `thermo` library at initialization to build a dense property grid,
    and returns a high-performance C++ B-Spline interpolant for the solver.
    """

    def __init__(
        self, fluid_name, T_bounds=(273.15, 600.0), P_bounds=(1.0, 10.0), grid_size=20
    ):
        """
        Instantiates the LUT package and fetches data from the `thermo` library.

        :param str fluid_name: Standard chemical name recognized by `thermo` (e.g., 'water').
        :param tuple T_bounds: (Min, Max) temperature for the grid in Kelvin.
        :param tuple P_bounds: (Min, Max) pressure for the grid in bar.
        :param int grid_size: Number of points per axis for the mesh grid.
        :raises ImportError: If the `thermo` library is not installed.
        """
        if not HAS_THERMO:
            raise ImportError(
                "The 'thermo' library is required to build a PureFluidLUT. Install it via pip."
            )

        super().__init__([fluid_name])
        self.fluid_name = fluid_name

        # Generate linear grids
        self.T_grid = np.linspace(T_bounds[0], T_bounds[1], grid_size).tolist()
        self.P_grid_bar = np.linspace(P_bounds[0], P_bounds[1], grid_size).tolist()
        P_grid_pa = [p * 1e5 for p in self.P_grid_bar]

        # Pre-allocate Enthalpy matrix
        self.H_matrix = np.zeros((grid_size, grid_size))

        print(
            f"[{self.__class__.__name__}] Generating B-Spline mesh for '{fluid_name}'... Please wait."
        )
        chem = Chemical(fluid_name)
        for i, T in enumerate(self.T_grid):
            for j, P in enumerate(P_grid_pa):
                chem.calculate(T, P)
                self.H_matrix[i, j] = chem.H  # Absolute Enthalpy in J/mol

        # Flatten in Fortran order (column-major) as required by CasADi B-Splines
        self._h_flat = self.H_matrix.ravel(order="F")

        # Pre-compile the CasADi interpolant object for efficiency
        self.interpolant = ca.interpolant(
            f"H_LUT_{self.fluid_name}",
            "bspline",
            [self.T_grid, self.P_grid_bar],
            self._h_flat,
        )
        print(f"[{self.__class__.__name__}] Data successfully cached.")

    def get_enthalpy_expression(self, T_sym, P_sym, z_sym_dict):
        """
        Evaluates the CasADi B-Spline object symbolically.

        :param EquationNode T_sym: Temperature node.
        :param EquationNode P_sym: Pressure node.
        :param dict z_sym_dict: Fractions (ignored for pure fluids).
        :return: Evaluated EquationNode for Enthalpy.
        :rtype: EquationNode
        """
        # Strip units to interact directly with the C++ CasADi Engine
        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        P_mx = P_sym.symbolic_object if hasattr(P_sym, "symbolic_object") else P_sym

        # Evaluate the interpolant
        h_mx = self.interpolant(ca.vertcat(T_mx, P_mx))

        # Re-wrap in the dimensional Guardian structure
        return EquationNode(
            name=f"H_LUT_{self.fluid_name}",
            symbolic_object=h_mx,
            unit_object=Unit("", "J/mol"),
        )


class MulticomponentEOS(PropertyPackage):
    """
    Equation of State (EOS) Property Package for complex mixtures.

    Interfaces with the `thermo` library at initialization to extract fundamental
    constants (Tc, Pc, Standard Enthalpy of Formation, Ideal Gas Heat Capacity).
    It then builds analytical thermodynamic mixing rules directly into the
    CasADi computational graph, providing exact analytical derivatives.
    """

    def __init__(self, components):
        """
        Instantiates the EOS package.

        :param list components: List of standard chemical names (e.g., ['water', 'ethanol']).
        """
        super().__init__(components)
        self.thermo_data = {}

    def fetch_parameters_from_db(self):
        """
        Invokes the external `thermo` library to automatically fetch and
        populate the fundamental constants for all components.

        :raises ImportError: If the `thermo` library is not installed.
        :raises ValueError: If a component is not found in the database.
        """
        if not HAS_THERMO:
            raise ImportError(
                "The 'thermo' library is required to build a MulticomponentEOS. Install it via pip."
            )

        for comp in self.components:
            try:
                chem = Chemical(comp)
            except Exception as e:
                raise ValueError(
                    f"Component '{comp}' not found in the thermo database."
                ) from e

            # Heuristic to extract generic polynomial coefficients for Ideal Gas Cp
            try:
                coeffs = chem.HeatCapacityGas.models[0].coeffs
                if len(coeffs) < 4:
                    coeffs = coeffs + [0.0] * (4 - len(coeffs))
            except (AttributeError, IndexError):
                coeffs = [0.0, 0.0, 0.0, 0.0]

            self.thermo_data[comp] = {
                "Hf_298": getattr(chem, "Hfgm", 0.0),
                "Tc": getattr(chem, "Tc", 300.0),
                "Pc": getattr(chem, "Pc", 1e5),
                "Cp_coeffs": coeffs[:4],
            }

        print(
            f"[{self.__class__.__name__}] Sucessfully loaded thermodinamical data for: {self.components}"
        )

    def get_enthalpy_expression(self, T_sym, P_sym, z_sym_dict):
        """
        Builds the analytical Ideal Mixture Enthalpy polynomial in CasADi.

        :param EquationNode T_sym: Temperature node.
        :param EquationNode P_sym: Pressure node.
        :param dict z_sym_dict: Dictionary of symbolic molar fractions.
        :return: Analytical polynomial expression for mixture Enthalpy.
        :rtype: EquationNode
        """
        h_mix_sym = 0.0
        T_ref = 298.15  # K

        # Strip units to prevent the Guardian from rejecting empirical polynomial additions
        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym

        for comp in self.components:
            data = self.thermo_data[comp]
            Hf_298 = data["Hf_298"]
            A, B, C, D = data["Cp_coeffs"]

            # Analytical integration: Integral(Cp dT) from T_ref to T_sym
            integral_Cp = (
                A * (T_mx - T_ref)
                + (B / 2.0) * (T_mx**2 - T_ref**2)
                + (C / 3.0) * (T_mx**3 - T_ref**3)
                + (D / 4.0) * (T_mx**4 - T_ref**4)
            )

            h_comp_pure = Hf_298 + integral_Cp

            # Ideal mixing rule based on molar fractions
            z_node = z_sym_dict[comp]
            z_mx = (
                z_node.symbolic_object if hasattr(z_node, "symbolic_object") else z_node
            )

            h_mix_sym += z_mx * h_comp_pure

        # Re-wrap in the dimensional Guardian structure with the physical unit
        return EquationNode(
            name="H_mix_eos", symbolic_object=h_mix_sym, unit_object=Unit("", "J/mol")
        )
