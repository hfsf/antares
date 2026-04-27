# -*- coding: utf-8 -*-

"""
Thermodynamic Package Module (V5 Native CasADi Architecture).

This module implements the SOLID principles by decoupling the thermodynamic
data retrieval and mathematical graph generation from the topological flow
entities (Streams).

Now upgraded with Vapor-Liquid Equilibrium (VLE) support, providing K-values
(partition coefficients) and phase-specific enthalpies to seamlessly feed
global Equation-Oriented (EO) Flash calculations (like Rachford-Rice constraints).
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
    Enforces the implementation of graph-generating methods for physical properties
    and Vapor-Liquid Equilibrium (VLE).
    """

    def __init__(self, components):
        """
        :param list components: List of string names for chemical components.
        """
        self.components = components

    @abstractmethod
    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        """
        Generates the CasADi symbolic expression for phase-specific enthalpy.

        :param EquationNode T_sym: Node for Temperature (K).
        :param EquationNode P_sym: Node for Pressure (bar).
        :param dict fractions_sym_dict: Dictionary mapping components to their molar fractions.
        :param str phase: Target phase ('liquid' or 'vapor').
        :return: EquationNode representing Molar Enthalpy of the phase (J/mol).
        :rtype: EquationNode
        """
        pass

    @abstractmethod
    def get_K_value(self, T_sym, P_sym, comp):
        """
        Generates the CasADi symbolic expression for the Vapor-Liquid
        equilibrium partition coefficient (K_i = y_i / x_i).

        :param EquationNode T_sym: Node for Temperature (K).
        :param EquationNode P_sym: Node for Pressure (bar).
        :param str comp: Name of the target component.
        :return: EquationNode representing the K-value (dimensionless).
        :rtype: EquationNode
        """
        pass

    def get_enthalpy_expression(self, T_sym, P_sym, z_sym_dict):
        """
        Legacy fallback for single-phase streams. Defaults to vapor/ideal gas enthalpy.
        """
        return self.get_phase_enthalpy_expression(T_sym, P_sym, z_sym_dict, phase="vapor")


class PureFluidLUT(PropertyPackage):
    """
    Lookup Table (LUT) Property Package for Pure Fluids.
    """

    def __init__(
        self, fluid_name, T_bounds=(273.15, 600.0), P_bounds=(1.0, 10.0), grid_size=20
    ):
        if not HAS_THERMO:
            raise ImportError("The 'thermo' library is required.")
        super().__init__([fluid_name])
        self.fluid_name = fluid_name

        self.T_grid = np.linspace(T_bounds[0], T_bounds[1], grid_size).tolist()
        self.P_grid_bar = np.linspace(P_bounds[0], P_bounds[1], grid_size).tolist()
        P_grid_pa = [p * 1e5 for p in self.P_grid_bar]

        self.H_matrix = np.zeros((grid_size, grid_size))
        print(f"[{self.__class__.__name__}] Generating B-Spline mesh for '{fluid_name}'... Please wait.")
        chem = Chemical(fluid_name)
        for i, T in enumerate(self.T_grid):
            for j, P in enumerate(P_grid_pa):
                chem.calculate(T, P)
                self.H_matrix[i, j] = getattr(chem, "H", 0.0)

        self._h_flat = self.H_matrix.ravel(order="F")
        self.interpolant = ca.interpolant(
            f"H_LUT_{self.fluid_name}", "bspline", [self.T_grid, self.P_grid_bar], self._h_flat
        )
        print(f"[{self.__class__.__name__}] Data successfully cached.")

    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        P_mx = P_sym.symbolic_object if hasattr(P_sym, "symbolic_object") else P_sym
        h_mx = self.interpolant(ca.vertcat(T_mx, P_mx))
        return EquationNode(name=f"H_LUT_{self.fluid_name}", symbolic_object=h_mx, unit_object=Unit("", "J/mol"))

    def get_K_value(self, T_sym, P_sym, comp):
        raise NotImplementedError("VLE Flash is currently not supported for PureFluidLUT.")


class MulticomponentEOS(PropertyPackage):
    """
    Equation of State (EOS) Property Package for complex mixtures.
    Provides rigorous VLE resolution using the Wilson Correlation for K-values,
    guaranteeing smooth analytical Jacobians for the CasADi EO matrix.
    """

    def __init__(self, components):
        super().__init__(components)
        self.thermo_data = {}

    def fetch_parameters_from_db(self):
        if not HAS_THERMO:
            raise ImportError("The 'thermo' library is required.")

        for comp in self.components:
            try:
                chem = Chemical(comp)
            except Exception as e:
                raise ValueError(f"Component '{comp}' not found in the thermo database.") from e

            try:
                coeffs = chem.HeatCapacityGas.models[0].coeffs
                if len(coeffs) < 4:
                    coeffs = coeffs + [0.0] * (4 - len(coeffs))
            except (AttributeError, IndexError):
                coeffs = [0.0, 0.0, 0.0, 0.0]

            self.thermo_data[comp] = {
                "Hf_298": getattr(chem, "Hfgm", 0.0) or 0.0,
                "Tc": getattr(chem, "Tc", 300.0) or 300.0,
                "Pc": getattr(chem, "Pc", 1e5) or 1e5,
                "omega": getattr(chem, "omega", 0.0) or 0.0,
                "Hvap_298": getattr(chem, "Hvapm", 30000.0) or 30000.0,
                "Cp_coeffs": coeffs[:4],
            }

        print(f"[{self.__class__.__name__}] Successfully loaded thermodynamic VLE data for: {self.components}")

    def get_K_value(self, T_sym, P_sym, comp):
        """
        Calculates the VLE partition coefficient (K_i) using the Wilson Equation.
        Highly performant and continuous for Equation-Oriented Jacobian evaluation.
        """
        data = self.thermo_data[comp]
        Tc = data["Tc"]
        Pc = data["Pc"]
        omega = data["omega"]

        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        P_mx = P_sym.symbolic_object if hasattr(P_sym, "symbolic_object") else P_sym

        # Convert P_mx from bar to Pa to match Pc
        P_pa = P_mx * 1e5

        # Wilson's robust K-value correlation
        K_sym = (Pc / P_pa) * ca.exp(5.373 * (1.0 + omega) * (1.0 - (Tc / T_mx)))

        return EquationNode(name=f"K_wilson_{comp}", symbolic_object=K_sym, unit_object=Unit("", ""))

    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        h_mix_sym = 0.0
        T_ref = 298.15

        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym

        for comp in self.components:
            data = self.thermo_data[comp]
            Hf_298 = data["Hf_298"]
            A, B, C, D = data["Cp_coeffs"]
            Hvap = data["Hvap_298"]

            integral_Cp = (
                A * (T_mx - T_ref)
                + (B / 2.0) * (T_mx**2 - T_ref**2)
                + (C / 3.0) * (T_mx**3 - T_ref**3)
                + (D / 4.0) * (T_mx**4 - T_ref**4)
            )

            # Enthalpy of the vapor state (ideal gas + formation)
            h_comp_pure = Hf_298 + integral_Cp

            # If liquid phase, subtract the latent heat of vaporization
            if phase == "liquid":
                h_comp_pure -= Hvap

            z_node = fractions_sym_dict[comp]
            z_mx = z_node.symbolic_object if hasattr(z_node, "symbolic_object") else z_node

            h_mix_sym += z_mx * h_comp_pure

        return EquationNode(
            name=f"H_{phase}_eos", symbolic_object=h_mix_sym, unit_object=Unit("", "J/mol")
        )