# -*- coding: utf-8 -*-

"""
Thermodynamic Package Module (V5 Native CasADi Architecture).

Implements the SOLID Strategy pattern. Thermodynamic Packages contain the physics
and empirical data retrieval logic, while Streams contain the spatial topology.
These packages dynamically inject variables (like Compressibility Factors) and
complex residual equations (like Iso-Fugacity or Gamma-Phi constraints) directly
into the Stream's DAE block, preserving native CasADi C++ acceleration.

Supported Models:
- PureFluidLUT (B-Spline Tables)
- IdealVLEPackage (Wilson K-Values)
- PengRobinsonEOS (Rigorous Cubic)
- SoaveRedlichKwongEOS (Rigorous Cubic)
- NRTLPackage (Activity Coefficient for highly non-ideal mixtures)
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
    Enforces the implementation of graph-generating methods for physical 
    properties and phase equilibrium constraints.
    """

    def __init__(self, components):
        """
        Initializes the Property Package.

        :param list components: List of string names for chemical components.
        """
        self.components = components

    @abstractmethod
    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        """
        Generates the CasADi symbolic expression for phase-specific enthalpy.

        :param EquationNode|MX T_sym: Temperature node (K).
        :param EquationNode|MX P_sym: Pressure node (bar).
        :param dict fractions_sym_dict: Component molar fractions.
        :param str phase: Target phase ('liquid' or 'vapor').
        :return: Analytical polynomial expression for the mixture enthalpy.
        :rtype: EquationNode
        """
        pass

    @abstractmethod
    def build_phase_equilibrium(self, stream_instance):
        """
        Strategy Method: Injects the specific Vapor-Liquid Equilibrium physics
        (equations and auxiliary variables) into the provided Stream instance.

        :param TwoPhaseStream stream_instance: The stream requiring equilibrium physics.
        """
        pass


class PureFluidLUT(PropertyPackage):
    """
    Lookup Table (LUT) Property Package for Pure Fluids.
    Generates a 2D B-Spline interpolation matrix for highly efficient
    thermodynamic property retrieval (e.g., Enthalpy), bypassing costly
    empirical correlations during the Newton-Raphson iterations. Ideal for
    thermal utilities (e.g., cooling water, steam).
    """

    def __init__(
        self, fluid_name, T_bounds=(273.15, 600.0), P_bounds=(1.0, 10.0), grid_size=20
    ):
        """
        Instantiates the LUT Package and pre-computes the B-Spline mesh.

        :param str fluid_name: The chemical name of the pure fluid.
        :param tuple T_bounds: Temperature limits (min, max) in Kelvin.
        :param tuple P_bounds: Pressure limits (min, max) in bar.
        :param int grid_size: Resolution of the discretization grid.
        :raises ImportError: If the `thermo` library is not installed.
        """
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

    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase="liquid"):
        """
        Retrieves the interpolated Enthalpy directly from the B-Spline.
        
        :param EquationNode|MX T_sym: Temperature node (K).
        :param EquationNode|MX P_sym: Pressure node (bar).
        :param dict fractions_sym_dict: Molar fractions (ignored for pure fluids).
        :param str phase: Phase string (default 'liquid').
        :return: Interpolated Molar Enthalpy.
        :rtype: EquationNode
        """
        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        P_mx = P_sym.symbolic_object if hasattr(P_sym, "symbolic_object") else P_sym
        h_mx = self.interpolant(ca.vertcat(T_mx, P_mx))
        return EquationNode(name=f"H_LUT_{self.fluid_name}", symbolic_object=h_mx, unit_object=Unit("", "J/mol"))

    def build_phase_equilibrium(self, stream_instance):
        """VLE flash operations are not supported for pure fluid LUTs."""
        pass


class IdealVLEPackage(PropertyPackage):
    """
    Ideal/Semi-Rigorous Property Package.
    Uses Wilson's Correlation for K-Values. Fast, highly differentiable, excellent
    for initial estimates or near-ideal hydrocarbon mixtures. Acts as the base class
    for advanced EOS fetching routines.
    """

    def __init__(self, components):
        """
        :param list components: List of string names for chemical components.
        """
        super().__init__(components)
        self.thermo_data = {}

    def fetch_parameters_from_db(self):
        """
        Connects to the 'thermo' database to extract critical properties,
        acentric factors, and heat capacity polynomials.

        :raises ImportError: If the `thermo` library is missing.
        :raises ValueError: If a component is not found in the database.
        """
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
                coeffs = [30.0, 0.0, 0.0, 0.0]
            
            # THE MATHEMATICAL VACCINE: Guarantee dH/dT != 0
            if coeffs[0] == 0.0 and coeffs[1] == 0.0:
                coeffs[0] = 30.0

            self.thermo_data[comp] = {
                "Hf_298": getattr(chem, "Hfgm", 0.0) or 0.0,
                "Tc": getattr(chem, "Tc", 300.0) or 300.0,
                "Pc": getattr(chem, "Pc", 1e5) or 1e5,
                "omega": getattr(chem, "omega", 0.0) or 0.0,
                "Hvap_298": getattr(chem, "Hvapm", 30000.0) or 30000.0,
                "Cp_coeffs": coeffs[:4],
            }
        print(f"[{self.__class__.__name__}] VLE data loaded for: {self.components}")

    def build_phase_equilibrium(self, stream_instance):
        """
        Injects standard K-value partitioning into the stream (y_i = K_i * x_i).
        
        :param TwoPhaseStream stream_instance: The stream requiring equilibrium physics.
        """
        T_sym = stream_instance.T
        P_sym = stream_instance.P

        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        P_mx = P_sym.symbolic_object if hasattr(P_sym, "symbolic_object") else P_sym
        P_pa = P_mx * 1e5

        for comp in self.components:
            data = self.thermo_data[comp]
            K_sym = (data["Pc"] / P_pa) * ca.exp(5.373 * (1.0 + data["omega"]) * (1.0 - (data["Tc"] / T_mx)))
            K_node = EquationNode(name=f"K_{comp}", symbolic_object=K_sym, unit_object=Unit("", ""))
            
            x_c = stream_instance.x[comp]()
            y_c = stream_instance.y[comp]()
            
            stream_instance.createEquation(
                f"VLE_Partition_{comp}", 
                description=f"Ideal K-Value partitioning for {comp}", 
                expr=y_c - (K_node * x_c)
            )

    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        """
        Calculates sensible and latent enthalpy via heat capacity integration.
        
        :param EquationNode|MX T_sym: Temperature node (K).
        :param EquationNode|MX P_sym: Pressure node (bar).
        :param dict fractions_sym_dict: Component molar fractions.
        :param str phase: Target phase ('liquid' or 'vapor').
        :return: Analytical polynomial expression for the mixture enthalpy.
        :rtype: EquationNode
        """
        h_mix_sym = 0.0
        T_ref = 298.15
        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym

        for comp in self.components:
            data = self.thermo_data[comp]
            A, B, C, D = data["Cp_coeffs"]
            integral_Cp = A*(T_mx - T_ref) + (B/2.0)*(T_mx**2 - T_ref**2) + (C/3.0)*(T_mx**3 - T_ref**3) + (D/4.0)*(T_mx**4 - T_ref**4)
            h_comp = data["Hf_298"] + integral_Cp
            if phase == "liquid":
                h_comp -= data["Hvap_298"]
            
            z_mx = fractions_sym_dict[comp].symbolic_object if hasattr(fractions_sym_dict[comp], "symbolic_object") else fractions_sym_dict[comp]
            h_mix_sym += z_mx * h_comp

        return EquationNode(name=f"H_{phase}", symbolic_object=h_mix_sym, unit_object=Unit("", "J/mol"))


class PengRobinsonEOS(IdealVLEPackage):
    """
    Rigorous Equation of State (EOS) Property Package (Peng-Robinson).
    Injects Compressibility Factors (Z) as algebraic variables and Iso-Fugacity 
    coefficients directly into the Stream's CasADi matrix.
    """

    def build_phase_equilibrium(self, stream_instance):
        """
        Injects rigorous Peng-Robinson Iso-Fugacity constraints into the stream.
        Calculates polynomial roots for compressibility (Z) within the DAE system.
        
        :param TwoPhaseStream stream_instance: The stream requiring equilibrium physics.
        """
        stream_instance.Z_L = stream_instance.createVariable("Z_L", "", description="Liquid Compressibility", value=0.01, exposure_type="algebraic")
        stream_instance.Z_V = stream_instance.createVariable("Z_V", "", description="Vapor Compressibility", value=0.99, exposure_type="algebraic")

        T_mx = stream_instance.T.symbolic_object
        P_mx = stream_instance.P.symbolic_object * 1e5  # Pa
        R = 8.314

        a_comp, b_comp = {}, {}
        for comp in self.components:
            Tc = self.thermo_data[comp]["Tc"]
            Pc = self.thermo_data[comp]["Pc"]
            w = self.thermo_data[comp]["omega"]

            kappa = 0.37464 + 1.54226 * w - 0.26992 * w**2
            alpha = (1.0 + kappa * (1.0 - ca.sqrt(T_mx / Tc)))**2
            a_comp[comp] = 0.45724 * (R**2 * Tc**2 / Pc) * alpha
            b_comp[comp] = 0.07780 * (R * Tc / Pc)

        def get_cubic_params(fractions_dict):
            a_mix, b_mix = 0.0, 0.0
            for i in self.components:
                xi = fractions_dict[i].symbolic_object
                b_mix += xi * b_comp[i]
                for j in self.components:
                    xj = fractions_dict[j].symbolic_object
                    a_mix += xi * xj * ca.sqrt(a_comp[i] * a_comp[j])
            
            A_mix = (a_mix * P_mx) / (R**2 * T_mx**2)
            B_mix = (b_mix * P_mx) / (R * T_mx)
            return a_mix, b_mix, A_mix, B_mix

        aL, bL, A_L, B_L = get_cubic_params(stream_instance.x)
        aV, bV, A_V, B_V = get_cubic_params(stream_instance.y)

        Z_L_sym = stream_instance.Z_L.symbolic_object
        res_L = Z_L_sym**3 - (1 - B_L)*Z_L_sym**2 + (A_L - 2*B_L - 3*B_L**2)*Z_L_sym - (A_L*B_L - B_L**2 - B_L**3)
        
        Z_V_sym = stream_instance.Z_V.symbolic_object
        res_V = Z_V_sym**3 - (1 - B_V)*Z_V_sym**2 + (A_V - 2*B_V - 3*B_V**2)*Z_V_sym - (A_V*B_V - B_V**2 - B_V**3)

        stream_instance.createEquation("EOS_Cubic_Liquid", description="PR Cubic Root for Liquid", expr=EquationNode("resL", res_L, Unit("","")))
        stream_instance.createEquation("EOS_Cubic_Vapor", description="PR Cubic Root for Vapor", expr=EquationNode("resV", res_V, Unit("","")))

        def calc_ln_phi(comp, Z, A, B, b_mix, a_mix):
            bi_b = b_comp[comp] / b_mix
            sum_x_aij = ca.sqrt(a_comp[comp] * a_mix) 
            term_a = (2.0 * sum_x_aij / a_mix) - bi_b
            return bi_b * (Z - 1) - ca.log(Z - B) - (A / (2 * ca.sqrt(2) * B)) * term_a * ca.log((Z + (1 + ca.sqrt(2))*B) / (Z + (1 - ca.sqrt(2))*B))

        for comp in self.components:
            ln_phi_L = calc_ln_phi(comp, stream_instance.Z_L.symbolic_object, A_L, B_L, bL, aL)
            ln_phi_V = calc_ln_phi(comp, stream_instance.Z_V.symbolic_object, A_V, B_V, bV, aV)

            fug_L = stream_instance.x[comp].symbolic_object * ca.exp(ln_phi_L)
            fug_V = stream_instance.y[comp].symbolic_object * ca.exp(ln_phi_V)

            eq_iso_fug = fug_L - fug_V
            stream_instance.createEquation(
                f"IsoFugacity_{comp}", description=f"PR Iso-Fugacity for {comp}", expr=EquationNode("isoF", eq_iso_fug, Unit("",""))
            )


class SoaveRedlichKwongEOS(IdealVLEPackage):
    """
    Rigorous Equation of State (EOS) Property Package (SRK).
    A robust alternative to Peng-Robinson, often preferred for light hydrocarbons.
    Injects Compressibility Factors (Z) and Iso-Fugacity constraints.
    """

    def build_phase_equilibrium(self, stream_instance):
        """
        Injects rigorous SRK Iso-Fugacity constraints into the stream.
        Calculates polynomial roots for compressibility (Z) within the DAE system.
        
        :param TwoPhaseStream stream_instance: The stream requiring equilibrium physics.
        """
        stream_instance.Z_L = stream_instance.createVariable("Z_L", "", description="Liquid Compressibility", value=0.01, exposure_type="algebraic")
        stream_instance.Z_V = stream_instance.createVariable("Z_V", "", description="Vapor Compressibility", value=0.99, exposure_type="algebraic")

        T_mx = stream_instance.T.symbolic_object
        P_mx = stream_instance.P.symbolic_object * 1e5  # Pa
        R = 8.314

        a_comp, b_comp = {}, {}
        for comp in self.components:
            Tc = self.thermo_data[comp]["Tc"]
            Pc = self.thermo_data[comp]["Pc"]
            w = self.thermo_data[comp]["omega"]

            m = 0.480 + 1.574 * w - 0.176 * w**2
            alpha = (1.0 + m * (1.0 - ca.sqrt(T_mx / Tc)))**2
            a_comp[comp] = 0.42748 * (R**2 * Tc**2 / Pc) * alpha
            b_comp[comp] = 0.08664 * (R * Tc / Pc)

        def get_cubic_params(fractions_dict):
            a_mix, b_mix = 0.0, 0.0
            for i in self.components:
                xi = fractions_dict[i].symbolic_object
                b_mix += xi * b_comp[i]
                for j in self.components:
                    xj = fractions_dict[j].symbolic_object
                    a_mix += xi * xj * ca.sqrt(a_comp[i] * a_comp[j]) # Assumption: k_ij = 0
            
            A_mix = (a_mix * P_mx) / (R**2 * T_mx**2)
            B_mix = (b_mix * P_mx) / (R * T_mx)
            return a_mix, b_mix, A_mix, B_mix

        aL, bL, A_L, B_L = get_cubic_params(stream_instance.x)
        aV, bV, A_V, B_V = get_cubic_params(stream_instance.y)

        # SRK Cubic Equation form: Z^3 - Z^2 + (A - B - B^2)Z - AB = 0
        Z_L_sym = stream_instance.Z_L.symbolic_object
        res_L = Z_L_sym**3 - Z_L_sym**2 + (A_L - B_L - B_L**2)*Z_L_sym - (A_L * B_L)
        
        Z_V_sym = stream_instance.Z_V.symbolic_object
        res_V = Z_V_sym**3 - Z_V_sym**2 + (A_V - B_V - B_V**2)*Z_V_sym - (A_V * B_V)

        stream_instance.createEquation("EOS_Cubic_Liquid", description="SRK Cubic Root for Liquid", expr=EquationNode("resL", res_L, Unit("","")))
        stream_instance.createEquation("EOS_Cubic_Vapor", description="SRK Cubic Root for Vapor", expr=EquationNode("resV", res_V, Unit("","")))

        def calc_ln_phi(comp, Z, A, B, b_mix, a_mix):
            bi_b = b_comp[comp] / b_mix
            sum_x_aij = ca.sqrt(a_comp[comp] * a_mix) 
            term_a = (2.0 * sum_x_aij / a_mix) - bi_b
            
            # SRK fugacity coefficient derivation
            ln_phi = bi_b * (Z - 1) - ca.log(Z - B) - (A / B) * term_a * ca.log(1.0 + (B / Z))
            return ln_phi

        for comp in self.components:
            ln_phi_L = calc_ln_phi(comp, stream_instance.Z_L.symbolic_object, A_L, B_L, bL, aL)
            ln_phi_V = calc_ln_phi(comp, stream_instance.Z_V.symbolic_object, A_V, B_V, bV, aV)

            fug_L = stream_instance.x[comp].symbolic_object * ca.exp(ln_phi_L)
            fug_V = stream_instance.y[comp].symbolic_object * ca.exp(ln_phi_V)

            eq_iso_fug = fug_L - fug_V
            stream_instance.createEquation(
                f"IsoFugacity_{comp}", description=f"SRK Iso-Fugacity for {comp}", expr=EquationNode("isoF", eq_iso_fug, Unit("",""))
            )


class NRTLPackage(IdealVLEPackage):
    """
    Non-Random Two-Liquid (NRTL) Activity Coefficient Model.
    Industry standard for highly non-ideal polar mixtures, Azeotropes, and
    Liquid-Liquid Equilibria (LLE). Implements a robust Gamma-Phi formulation.
    """

    def __init__(self, components):
        """
        Initializes the NRTL package with default interaction parameters.

        :param list components: List of string names for chemical components.
        """
        super().__init__(components)
        n = len(components)
        self.tau_matrix = np.zeros((n, n))
        self.alpha_matrix = np.full((n, n), 0.3)  # Standard non-randomness parameter
        
    def set_binary_parameters(self, tau_matrix, alpha_matrix):
        """
        Allows the user to inject rigorous binary interaction parameters (BIPs).

        :param ndarray tau_matrix: Interaction parameters matrix (tau_ij).
        :param ndarray alpha_matrix: Non-randomness matrix (alpha_ij).
        """
        self.tau_matrix = tau_matrix
        self.alpha_matrix = alpha_matrix

    def _get_lee_kesler_psat(self, T_mx, comp):
        """
        Generates a purely symbolic, continuously differentiable estimation
        of the saturation pressure (Pa) using the Lee-Kesler analytic equation.
        Ensures CasADi Jacobian continuity.

        :param MX T_mx: Symbolic temperature (K).
        :param str comp: Target chemical component.
        :return: Saturation pressure evaluated at T_mx (Pa).
        :rtype: MX
        """
        Tc = self.thermo_data[comp]["Tc"]
        Pc = self.thermo_data[comp]["Pc"]
        w = self.thermo_data[comp]["omega"]

        Tr = T_mx / Tc
        f0 = 5.92714 - 6.09648 / Tr - 1.28862 * ca.log(Tr) + 0.169347 * Tr**6
        f1 = 15.2518 - 15.6875 / Tr - 13.4721 * ca.log(Tr) + 0.43577 * Tr**6
        
        ln_Pr_sat = f0 + w * f1
        return Pc * ca.exp(ln_Pr_sat)

    def build_phase_equilibrium(self, stream_instance):
        """
        Injects the Gamma-Phi formulation into the stream.
        y_i * P = x_i * gamma_i * P_sat_i

        :param TwoPhaseStream stream_instance: The stream requiring equilibrium physics.
        """
        T_sym = stream_instance.T
        P_sym = stream_instance.P

        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        P_mx = P_sym.symbolic_object if hasattr(P_sym, "symbolic_object") else P_sym
        P_pa = P_mx * 1e5

        n = len(self.components)
        G = ca.MX.zeros(n, n)
        tau = ca.MX.zeros(n, n)

        for i in range(n):
            for j in range(n):
                tau[i, j] = self.tau_matrix[i, j]
                G[i, j] = ca.exp(-self.alpha_matrix[i, j] * tau[i, j])

        # Extract liquid fractions as a column vector
        x_vec = ca.vertcat(*[stream_instance.x[c].symbolic_object for c in self.components])

        for i, comp in enumerate(self.components):
            # Calculate Activity Coefficient (Gamma_i) natively in CasADi
            sum_Gj_xj = ca.sum1(G[:, i] * x_vec)
            term1 = ca.sum1(tau[:, i] * G[:, i] * x_vec) / sum_Gj_xj
            
            term2 = 0.0
            for j in range(n):
                num = x_vec[j] * G[i, j]
                den = ca.sum1(G[:, j] * x_vec)
                sub_term = tau[i, j] - (ca.sum1(x_vec * tau[:, j] * G[:, j]) / den)
                term2 += (num / den) * sub_term
                
            ln_gamma_i = term1 + term2
            gamma_i = ca.exp(ln_gamma_i)

            P_sat_i = self._get_lee_kesler_psat(T_mx, comp)

            # Gamma-Phi Equation
            x_c = stream_instance.x[comp].symbolic_object
            y_c = stream_instance.y[comp].symbolic_object

            eq_gamma_phi = (y_c * P_pa) - (x_c * gamma_i * P_sat_i)

            stream_instance.createEquation(
                f"VLE_Gamma_Phi_{comp}", 
                description=f"NRTL Gamma-Phi Equilibrium for {comp}", 
                expr=EquationNode(f"VLE_{comp}", eq_gamma_phi, Unit("", ""))
            )