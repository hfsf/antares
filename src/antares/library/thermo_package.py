# -*- coding: utf-8 -*-

r"""
Thermodynamic Package Module (V5 Native CasADi Architecture).

This module implements the Strategy Pattern for property evaluation within the
Equation-Oriented (EO) framework. It provides robust thermodynamic engines
(Ideal, Cubic EOS, and Activity Models) specifically designed with mathematical
"shields" (Leaky Abstractions, Archimedean smoothing, and Normalized K-Factors).
These shields prevent vanishing gradients, NaNs, and singular Jacobians during
the rigorous NLP/Rootfinding process.
"""

from abc import ABC, abstractmethod

import casadi as ca
import numpy as np

from antares.core.expression_evaluation import EquationNode
from antares.core.unit import Unit
import antares.core.GLOBAL_CFG as cfg

try:
    import thermo
    from thermo.chemical import Chemical
    HAS_THERMO = True
except ImportError:
    HAS_THERMO = False

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False


def leaky_log(x, eps=1e-5):
    r"""
    Evaluates a mathematically safe, leaky natural logarithm.

    In Equation-Oriented (EO) architectures, Newton-Raphson steps can
    temporarily push variables into non-physical domains (e.g., $x \le 0$).
    A standard natural logarithm would return NaN, causing the solver
    to fail unrecoverably. This function computes a standard logarithm
    for $x > eps$, and applies a highly sloped linear extrapolation for
    $x \le eps$, ensuring continuous, non-zero gradients.

    :param x: The symbolic variable or numerical value to be evaluated.
    :type x: casadi.MX or float
    :param float eps: The threshold below which linear extrapolation is applied. Defaults to 1e-5.
    :return: The smoothed, numerically safe logarithmic evaluation.
    :rtype: casadi.MX
    """
    safe_x = ca.fmax(x, eps)
    return ca.if_else(x > eps, ca.log(safe_x), ca.log(eps) + (1.0 / eps) * (x - eps))


def smooth_abs(x, eps=1e-8):
    r"""
    Evaluates an Archimedean smoothed absolute value.

    Standard absolute values or strict `max(x, 0)` functions create flat
    gradients ($0.0$) when the condition is not met. This prevents the
    Newton-Raphson method from calculating a valid directional step. This
    function uses a parabolic root approach to maintain a continuous,
    non-zero derivative across all domains.

    :param x: The symbolic variable or numerical value to evaluate.
    :type x: casadi.MX or float
    :param float eps: The smoothing parameter. Defaults to 1e-8.
    :return: The smoothed evaluation $\sqrt{x^2 + \epsilon}$.
    :rtype: casadi.MX
    """
    return ca.sqrt(x**2 + eps)


class PropertyPackage(ABC):
    r"""
    Abstract Base Class for all Thermodynamic Property Packages.

    Enforces the implementation of phase equilibrium constraints and
    enthalpy expressions. Also provides native topological initialization
    methods and NLP bound injections to prevent singular Jacobians.
    
    :param list components: A list of string identifiers for the chemical components.
    """

    def __init__(self, components):
        self.components = components

    @abstractmethod
    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        r"""
        Generates the symbolic enthalpy expression for a given phase.

        :param casadi.MX T_sym: Symbolic temperature variable.
        :param casadi.MX P_sym: Symbolic pressure variable.
        :param dict fractions_sym_dict: Dictionary mapping components to their symbolic molar fractions.
        :param str phase: The target phase ('liquid' or 'vapor').
        :return: An EquationNode containing the computed enthalpy.
        :rtype: EquationNode
        """
        pass

    @abstractmethod
    def build_phase_equilibrium(self, stream_instance):
        r"""
        Constructs the phase equilibrium constraints (e.g., isofugacity) inside a stream.

        :param Model stream_instance: The stream model where constraints will be injected.
        """
        pass

    def _apply_nlp_bounds(self, stream_instance):
        r"""
        Injects strict thermodynamic boundary limits into the stream's variables.
        This is heavily utilized by Interior Point NLP Solvers (like IPOPT) to 
        prevent the engine from exploring mathematically valid but physically 
        impossible domains (e.g., negative temperatures, fractions > 1.0).

        :param Model stream_instance: The target stream model.
        """
        # Absolute Temperature (K) must be strictly positive
        if hasattr(stream_instance, "T"):
            stream_instance.T.is_lower_bounded = True
            stream_instance.T.lower_bound = 1e-2
            
        # Absolute Pressure (bar) must be strictly positive
        if hasattr(stream_instance, "P"):
            stream_instance.P.is_lower_bounded = True
            stream_instance.P.lower_bound = 1e-5
            
        # Vapor fraction strictly between 0 and 1
        if hasattr(stream_instance, "V_frac"):
            stream_instance.V_frac.is_lower_bounded = True
            stream_instance.V_frac.lower_bound = 0.0
            stream_instance.V_frac.is_upper_bounded = True
            stream_instance.V_frac.upper_bound = 1.0
            
        # Molar Fractions strictly between 0 and 1
        for comp in self.components:
            if hasattr(stream_instance, "x") and comp in stream_instance.x:
                stream_instance.x[comp].is_lower_bounded = True
                stream_instance.x[comp].lower_bound = 0.0
                stream_instance.x[comp].is_upper_bounded = True
                stream_instance.x[comp].upper_bound = 1.0
                
            if hasattr(stream_instance, "y") and comp in stream_instance.y:
                stream_instance.y[comp].is_lower_bounded = True
                stream_instance.y[comp].lower_bound = 0.0
                stream_instance.y[comp].is_upper_bounded = True
                stream_instance.y[comp].upper_bound = 1.0

    def _apply_asymmetric_initialization(self, stream_instance):
        r"""
        Applies a physically robust asymmetric warm-start to phase compositions.

        Acts as a topological shield against the "Bilinearity Trap". 
        Assumes by generic convention that the first component is the lightest 
        (biases towards vapor) and the last is the heaviest (biases towards liquid).
        
        :param Model stream_instance: The target stream model.
        """
        n = len(self.components)
        if n < 2:
            return
            
        for i, comp in enumerate(self.components):
            if i == 0:
                val_x, val_y = 0.1, 0.9
            elif i == n - 1:
                val_x, val_y = 0.9, 0.1
            else:
                val_x, val_y = 1.0 / n, 1.0 / n
                
            for var_dict, val in [(stream_instance.x, val_x), (stream_instance.y, val_y)]:
                var = var_dict[comp]
                if hasattr(var, "initial_condition_array") and var.initial_condition_array is not None:
                    var.initial_condition_array[:] = val
                if hasattr(var, "setValue"):
                    var.setValue(val)
                elif hasattr(var, "value"):
                    var.value = val


class IdealVLEPackage(PropertyPackage):
    r"""
    Ideal Vapor-Liquid Equilibrium Property Package.
    Handles critical parameters and automatic Binary Interaction Parameters ($k_{ij}$) retrieval.
    
    :param list components: List of chemical component names.
    """
    def __init__(self, components):
        super().__init__(components)
        self.thermo_data = {}
        n = len(components)
        self.kij_matrix = np.zeros((n, n))
        self._kij_user_defined = False

    def set_binary_interactions(self, kij_matrix):
        r"""
        Manually injects the Binary Interaction Parameters ($k_{ij}$) matrix,
        overriding the automatic database fetch behavior.

        :param kij_matrix: A symmetric $N \times N$ matrix where $k_{ii} = 0$.
        :type kij_matrix: numpy.ndarray or list of list
        """
        self.kij_matrix = np.array(kij_matrix)
        self._kij_user_defined = True
        
        if getattr(cfg, "VERBOSITY_LEVEL", 0) >= 1:
            print("[ANTARES THERMO] User-defined binary interaction parameters (k_ij) injected successfully.")

    def _auto_fetch_kij(self):
        r"""
        Attempts to automatically fetch Binary Interaction Parameters ($k_{ij}$) 
        utilizing a robust fallback mechanism.

        Fallback Priority:
        1. ``thermo`` library (Best for PR/SRK Cubics).
        2. ``CoolProp`` library (Advanced HEOS database).
        3. Default to 0.0 (Fallback to prevent solver collapse).
        """
        n = len(self.components)
        target_eos = "PR" if "PengRobinson" in self.__class__.__name__ else "SRK"
        
        if HAS_THERMO:
            try:
                from thermo import interaction_parameters
                db_dict = getattr(interaction_parameters, f"{target_eos}_kij", None)
                if db_dict is not None:
                    for i in range(n):
                        for j in range(i + 1, n):
                            cas_i = self.thermo_data[self.components[i]].get("CAS", None)
                            cas_j = self.thermo_data[self.components[j]].get("CAS", None)
                            if cas_i and cas_j:
                                pair_key = frozenset([cas_i, cas_j])
                                if pair_key in db_dict:
                                    val = db_dict[pair_key]
                                    self.kij_matrix[i, j] = val
                                    self.kij_matrix[j, i] = val
                    return  
            except Exception:
                pass

        if HAS_COOLPROP:
            try:
                for i in range(n):
                    for j in range(i + 1, n):
                        comp_i_cp = self.components[i].capitalize()
                        comp_j_cp = self.components[j].capitalize()
                        try:
                            val = CP.get_fluid_param_string(f"{comp_i_cp}&{comp_j_cp}", "betaT")
                            numeric_val = float(val) if val else 0.0
                            self.kij_matrix[i, j] = numeric_val
                            self.kij_matrix[j, i] = numeric_val
                        except Exception:
                            continue
                return 
            except Exception:
                pass

    def fetch_parameters_from_db(self):
        r"""
        Fetches critical properties, acentric factors, heat capacity polynomials, 
        and automatic interaction parameters from the primary database. 

        :raises ImportError: If the 'thermo' library is not installed (Required for critical properties).
        """
        if not HAS_THERMO:
            raise ImportError("The 'thermo' library is required to fetch pure component critical properties.")
            
        for comp in self.components:
            chem = Chemical(comp)
            try:
                coeffs = chem.HeatCapacityGas.models[0].coeffs
                if len(coeffs) < 4:
                    coeffs = coeffs + [0.0] * (4 - len(coeffs))
            except (AttributeError, IndexError):
                coeffs = [30.0, 0.0, 0.0, 0.0]
            if coeffs[0] == 0.0 and coeffs[1] == 0.0:
                coeffs[0] = 30.0

            self.thermo_data[comp] = {
                "CAS": getattr(chem, "CAS", None),
                "Hf_298": getattr(chem, "Hfgm", 0.0) or 0.0,
                "Tc": getattr(chem, "Tc", 300.0) or 300.0,
                "Pc": (getattr(chem, "Pc", 1e5) or 1e5) / 1e5,
                "omega": getattr(chem, "omega", 0.0) or 0.0,
                "Hvap_298": getattr(chem, "Hvapm", 30000.0) or 30000.0,
                "Cp_coeffs": coeffs[:4],
            }

        if not self._kij_user_defined:
            self._auto_fetch_kij()

    def build_phase_equilibrium(self, stream_instance):
        r"""
        Builds Ideal Raoult/Antoine-based phase equilibrium equations.

        :param Model stream_instance: Target stream for the equations.
        """
        self._apply_nlp_bounds(stream_instance)
        self._apply_asymmetric_initialization(stream_instance)
        
        T_mx = stream_instance.T.symbolic_object if hasattr(stream_instance.T, "symbolic_object") else stream_instance.T
        P_mx = stream_instance.P.symbolic_object if hasattr(stream_instance.P, "symbolic_object") else stream_instance.P

        T_safe = ca.fmax(T_mx, 1.0)
        P_bar = ca.fmax(P_mx, 0.01)

        for comp in self.components:
            data = self.thermo_data[comp]
            K_sym = (data["Pc"] / P_bar) * ca.exp(5.373 * (1.0 + data["omega"]) * (1.0 - (data["Tc"] / T_safe)))
            K_node = EquationNode(name=f"K_{comp}", symbolic_object=K_sym, unit_object=Unit("", ""))

            x_c = stream_instance.x[comp]()
            y_c = stream_instance.y[comp]()
            stream_instance.createEquation(f"VLE_Partition_{comp}", expr=y_c - (K_node * x_c))

    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        r"""
        Calculates the Ideal Gas/Liquid enthalpy using integrated heat capacity polynomials.

        :param casadi.MX T_sym: Symbolic temperature.
        :param casadi.MX P_sym: Symbolic pressure.
        :param dict fractions_sym_dict: Molar fractions.
        :param str phase: Phase target ('liquid' or 'vapor').
        :return: EquationNode representing the enthalpy.
        :rtype: EquationNode
        """
        h_mix_sym = 0.0
        T_ref = 298.15
        T_mx = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        T_safe = ca.fmax(T_mx, 1.0)

        for comp in self.components:
            data = self.thermo_data[comp]
            A, B, C, D = data["Cp_coeffs"]
            integral_Cp = (
                A * (T_safe - T_ref)
                + (B / 2.0) * (T_safe**2 - T_ref**2)
                + (C / 3.0) * (T_safe**3 - T_ref**3)
                + (D / 4.0) * (T_safe**4 - T_ref**4)
            )
            h_comp = data["Hf_298"] + integral_Cp
            if phase == "liquid":
                h_comp -= data["Hvap_298"]

            z_mx = fractions_sym_dict[comp].symbolic_object if hasattr(fractions_sym_dict[comp], "symbolic_object") else fractions_sym_dict[comp]
            h_mix_sym += z_mx * h_comp

        return EquationNode(name=f"H_{phase}", symbolic_object=h_mix_sym, unit_object=Unit("", "J/mol"))


class PengRobinsonEOS(IdealVLEPackage):
    r"""
    Rigorous Equation of State (Peng-Robinson).
    """
    def build_phase_equilibrium(self, stream_instance):
        self._apply_nlp_bounds(stream_instance)
        self._apply_asymmetric_initialization(stream_instance)

        # NLP Boundaries for compressibility roots (Z > 0)
        stream_instance.Z_L = stream_instance.createVariable(
            "Z_L", "", value=0.05, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-5
        )
        stream_instance.Z_V = stream_instance.createVariable(
            "Z_V", "", value=0.95, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-5
        )

        T_raw = stream_instance.T.symbolic_object if hasattr(stream_instance.T, "symbolic_object") else stream_instance.T
        P_raw = stream_instance.P.symbolic_object if hasattr(stream_instance.P, "symbolic_object") else stream_instance.P

        T_safe = ca.fmax(T_raw, 1.0)
        P_safe_pa = ca.fmax(P_raw * 1e5, 100.0)
        R_gas = 8.314

        a_comp, b_comp = [], []
        for comp in self.components:
            data = self.thermo_data[comp]
            Tr = T_safe / ca.fmax(data["Tc"], 1.0)
            kappa = 0.37464 + 1.54226 * data["omega"] - 0.26992 * data["omega"] ** 2
            alpha = (1.0 + kappa * (1.0 - ca.sqrt(Tr))) ** 2
            Pc_pa = data["Pc"] * 1e5
            a_comp.append(0.45724 * (R_gas**2 * data["Tc"] ** 2 / Pc_pa) * alpha)
            b_comp.append(0.07780 * (R_gas * data["Tc"] / Pc_pa))

        def get_cubic_params(fractions_dict):
            sum_fracs = ca.fmax(ca.sum1(ca.vertcat(*[fractions_dict[c].symbolic_object for c in self.components])), 1e-12)
            a_mix, b_mix = 0.0, 0.0
            for i in range(len(self.components)):
                xi = fractions_dict[self.components[i]].symbolic_object / sum_fracs
                b_mix += xi * b_comp[i]
                for j in range(len(self.components)):
                    xj = fractions_dict[self.components[j]].symbolic_object / sum_fracs
                    a_mix += xi * xj * ca.sqrt(a_comp[i] * a_comp[j]) * (1.0 - self.kij_matrix[i, j])
            A_mix = (a_mix * P_safe_pa) / (R_gas**2 * T_safe**2)
            B_mix = (b_mix * P_safe_pa) / (R_gas * T_safe)
            return a_mix, b_mix, A_mix, B_mix

        aL, bL, A_L, B_L = get_cubic_params(stream_instance.x)
        aV, bV, A_V, B_V = get_cubic_params(stream_instance.y)

        Z_L = stream_instance.Z_L.symbolic_object
        Z_V = stream_instance.Z_V.symbolic_object

        res_L = Z_L**3 - (1.0 - B_L) * Z_L**2 + (A_L - 2.0 * B_L - 3.0 * B_L**2) * Z_L - (A_L * B_L - B_L**2 - B_L**3)
        res_V = Z_V**3 - (1.0 - B_V) * Z_V**2 + (A_V - 2.0 * B_V - 3.0 * B_V**2) * Z_V - (A_V * B_V - B_V**2 - B_V**3)

        stream_instance.createEquation("EOS_Cubic_Liquid", expr=EquationNode("resL", res_L, Unit("", "")))
        stream_instance.createEquation("EOS_Cubic_Vapor", expr=EquationNode("resV", res_V, Unit("", "")))

        def calc_phi(comp, Z, A, B, b_mix, a_mix, i_idx):
            bi_b = b_comp[i_idx] / ca.fmax(b_mix, 1e-12)
            sum_x_aij = 0.0
            for j in range(len(self.components)):
                frac = stream_instance.x[self.components[j]].symbolic_object if Z is Z_L else stream_instance.y[self.components[j]].symbolic_object
                sum_x_aij += ca.fmax(frac, 0.0) * ca.sqrt(a_comp[i_idx] * a_comp[j]) * (1.0 - self.kij_matrix[i_idx, j])

            sum_fracs = ca.fmax(
                ca.sum1(ca.vertcat(*[ca.fmax(stream_instance.x[c].symbolic_object if Z is Z_L else stream_instance.y[c].symbolic_object, 0.0) for c in self.components])),
                1e-12,
            )
            term_a = (2.0 * sum_x_aij / (ca.fmax(a_mix, 1e-12) * sum_fracs)) - bi_b
            arg1 = Z - B
            arg2 = Z + (1.0 + 1.41421356) * B
            arg3 = Z + (1.0 - 1.41421356) * B

            ln_phi = bi_b * (Z - 1.0) - leaky_log(arg1) - (A / ca.fmax(2.82842712 * B, 1e-12)) * term_a * (leaky_log(arg2) - leaky_log(arg3))
            return ca.exp(ca.fmax(ca.fmin(ln_phi, 20.0), -20.0))

        for i, comp in enumerate(self.components):
            phi_L = calc_phi(comp, Z_L, A_L, B_L, bL, aL, i)
            phi_V = calc_phi(comp, Z_V, A_V, B_V, bV, aV, i)
            x_c = stream_instance.x[comp].symbolic_object
            y_c = stream_instance.y[comp].symbolic_object
            K_i = phi_L / ca.fmax(phi_V, 1e-12)
            stream_instance.createEquation(f"IsoFugacity_{comp}", expr=EquationNode("isoF", y_c - K_i * x_c, Unit("", "")))


class SoaveRedlichKwongEOS(IdealVLEPackage):
    r"""
    Rigorous Equation of State (Soave-Redlich-Kwong).
    """
    def build_phase_equilibrium(self, stream_instance):
        self._apply_nlp_bounds(stream_instance)
        self._apply_asymmetric_initialization(stream_instance)

        stream_instance.Z_L = stream_instance.createVariable(
            "Z_L", "", value=0.05, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-5
        )
        stream_instance.Z_V = stream_instance.createVariable(
            "Z_V", "", value=0.95, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-5
        )

        T_raw = stream_instance.T.symbolic_object if hasattr(stream_instance.T, "symbolic_object") else stream_instance.T
        P_raw = stream_instance.P.symbolic_object if hasattr(stream_instance.P, "symbolic_object") else stream_instance.P

        T_safe = ca.fmax(T_raw, 1.0)
        P_safe_pa = ca.fmax(P_raw * 1e5, 100.0)
        R_gas = 8.314

        a_comp, b_comp = [], []
        for comp in self.components:
            data = self.thermo_data[comp]
            Tr = T_safe / ca.fmax(data["Tc"], 1.0)
            m = 0.480 + 1.574 * data["omega"] - 0.176 * data["omega"] ** 2
            alpha = (1.0 + m * (1.0 - ca.sqrt(Tr))) ** 2
            Pc_pa = data["Pc"] * 1e5
            a_comp.append(0.42748 * (R_gas**2 * data["Tc"] ** 2 / Pc_pa) * alpha)
            b_comp.append(0.08664 * (R_gas * data["Tc"] / Pc_pa))

        def get_cubic_params(fractions_dict):
            sum_fracs = ca.fmax(ca.sum1(ca.vertcat(*[fractions_dict[c].symbolic_object for c in self.components])), 1e-12)
            a_mix, b_mix = 0.0, 0.0
            for i in range(len(self.components)):
                xi = fractions_dict[self.components[i]].symbolic_object / sum_fracs
                b_mix += xi * b_comp[i]
                for j in range(len(self.components)):
                    xj = fractions_dict[self.components[j]].symbolic_object / sum_fracs
                    a_mix += xi * xj * ca.sqrt(a_comp[i] * a_comp[j]) * (1.0 - self.kij_matrix[i, j])
            A_mix = (a_mix * P_safe_pa) / (R_gas**2 * T_safe**2)
            B_mix = (b_mix * P_safe_pa) / (R_gas * T_safe)
            return a_mix, b_mix, A_mix, B_mix

        aL, bL, A_L, B_L = get_cubic_params(stream_instance.x)
        aV, bV, A_V, B_V = get_cubic_params(stream_instance.y)

        Z_L = stream_instance.Z_L.symbolic_object
        Z_V = stream_instance.Z_V.symbolic_object

        res_L = Z_L**3 - Z_L**2 + (A_L - B_L - B_L**2) * Z_L - (A_L * B_L)
        res_V = Z_V**3 - Z_V**2 + (A_V - B_V - B_V**2) * Z_V - (A_V * B_V)

        stream_instance.createEquation("EOS_Cubic_Liquid", expr=EquationNode("resL", res_L, Unit("", "")))
        stream_instance.createEquation("EOS_Cubic_Vapor", expr=EquationNode("resV", res_V, Unit("", "")))

        def calc_phi(comp, Z, A, B, b_mix, a_mix, i_idx):
            bi_b = b_comp[i_idx] / ca.fmax(b_mix, 1e-12)
            sum_x_aij = 0.0
            for j in range(len(self.components)):
                frac = stream_instance.x[self.components[j]].symbolic_object if Z is Z_L else stream_instance.y[self.components[j]].symbolic_object
                sum_x_aij += ca.fmax(frac, 0.0) * ca.sqrt(a_comp[i_idx] * a_comp[j]) * (1.0 - self.kij_matrix[i_idx, j])

            sum_fracs = ca.fmax(
                ca.sum1(ca.vertcat(*[ca.fmax(stream_instance.x[c].symbolic_object if Z is Z_L else stream_instance.y[c].symbolic_object, 0.0) for c in self.components])),
                1e-12,
            )
            term_a = (2.0 * sum_x_aij / (ca.fmax(a_mix, 1e-12) * sum_fracs)) - bi_b
            arg1 = Z - B
            arg2 = Z + B

            ln_phi = bi_b * (Z - 1.0) - leaky_log(arg1) - (A / ca.fmax(B, 1e-12)) * term_a * leaky_log(arg2 / ca.fmax(Z, 1e-8))
            return ca.exp(ca.fmax(ca.fmin(ln_phi, 20.0), -20.0))

        for i, comp in enumerate(self.components):
            phi_L = calc_phi(comp, Z_L, A_L, B_L, bL, aL, i)
            phi_V = calc_phi(comp, Z_V, A_V, B_V, bV, aV, i)
            x_c = stream_instance.x[comp].symbolic_object
            y_c = stream_instance.y[comp].symbolic_object
            K_i = phi_L / ca.fmax(phi_V, 1e-12)
            stream_instance.createEquation(f"IsoFugacity_{comp}", expr=EquationNode("isoF", y_c - K_i * x_c, Unit("", "")))