# -*- coding: utf-8 -*-

"""
Thermodynamic Package Module (V5 Native CasADi Architecture).

Implements the SOLID Strategy pattern. Thermodynamic Packages contain the physics
and empirical data retrieval logic, while Streams contain the spatial topology.
These packages dynamically inject variables and complex residual equations
directly into the Stream's DAE block, preserving native CasADi C++ acceleration.

Supported Models:
- PureFluidLUT (B-Spline Tables)
- IdealVLEPackage (Wilson K-Values)
- PengRobinsonEOS (Rigorous Cubic - Verified Analytic Architecture)
- SoaveRedlichKwongEOS (Rigorous Cubic - Verified Analytic Architecture)
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
    def __init__(self, components):
        self.components = components

    @abstractmethod
    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
        pass

    @abstractmethod
    def build_phase_equilibrium(self, stream_instance):
        pass

    def _apply_asymmetric_initialization(self, stream_instance):
        """Forces distinct initial starting points for phase fractions."""
        n = len(self.components)
        if n < 2:
            return
        for i, comp in enumerate(self.components):
            val_x = 0.8 if i == 0 else 0.2 / (n - 1)
            val_y = 0.2 if i == 0 else 0.8 / (n - 1)
            for var_dict, val in [
                (stream_instance.x, val_x),
                (stream_instance.y, val_y),
            ]:
                var = var_dict[comp]
                if (
                    hasattr(var, "initial_condition_array")
                    and var.initial_condition_array is not None
                ):
                    var.initial_condition_array[:] = val
                if hasattr(var, "setValue"):
                    var.setValue(val)
                elif hasattr(var, "value"):
                    var.value = val


class PureFluidLUT(PropertyPackage):
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
        chem = Chemical(fluid_name)
        for i, T in enumerate(self.T_grid):
            for j, P in enumerate(P_grid_pa):
                chem.calculate(T, P)
                self.H_matrix[i, j] = getattr(chem, "H", 0.0)

        self._h_flat = self.H_matrix.ravel(order="F")
        self.interpolant = ca.interpolant(
            f"H_LUT_{self.fluid_name}",
            "bspline",
            [self.T_grid, self.P_grid_bar],
            self._h_flat,
        )

    def get_phase_enthalpy_expression(
        self, T_sym, P_sym, fractions_sym_dict, phase="liquid"
    ):
        T_raw = T_sym.symbolic_object if hasattr(T_sym, "symbolic_object") else T_sym
        P_raw = P_sym.symbolic_object if hasattr(P_sym, "symbolic_object") else P_sym
        h_mx = self.interpolant(ca.vertcat(ca.fmax(T_raw, 1.0), ca.fmax(P_raw, 0.01)))
        return EquationNode(
            name=f"H_LUT_{self.fluid_name}",
            symbolic_object=h_mx,
            unit_object=Unit("", "J/mol"),
        )

    def build_phase_equilibrium(self, stream_instance):
        pass


class IdealVLEPackage(PropertyPackage):
    def __init__(self, components):
        super().__init__(components)
        self.thermo_data = {}

    def fetch_parameters_from_db(self):
        if not HAS_THERMO:
            raise ImportError("The 'thermo' library is required.")
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
                "Hf_298": getattr(chem, "Hfgm", 0.0) or 0.0,
                "Tc": getattr(chem, "Tc", 300.0) or 300.0,
                "Pc": getattr(chem, "Pc", 1e5) or 1e5,
                "omega": getattr(chem, "omega", 0.0) or 0.0,
                "Hvap_298": getattr(chem, "Hvapm", 30000.0) or 30000.0,
                "Cp_coeffs": coeffs[:4],
            }

    def build_phase_equilibrium(self, stream_instance):
        self._apply_asymmetric_initialization(stream_instance)

        T_mx = (
            stream_instance.T.symbolic_object
            if hasattr(stream_instance.T, "symbolic_object")
            else stream_instance.T
        )
        P_mx = (
            stream_instance.P.symbolic_object
            if hasattr(stream_instance.P, "symbolic_object")
            else stream_instance.P
        )

        T_safe = ca.fmax(T_mx, 1.0)
        P_pa = ca.fmax(P_mx * 1e5, 100.0)

        for comp in self.components:
            data = self.thermo_data[comp]
            K_sym = (data["Pc"] / P_pa) * ca.exp(
                5.373 * (1.0 + data["omega"]) * (1.0 - (data["Tc"] / T_safe))
            )
            K_node = EquationNode(
                name=f"K_{comp}", symbolic_object=K_sym, unit_object=Unit("", "")
            )

            x_c = stream_instance.x[comp]()
            y_c = stream_instance.y[comp]()
            stream_instance.createEquation(
                f"VLE_Partition_{comp}", expr=y_c - (K_node * x_c)
            )

    def get_phase_enthalpy_expression(self, T_sym, P_sym, fractions_sym_dict, phase):
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
            z_mx = (
                fractions_sym_dict[comp].symbolic_object
                if hasattr(fractions_sym_dict[comp], "symbolic_object")
                else fractions_sym_dict[comp]
            )
            h_mix_sym += z_mx * h_comp

        return EquationNode(
            name=f"H_{phase}", symbolic_object=h_mix_sym, unit_object=Unit("", "J/mol")
        )


class PengRobinsonEOS(IdealVLEPackage):
    def build_phase_equilibrium(self, stream_instance):
        self._apply_asymmetric_initialization(stream_instance)

        stream_instance.Z_L = stream_instance.createVariable(
            "Z_L", "", value=0.05, exposure_type="algebraic"
        )
        stream_instance.Z_V = stream_instance.createVariable(
            "Z_V", "", value=0.95, exposure_type="algebraic"
        )

        T_mx = stream_instance.T.symbolic_object
        P_mx = stream_instance.P.symbolic_object * 1e5

        T_safe = ca.fmax(T_mx, 1.0)
        P_safe = ca.fmax(P_mx, 100.0)
        R_gas = 8.314

        a_comp, b_comp = [], []
        for comp in self.components:
            data = self.thermo_data[comp]
            Tr = T_safe / ca.fmax(data["Tc"], 1.0)
            kappa = 0.37464 + 1.54226 * data["omega"] - 0.26992 * data["omega"] ** 2
            alpha = (1.0 + kappa * (1.0 - ca.sqrt(Tr))) ** 2
            a_comp.append(0.45724 * (R_gas**2 * data["Tc"] ** 2 / data["Pc"]) * alpha)
            b_comp.append(0.07780 * (R_gas * data["Tc"] / data["Pc"]))

        aL, bL, aV, bV = 0.0, 0.0, 0.0, 0.0
        for i, comp_i in enumerate(self.components):
            xi = stream_instance.x[comp_i].symbolic_object
            yi = stream_instance.y[comp_i].symbolic_object
            bL += xi * b_comp[i]
            bV += yi * b_comp[i]
            for j, comp_j in enumerate(self.components):
                xj = stream_instance.x[comp_j].symbolic_object
                yj = stream_instance.y[comp_j].symbolic_object
                aL += xi * xj * ca.sqrt(a_comp[i] * a_comp[j])
                aV += yi * yj * ca.sqrt(a_comp[i] * a_comp[j])

        A_L = (aL * P_safe) / (R_gas**2 * T_safe**2)
        B_L = (bL * P_safe) / (R_gas * T_safe)
        A_V = (aV * P_safe) / (R_gas**2 * T_safe**2)
        B_V = (bV * P_safe) / (R_gas * T_safe)

        Z_L = stream_instance.Z_L.symbolic_object
        Z_V = stream_instance.Z_V.symbolic_object

        res_L = (
            Z_L**3
            - (1.0 - B_L) * Z_L**2
            + (A_L - 2.0 * B_L - 3.0 * B_L**2) * Z_L
            - (A_L * B_L - B_L**2 - B_L**3)
        )
        res_V = (
            Z_V**3
            - (1.0 - B_V) * Z_V**2
            + (A_V - 2.0 * B_V - 3.0 * B_V**2) * Z_V
            - (A_V * B_V - B_V**2 - B_V**3)
        )

        stream_instance.createEquation(
            "EOS_Cubic_Liquid", expr=EquationNode("resL", res_L, Unit("", ""))
        )
        stream_instance.createEquation(
            "EOS_Cubic_Vapor", expr=EquationNode("resV", res_V, Unit("", ""))
        )

        for i, comp in enumerate(self.components):
            bi_b_L = b_comp[i] / ca.fmax(bL, 1e-12)
            sum_x_aij_L = 0.0
            for j, comp_j in enumerate(self.components):
                xj = stream_instance.x[comp_j].symbolic_object
                sum_x_aij_L += xj * ca.sqrt(a_comp[i] * a_comp[j])
            term_a_L = (2.0 * sum_x_aij_L / ca.fmax(aL, 1e-12)) - bi_b_L

            arg1_L = ca.fmax(Z_L - B_L, 1e-8)
            arg2_L = ca.fmax(Z_L + (1.0 + 1.41421356) * B_L, 1e-8)
            arg3_L = ca.fmax(Z_L + (1.0 - 1.41421356) * B_L, 1e-8)
            ln_phi_L = (
                bi_b_L * (Z_L - 1.0)
                - ca.log(arg1_L)
                - (A_L / ca.fmax(2.82842712 * B_L, 1e-12))
                * term_a_L
                * ca.log(arg2_L / arg3_L)
            )

            bi_b_V = b_comp[i] / ca.fmax(bV, 1e-12)
            sum_y_aij_V = 0.0
            for j, comp_j in enumerate(self.components):
                yj = stream_instance.y[comp_j].symbolic_object
                sum_y_aij_V += yj * ca.sqrt(a_comp[i] * a_comp[j])
            term_a_V = (2.0 * sum_y_aij_V / ca.fmax(aV, 1e-12)) - bi_b_V

            arg1_V = ca.fmax(Z_V - B_V, 1e-8)
            arg2_V = ca.fmax(Z_V + (1.0 + 1.41421356) * B_V, 1e-8)
            arg3_V = ca.fmax(Z_V + (1.0 - 1.41421356) * B_V, 1e-8)
            ln_phi_V = (
                bi_b_V * (Z_V - 1.0)
                - ca.log(arg1_V)
                - (A_V / ca.fmax(2.82842712 * B_V, 1e-12))
                * term_a_V
                * ca.log(arg2_V / arg3_V)
            )

            delta_ln = ca.fmax(ca.fmin(ln_phi_L - ln_phi_V, 25.0), -25.0)
            K_i = ca.exp(delta_ln)

            x_c = stream_instance.x[comp].symbolic_object
            y_c = stream_instance.y[comp].symbolic_object

            stream_instance.createEquation(
                f"IsoFugacity_{comp}",
                expr=EquationNode("isoF", y_c - K_i * x_c, Unit("", "")),
            )


class SoaveRedlichKwongEOS(IdealVLEPackage):
    def build_phase_equilibrium(self, stream_instance):
        self._apply_asymmetric_initialization(stream_instance)

        stream_instance.Z_L = stream_instance.createVariable(
            "Z_L", "", value=0.05, exposure_type="algebraic"
        )
        stream_instance.Z_V = stream_instance.createVariable(
            "Z_V", "", value=0.95, exposure_type="algebraic"
        )

        T_mx = stream_instance.T.symbolic_object
        P_mx = stream_instance.P.symbolic_object * 1e5

        T_safe = ca.fmax(T_mx, 1.0)
        P_safe = ca.fmax(P_mx, 100.0)
        R_gas = 8.314

        a_comp, b_comp = [], []
        for comp in self.components:
            data = self.thermo_data[comp]
            Tr = T_safe / ca.fmax(data["Tc"], 1.0)
            m = 0.480 + 1.574 * data["omega"] - 0.176 * data["omega"] ** 2
            alpha = (1.0 + m * (1.0 - ca.sqrt(Tr))) ** 2
            a_comp.append(0.42748 * (R_gas**2 * data["Tc"] ** 2 / data["Pc"]) * alpha)
            b_comp.append(0.08664 * (R_gas * data["Tc"] / data["Pc"]))

        aL, bL, aV, bV = 0.0, 0.0, 0.0, 0.0
        for i, comp_i in enumerate(self.components):
            xi = stream_instance.x[comp_i].symbolic_object
            yi = stream_instance.y[comp_i].symbolic_object
            bL += xi * b_comp[i]
            bV += yi * b_comp[i]
            for j, comp_j in enumerate(self.components):
                xj = stream_instance.x[comp_j].symbolic_object
                yj = stream_instance.y[comp_j].symbolic_object
                aL += xi * xj * ca.sqrt(a_comp[i] * a_comp[j])
                aV += yi * yj * ca.sqrt(a_comp[i] * a_comp[j])

        A_L = (aL * P_safe) / (R_gas**2 * T_safe**2)
        B_L = (bL * P_safe) / (R_gas * T_safe)
        A_V = (aV * P_safe) / (R_gas**2 * T_safe**2)
        B_V = (bV * P_safe) / (R_gas * T_safe)

        Z_L = stream_instance.Z_L.symbolic_object
        Z_V = stream_instance.Z_V.symbolic_object

        res_L = Z_L**3 - Z_L**2 + (A_L - B_L - B_L**2) * Z_L - (A_L * B_L)
        res_V = Z_V**3 - Z_V**2 + (A_V - B_V - B_V**2) * Z_V - (A_V * B_V)

        stream_instance.createEquation(
            "EOS_Cubic_Liquid", expr=EquationNode("resL", res_L, Unit("", ""))
        )
        stream_instance.createEquation(
            "EOS_Cubic_Vapor", expr=EquationNode("resV", res_V, Unit("", ""))
        )

        for i, comp in enumerate(self.components):
            bi_b_L = b_comp[i] / ca.fmax(bL, 1e-12)
            sum_x_aij_L = 0.0
            for j, comp_j in enumerate(self.components):
                xj = stream_instance.x[comp_j].symbolic_object
                sum_x_aij_L += xj * ca.sqrt(a_comp[i] * a_comp[j])
            term_a_L = (2.0 * sum_x_aij_L / ca.fmax(aL, 1e-12)) - bi_b_L

            arg1_L = ca.fmax(Z_L - B_L, 1e-8)
            arg2_L = 1.0 + B_L / ca.fmax(Z_L, 1e-8)
            ln_phi_L = (
                bi_b_L * (Z_L - 1.0)
                - ca.log(arg1_L)
                - (A_L / ca.fmax(B_L, 1e-12)) * term_a_L * ca.log(arg2_L)
            )

            bi_b_V = b_comp[i] / ca.fmax(bV, 1e-12)
            sum_y_aij_V = 0.0
            for j, comp_j in enumerate(self.components):
                yj = stream_instance.y[comp_j].symbolic_object
                sum_y_aij_V += yj * ca.sqrt(a_comp[i] * a_comp[j])
            term_a_V = (2.0 * sum_y_aij_V / ca.fmax(aV, 1e-12)) - bi_b_V

            arg1_V = ca.fmax(Z_V - B_V, 1e-8)
            arg2_V = 1.0 + B_V / ca.fmax(Z_V, 1e-8)
            ln_phi_V = (
                bi_b_V * (Z_V - 1.0)
                - ca.log(arg1_V)
                - (A_V / ca.fmax(B_V, 1e-12)) * term_a_V * ca.log(arg2_V)
            )

            delta_ln = ca.fmax(ca.fmin(ln_phi_L - ln_phi_V, 25.0), -25.0)
            K_i = ca.exp(delta_ln)

            x_c = stream_instance.x[comp].symbolic_object
            y_c = stream_instance.y[comp].symbolic_object

            stream_instance.createEquation(
                f"IsoFugacity_{comp}",
                expr=EquationNode("isoF", y_c - K_i * x_c, Unit("", "")),
            )


class NRTLPackage(IdealVLEPackage):
    def __init__(self, components):
        super().__init__(components)
        n = len(components)
        self.tau_matrix = np.zeros((n, n))
        self.alpha_matrix = np.full((n, n), 0.3)

    def set_binary_parameters(self, tau_matrix, alpha_matrix):
        self.tau_matrix = tau_matrix
        self.alpha_matrix = alpha_matrix

    def _get_lee_kesler_psat(self, T_mx, comp):
        Tc = self.thermo_data[comp]["Tc"]
        Pc = self.thermo_data[comp]["Pc"]
        w = self.thermo_data[comp]["omega"]

        Tr = T_mx / Tc
        f0 = (
            5.92714
            - 6.09648 / ca.fmax(Tr, 1e-5)
            - 1.28862 * ca.log(ca.fmax(Tr, 1e-5))
            + 0.169347 * Tr**6
        )
        f1 = (
            15.2518
            - 15.6875 / ca.fmax(Tr, 1e-5)
            - 13.4721 * ca.log(ca.fmax(Tr, 1e-5))
            + 0.43577 * Tr**6
        )

        ln_Pr_sat = f0 + w * f1
        return Pc * ca.exp(ca.fmax(ca.fmin(ln_Pr_sat, 50.0), -50.0))

    def build_phase_equilibrium(self, stream_instance):
        self._apply_asymmetric_initialization(stream_instance)

        T_mx = (
            stream_instance.T.symbolic_object
            if hasattr(stream_instance.T, "symbolic_object")
            else stream_instance.T
        )
        P_mx = (
            stream_instance.P.symbolic_object
            if hasattr(stream_instance.P, "symbolic_object")
            else stream_instance.P
        )
        P_pa = ca.fmax(P_mx * 1e5, 100.0)

        n = len(self.components)
        G = ca.MX.zeros(n, n)
        tau = ca.MX.zeros(n, n)

        for i in range(n):
            for j in range(n):
                tau[i, j] = self.tau_matrix[i, j]
                G[i, j] = ca.exp(
                    ca.fmax(ca.fmin(-self.alpha_matrix[i, j] * tau[i, j], 50.0), -50.0)
                )

        x_vec = ca.vertcat(
            *[stream_instance.x[c].symbolic_object for c in self.components]
        )

        for i, comp in enumerate(self.components):
            sum_Gj_xj = ca.fmax(ca.sum1(G[:, i] * x_vec), 1e-12)
            term1 = ca.sum1(tau[:, i] * G[:, i] * x_vec) / sum_Gj_xj

            term2 = 0.0
            for j in range(n):
                num = x_vec[j] * G[i, j]
                den = ca.fmax(ca.sum1(G[:, j] * x_vec), 1e-12)
                sub_term = tau[i, j] - (ca.sum1(x_vec * tau[:, j] * G[:, j]) / den)
                term2 += (num / den) * sub_term

            gamma_i = ca.exp(ca.fmax(ca.fmin(term1 + term2, 50.0), -50.0))
            P_sat_i = self._get_lee_kesler_psat(ca.fmax(T_mx, 1.0), comp)

            x_c = stream_instance.x[comp].symbolic_object
            y_c = stream_instance.y[comp].symbolic_object

            eq_gamma_phi = (y_c * P_pa) - (x_c * gamma_i * P_sat_i)

            stream_instance.createEquation(
                f"VLE_Gamma_Phi_{comp}",
                description=f"NRTL Gamma-Phi Equilibrium for {comp}",
                expr=EquationNode(f"VLE_{comp}", eq_gamma_phi, Unit("", "")),
            )
