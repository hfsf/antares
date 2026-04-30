# -*- coding: utf-8 -*-

"""
Stream Module (V5 Native CasADi Architecture).
"""

from .model import Model


class MaterialStream(Model):
    def __init__(self, name, property_package, description=""):
        self.property_package = property_package
        self.components = property_package.components
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        self.T = self.createVariable(
            "T",
            "K",
            description="Absolute Temperature",
            value=298.15,
            exposure_type="algebraic",
        )
        self.P = self.createVariable(
            "P",
            "bar",
            description="Absolute Pressure",
            value=1.01325,
            exposure_type="algebraic",
        )
        self.F_molar = self.createVariable(
            "F_molar",
            "mol/s",
            description="Total Molar Flow Rate",
            value=100.0,
            exposure_type="algebraic",
        )

        self.H_molar = self.createVariable(
            "H_molar",
            "J/mol",
            description="Molar Enthalpy",
            value=25000.0,
            exposure_type="algebraic",
        )

        self.z = {}
        for comp in self.components:
            self.z[comp] = self.createVariable(
                f"z_{comp}",
                "",
                description=f"Molar fraction of {comp}",
                lower_bound=0.0,
                upper_bound=1.0,
                value=1.0 / len(self.components),
                exposure_type="algebraic",
            )

    def DeclareEquations(self):
        sum_fractions = sum(self.z[comp]() for comp in self.components)
        self.createEquation(
            "Fractions_Sum",
            description="Molar fractions unity sum",
            expr=sum_fractions - 1.0,
        )

        z_sym_dict = {comp: self.z[comp]() for comp in self.components}
        h_expression = self.property_package.get_phase_enthalpy_expression(
            self.T(), self.P(), z_sym_dict, phase="vapor"
        )

        # FIX MACRO: SCALING DA ENTALPIA.
        # Impede que o Newton-step do KINSOL exploda ao tentar corrigir milhões de Joules.
        self.createEquation(
            "Enthalpy_Closure",
            description="Enthalpy closure (Scaled)",
            expr=(self.H_molar() - h_expression) / 1e5,
        )


class TwoPhaseStream(Model):
    def __init__(self, name, property_package, description=""):
        self.property_package = property_package
        self.components = property_package.components
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        self.T = self.createVariable(
            "T",
            "K",
            description="Equilibrium Temperature",
            value=350.0,
            exposure_type="algebraic",
        )
        self.P = self.createVariable(
            "P",
            "bar",
            description="Equilibrium Pressure",
            value=1.01325,
            exposure_type="algebraic",
        )
        self.F_molar = self.createVariable(
            "F_molar",
            "mol/s",
            description="Total Molar Flow Rate",
            value=100.0,
            exposure_type="algebraic",
        )
        self.H_molar = self.createVariable(
            "H_molar",
            "J/mol",
            description="Overall Molar Enthalpy",
            value=25000.0,
            exposure_type="algebraic",
        )

        self.V_frac = self.createVariable(
            "V_frac",
            "",
            description="Vapor Phase Molar Fraction",
            lower_bound=-0.05,
            upper_bound=1.05,
            value=0.5,
            exposure_type="algebraic",
        )

        self.z = {}
        self.x = {}
        self.y = {}
        n_comp = len(self.components)

        for comp in self.components:
            self.z[comp] = self.createVariable(
                f"z_{comp}", "", value=1.0 / n_comp, exposure_type="algebraic"
            )
            self.x[comp] = self.createVariable(
                f"x_{comp}", "", value=0.1, exposure_type="algebraic"
            )
            self.y[comp] = self.createVariable(
                f"y_{comp}", "", value=0.9, exposure_type="algebraic"
            )

    def DeclareEquations(self):
        sum_z = sum(self.z[comp]() for comp in self.components)
        self.createEquation(
            "Overall_Fractions_Sum",
            description="Overall fractions unity sum",
            expr=sum_z - 1.0,
        )

        sum_y = sum(self.y[comp]() for comp in self.components)
        sum_x = sum(self.x[comp]() for comp in self.components)
        self.createEquation(
            "Rachford_Rice_Objective",
            description="Phase fractions strict closure",
            expr=sum_y - sum_x,
        )

        for comp in self.components:
            z_c = self.z[comp]()
            x_c = self.x[comp]()
            y_c = self.y[comp]()
            v_f = self.V_frac()
            mass_bal_expr = z_c - (v_f * y_c + (1.0 - v_f) * x_c)
            self.createEquation(
                f"Mass_Bal_{comp}",
                description=f"Phase allocation for {comp}",
                expr=mass_bal_expr,
            )

        self.property_package.build_phase_equilibrium(self)

        x_sym_dict = {comp: self.x[comp]() for comp in self.components}
        y_sym_dict = {comp: self.y[comp]() for comp in self.components}

        H_liq_expr = self.property_package.get_phase_enthalpy_expression(
            self.T(), self.P(), x_sym_dict, phase="liquid"
        )
        H_vap_expr = self.property_package.get_phase_enthalpy_expression(
            self.T(), self.P(), y_sym_dict, phase="vapor"
        )

        global_h_expr = self.H_molar() - (
            self.V_frac() * H_vap_expr + (1.0 - self.V_frac()) * H_liq_expr
        )

        # FIX MACRO: SCALING DA ENTALPIA VLE.
        # Proteção absoluta contra overflow de escala numérica na EO Matrix.
        self.createEquation(
            "Global_Enthalpy_Closure",
            description="Two-phase enthalpy mix (Scaled)",
            expr=global_h_expr / 1e5,
        )


class EnergyStream(Model):
    def __init__(self, name, stream_type="heat", description=""):
        self.stream_type = str(stream_type).lower()
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        if self.stream_type == "heat":
            self.Q = self.createVariable("Q", "W", value=1e6, exposure_type="algebraic")
            self.T_source = self.createVariable(
                "T_source", "K", value=298.15, exposure_type="algebraic"
            )
        elif self.stream_type == "work":
            self.W = self.createVariable("W", "W", value=0.0, exposure_type="algebraic")
            self.RPM = self.createVariable(
                "RPM", "rev/min", value=0.0, exposure_type="algebraic"
            )

    def DeclareEquations(self):
        pass
