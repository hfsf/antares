# -*- coding: utf-8 -*-

"""
Stream Module (V5 Native CasADi Architecture).

Defines the core topological flow entities: MaterialStream and EnergyStream.
In the ANTARES framework, streams are not merely variables; they are fully-fledged
topological submodels. They encapsulate mass and energy conservation constraints.

By adhering to SOLID principles, thermodynamic calculations have been completely
decoupled from this module. MaterialStreams now rely entirely on injected
'PropertyPackage' objects (from thermo_package.py) to resolve thermodynamic
state calculations dynamically.
"""

from .model import Model


class MaterialStream(Model):
    """
    MaterialStream refactored to comply with SOLID principles.
    Delegates all property evaluations to an external PropertyPackage.
    """

    def __init__(self, name, property_package, description=""):
        self.property_package = property_package
        self.components = property_package.components

        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        # A MÁGICA ESTÁ AQUI: Adicionar exposure_type="algebraic" a todas as variáveis de estado!
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
            description="Molar Enthalpy of the Mixture",
            value=0.0,
            exposure_type="algebraic",
        )

        # Composition vector setup
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
        # 1. Physical Constraint: Molar fractions must strictly sum to 1.0
        sum_fractions = sum(self.z[comp]() for comp in self.components)
        self.createEquation(
            "Fractions_Sum",
            description="Molar fractions unity sum",
            expr=sum_fractions - 1.0,
        )

        # 2. Thermodynamic Resolution Delegation
        z_sym_dict = {comp: self.z[comp]() for comp in self.components}

        h_expression = self.property_package.get_enthalpy_expression(
            T_sym=self.T(), P_sym=self.P(), z_sym_dict=z_sym_dict
        )

        self.createEquation(
            "Enthalpy_Thermodynamic_Closure",
            description="Enthalpy closure via Property Package",
            expr=self.H_molar() - h_expression,
        )


class EnergyStream(Model):
    """
    Represents a pure energy flow within the flowsheet.
    """

    def __init__(self, name, stream_type="heat", description=""):
        self.stream_type = str(stream_type).lower()
        if self.stream_type not in ["heat", "work"]:
            raise ValueError(f"EnergyStream '{name}' must be of type 'heat' or 'work'.")

        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        if self.stream_type == "heat":
            self.Q = self.createVariable(
                "Q",
                "W",
                description="Thermal Heat Transfer Rate",
                value=0.0,
                exposure_type="algebraic",
            )
            self.T_source = self.createVariable(
                "T_source",
                "K",
                description="Temperature of the Heat Source/Sink",
                value=298.15,
                exposure_type="algebraic",
            )

        elif self.stream_type == "work":
            self.W = self.createVariable(
                "W",
                "W",
                description="Mechanical or Electrical Power Rate",
                value=0.0,
                exposure_type="algebraic",
            )
            self.RPM = self.createVariable(
                "RPM",
                "rev/min",
                description="Rotational Speed",
                value=0.0,
                exposure_type="algebraic",
            )

    def DeclareEquations(self):
        pass
