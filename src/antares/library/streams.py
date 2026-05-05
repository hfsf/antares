# -*- coding: utf-8 -*-

r"""
Stream Module (V5 Native CasADi Architecture).

Defines the core topological flow entities: MaterialStream, TwoPhaseStream,
and EnergyStream. In the ANTARES framework, streams are fully-fledged topological
submodels. They autonomously encapsulate mass constraints, energy flow calculations, 
and phase equilibrium interactions, preventing abstraction leaks in the unit operations.

In this update, strict physical boundaries (e.g., Absolute Temperature > 0, 
fractions within [0, 1]) are natively injected during variable declaration. 
This provides robust out-of-the-box support for Non-Linear Programming (NLP) 
solvers like IPOPT, avoiding non-physical domains during Newton steps.
"""

from ..core.model import Model


class MaterialStream(Model):
    r"""
    Single-Phase Material Stream.
    
    Represents a standard macroscopic single-phase fluid stream (vapor or ideal liquid). 
    It delegates intensive thermodynamic property calculations (like molar enthalpy) 
    to the injected PropertyPackage.
    """

    def __init__(self, name, property_package, description=""):
        """
        Initializes the single-phase material stream.

        :param str name: The unique identifier for the stream.
        :param PropertyPackage property_package: The thermodynamic engine linked to this stream.
        :param str description: Optional physical description.
        """
        self.property_package = property_package
        self.components = property_package.components
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        """
        Declares the macroscopic physical properties of the stream.
        Strict thermodynamic bounds are applied to protect NLP solvers.
        """
        self.T = self.createVariable(
            "T", "K", description="Absolute Temperature", value=298.15, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-2
        )
        self.P = self.createVariable(
            "P", "bar", description="Absolute Pressure", value=1.01325, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-5
        )
        self.F_molar = self.createVariable(
            "F_molar", "mol/s", description="Total Molar Flow Rate", value=100.0, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=0.0
        )
        # Enthalpy can be mathematically negative depending on the reference state
        self.H_molar = self.createVariable(
            "H_molar", "J/mol", description="Molar Enthalpy", value=25000.0, exposure_type="algebraic"
        )

        self.z = {}
        n_comp = len(self.components)
        for comp in self.components:
            self.z[comp] = self.createVariable(
                f"z_{comp}", "", description=f"Molar fraction of {comp}",
                is_lower_bounded=True, lower_bound=0.0, 
                is_upper_bounded=True, upper_bound=1.0, 
                value=1.0 / n_comp, exposure_type="algebraic",
            )

    def DeclareEquations(self):
        """
        Formulates the macroscopic unity and energy constraints.
        """
        sum_fractions = sum(self.z[comp]() for comp in self.components)
        self.createEquation(
            "Fractions_Sum", description="Molar fractions unity sum", expr=sum_fractions - 1.0,
        )

        z_sym_dict = {comp: self.z[comp]() for comp in self.components}
        h_expression = self.property_package.get_phase_enthalpy_expression(
            self.T(), self.P(), z_sym_dict, phase="vapor"
        )

        self.createEquation(
            "Enthalpy_Closure", description="Intrinsic enthalpy evaluation", expr=self.H_molar() - h_expression,
        )
        
    def Energy_Flow(self):
        r"""
        Extrinsic physical property method.
        Returns the symbolic evaluation of the extensive Enthalpy Flow in Watts.
        Bypasses the creation of explicit variables to prevent KINSOL step-length limits.
        
        :return: Extrinsic energy flow (W).
        :rtype: casadi.MX
        """
        return self.F_molar() * self.H_molar()


class TwoPhaseStream(Model):
    r"""
    Multiphase Material Stream handling rigorous equilibrium.

    Inherits the macroscopic topological structure of a standard process stream 
    but expands the internal phenomenological equations to solve the isothermal/isobaric 
    flash autonomously. Bounds are strictly enforced for NLP convergence.
    """

    def __init__(self, name, property_package, regime="VLE", description=""):
        """
        Initializes the multiphase stream.

        :param str name: The unique identifier for the stream.
        :param PropertyPackage property_package: The thermodynamic engine evaluating the phases.
        :param str regime: The phase regime targeted (e.g., 'VLE', 'LLE').
        :param str description: Optional physical description.
        """
        self.property_package = property_package
        self.components = property_package.components
        self.regime = str(regime).upper()
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        r"""
        Declares the multiphase physical properties, phase fractions, and species partitions.
        Enforces $0 \le x, y, z, V_{frac} \le 1$ to shield the solver.
        """
        self.T = self.createVariable(
            "T", "K", description="Equilibrium Temperature", value=350.0, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-2
        )
        self.P = self.createVariable(
            "P", "bar", description="Equilibrium Pressure", value=1.01325, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=1e-5
        )
        self.F_molar = self.createVariable(
            "F_molar", "mol/s", description="Total Molar Flow Rate", value=100.0, exposure_type="algebraic",
            is_lower_bounded=True, lower_bound=0.0
        )
        self.H_molar = self.createVariable(
            "H_molar", "J/mol", description="Overall Molar Enthalpy", value=25000.0, exposure_type="algebraic"
        )

        self.V_frac = self.createVariable(
            "V_frac", "", description="Second Phase Molar Fraction", 
            is_lower_bounded=True, lower_bound=-0.05, 
            is_upper_bounded=True, upper_bound=1.05, 
            value=0.5, exposure_type="algebraic"
        )

        self.z = {}
        self.x = {}
        self.y = {}
        n_comp = len(self.components)

        for comp in self.components:
            self.z[comp] = self.createVariable(
                f"z_{comp}", "", value=1.0 / n_comp, exposure_type="algebraic",
                is_lower_bounded=True, lower_bound=-0.01, is_upper_bounded=True, upper_bound=1.01
            )
            self.x[comp] = self.createVariable(
                f"x_{comp}", "", value=0.1, exposure_type="algebraic",
                is_lower_bounded=True, lower_bound=-0.01, is_upper_bounded=True, upper_bound=1.01
            )
            self.y[comp] = self.createVariable(
                f"y_{comp}", "", value=0.9, exposure_type="algebraic",
                is_lower_bounded=True, lower_bound=-0.01, is_upper_bounded=True, upper_bound=1.01
            )

    def DeclareEquations(self):
        """
        Formulates the macroscopic unity, material balances, Rachford-Rice constraints,
        and triggers the specific thermodynamic equilibrium mappings.
        """
        sum_z = sum(self.z[comp]() for comp in self.components)
        self.createEquation("Overall_Fractions_Sum", description="Overall fractions unity sum", expr=sum_z - 1.0)

        sum_y = sum(self.y[comp]() for comp in self.components)
        sum_x = sum(self.x[comp]() for comp in self.components)
        self.createEquation("Rachford_Rice_Objective", description="Phase fractions strict closure", expr=sum_y - sum_x)

        for comp in self.components:
            z_c = self.z[comp]()
            x_c = self.x[comp]()
            y_c = self.y[comp]()
            v_f = self.V_frac()
            mass_bal_expr = z_c - (v_f * y_c + (1.0 - v_f) * x_c)
            self.createEquation(f"Mass_Bal_{comp}", description=f"Phase allocation for {comp}", expr=mass_bal_expr)

        # Delegate intensive physical property calculations to the thermodynamic engine
        self.property_package.build_phase_equilibrium(self)

        x_sym_dict = {comp: self.x[comp]() for comp in self.components}
        y_sym_dict = {comp: self.y[comp]() for comp in self.components}

        phase_1 = "liquid" if self.regime == "VLE" else "liquid_1"
        phase_2 = "vapor" if self.regime == "VLE" else "liquid_2"

        H_p1_expr = self.property_package.get_phase_enthalpy_expression(self.T(), self.P(), x_sym_dict, phase=phase_1)
        H_p2_expr = self.property_package.get_phase_enthalpy_expression(self.T(), self.P(), y_sym_dict, phase=phase_2)

        global_h_expr = self.H_molar() - (self.V_frac() * H_p2_expr + (1.0 - self.V_frac()) * H_p1_expr)

        self.createEquation("Global_Enthalpy_Closure", description="Two-phase enthalpy mix", expr=global_h_expr)

    def Energy_Flow(self):
        r"""
        Extrinsic physical property method.
        Returns the symbolic evaluation of the extensive Enthalpy Flow in Watts.
        
        :return: Extrinsic energy flow (W).
        :rtype: casadi.MX
        """
        return self.F_molar() * self.H_molar()


class EnergyStream(Model):
    r"""
    Energy Stream defining generic thermal or mechanical transfers.
    Provides purely algebraic energetic variables.
    """
    def __init__(self, name, stream_type="heat", description=""):
        """
        Initializes the generic energy stream.

        :param str name: The unique identifier for the stream.
        :param str stream_type: Defines the interaction format ('heat' or 'work').
        :param str description: Optional physical description.
        """
        self.stream_type = str(stream_type).lower()
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        """
        Declares the specific variables (e.g., Heat Transfer, Work, RPM) depending
        on the required abstraction.
        """
        if self.stream_type == "heat":
            self.Q = self.createVariable("Q", "W", value=1e6, exposure_type="algebraic")
            self.T_source = self.createVariable(
                "T_source", "K", value=298.15, exposure_type="algebraic",
                is_lower_bounded=True, lower_bound=1e-2
            )
        elif self.stream_type == "work":
            self.W = self.createVariable("W", "W", value=0.0, exposure_type="algebraic")
            self.RPM = self.createVariable("RPM", "rev/min", value=0.0, exposure_type="algebraic")

    def DeclareEquations(self):
        """
        Energy streams generally act as free boundary elements and do not inherently
        formulate closure equations unless coupled with unit operations.
        """
        pass