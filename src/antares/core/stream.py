# -*- coding: utf-8 -*-

"""
Stream Module (V5 Native CasADi Architecture).

Defines the core topological flow entities: MaterialStream, TwoPhaseStream,
and EnergyStream. In the ANTARES framework, streams are fully-fledged topological
submodels encapsulating mass, energy, and phase equilibrium constraints.
Includes robust baseline initializations to prevent KINSOL step-limit truncation.
"""

from .model import Model


class MaterialStream(Model):
    """
    Single-Phase MaterialStream.
    
    Represents a standard single-phase fluid stream (vapor or ideal liquid). 
    It delegates intensive thermodynamic property calculations (like enthalpy) 
    to the injected PropertyPackage.
    """

    def __init__(self, name, property_package, description=""):
        """
        Initializes the MaterialStream.

        :param str name: Unique identifier for the stream instance.
        :param PropertyPackage property_package: The thermodynamic package used to evaluate physical properties.
        :param str description: Optional physical description of the stream.
        """
        self.property_package = property_package
        self.components = property_package.components
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        """
        Declares the fundamental state variables of the stream (T, P, F_molar, H_molar, and compositions).
        Injects a realistic baseline for H_molar to prevent KINSOL Newton-step saturation.
        """
        self.T = self.createVariable("T", "K", description="Absolute Temperature", value=298.15, exposure_type="algebraic")
        self.P = self.createVariable("P", "bar", description="Absolute Pressure", value=1.01325, exposure_type="algebraic")
        self.F_molar = self.createVariable("F_molar", "mol/s", description="Total Molar Flow Rate", value=100.0, exposure_type="algebraic")
        
        # FIX: Realistic baseline to expand the solver's Newton Step Norm (mxnewtstep)
        self.H_molar = self.createVariable("H_molar", "J/mol", description="Molar Enthalpy", value=25000.0, exposure_type="algebraic")

        self.z = {}
        for comp in self.components:
            self.z[comp] = self.createVariable(
                f"z_{comp}", "", description=f"Molar fraction of {comp}",
                lower_bound=0.0, upper_bound=1.0, value=1.0 / len(self.components), exposure_type="algebraic"
            )

    def DeclareEquations(self):
        """
        Declares the conservation equations for the single-phase stream, 
        including the molar fraction unity sum and the enthalpy closure constraint.
        """
        sum_fractions = sum(self.z[comp]() for comp in self.components)
        self.createEquation("Fractions_Sum", description="Molar fractions unity sum", expr=sum_fractions - 1.0)

        z_sym_dict = {comp: self.z[comp]() for comp in self.components}
        h_expression = self.property_package.get_phase_enthalpy_expression(self.T(), self.P(), z_sym_dict, phase="vapor")
        self.createEquation("Enthalpy_Closure", description="Enthalpy closure", expr=self.H_molar() - h_expression)


class TwoPhaseStream(Model):
    """
    Two-Phase Material Stream handling rigorous Vapor-Liquid Equilibrium (VLE).

    Inherits the topological structure of a process stream but expands the
    phenomenological equations to solve the isothermal/isobaric flash
    simultaneously within the global CasADi Equation-Oriented (EO) matrix.
    """

    def __init__(self, name, property_package, description=""):
        """
        Initializes the TwoPhaseStream.

        :param str name: Unique identifier for the stream instance.
        :param PropertyPackage property_package: The thermodynamic package providing phase equilibrium physics.
        :param str description: Optional physical description of the stream.
        """
        self.property_package = property_package
        self.components = property_package.components
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        """
        Declares all macroscopic and phase-specific variables.
        Implements numerical vaccines against Structural Rank Deficiency (Bilinearity Trap)
        and limits to the KINSOL solver's Newton step size.
        """
        # 1. Base State Variables
        self.T = self.createVariable("T", "K", description="Equilibrium Temperature", value=350.0, exposure_type="algebraic")
        self.P = self.createVariable("P", "bar", description="Equilibrium Pressure", value=1.01325, exposure_type="algebraic")
        self.F_molar = self.createVariable("F_molar", "mol/s", description="Total Molar Flow Rate", value=100.0, exposure_type="algebraic")
        
        # FIX: Realistic baseline for thermodynamic magnitudes
        self.H_molar = self.createVariable("H_molar", "J/mol", description="Overall Molar Enthalpy", value=25000.0, exposure_type="algebraic")

        # 2. Phase Partition Variable (Theta)
        self.V_frac = self.createVariable(
            "V_frac", "", description="Vapor Phase Molar Fraction",
            lower_bound=-0.05, upper_bound=1.05, value=0.5, exposure_type="algebraic"
        )

        # 3. Composition Vectors
        self.z = {}  # Overall fractions
        self.x = {}  # Liquid fractions
        self.y = {}  # Vapor fractions
        n_comp = len(self.components)

        for comp in self.components:
            self.z[comp] = self.createVariable(f"z_{comp}", "", value=1.0 / n_comp, exposure_type="algebraic")
            
            # =================================================================
            # BILINEARITY TRAP FIX: Native Numerical Asymmetry
            # =================================================================
            # Liquid and vapor fractions must start mathematically apart (0.1 and 0.9)
            # to prevent the Jacobian derivative (x - y) from evaluating to 0.0 at t=0.
            self.x[comp] = self.createVariable(f"x_{comp}", "", value=0.1, exposure_type="algebraic")
            self.y[comp] = self.createVariable(f"y_{comp}", "", value=0.9, exposure_type="algebraic")

    def DeclareEquations(self):
        """
        Declares the rigorous Rachford-Rice constraints, phase-specific mass allocations,
        and injects the thermodynamic equilibrium physics from the PropertyPackage.
        """
        # 1. Overall Mass Closure
        sum_z = sum(self.z[comp]() for comp in self.components)
        self.createEquation("Overall_Fractions_Sum", description="Overall fractions unity sum", expr=sum_z - 1.0)

        # 2. Rachford-Rice Isothermal Objective
        sum_y = sum(self.y[comp]() for comp in self.components)
        sum_x = sum(self.x[comp]() for comp in self.components)
        self.createEquation("Rachford_Rice_Objective", description="Phase fractions strict closure", expr=sum_y - sum_x)

        for comp in self.components:
            z_c = self.z[comp]()
            x_c = self.x[comp]()
            y_c = self.y[comp]()
            v_f = self.V_frac()

            # 3. Component Mass Balance
            mass_bal_expr = z_c - (v_f * y_c + (1.0 - v_f) * x_c)
            self.createEquation(f"Mass_Bal_{comp}", description=f"Phase allocation for {comp}", expr=mass_bal_expr)

        # 4. Dependency Injection: Equilibrium physics from Thermo Package
        self.property_package.build_phase_equilibrium(self)

        # 5. Global Energy/Enthalpy Closure
        x_sym_dict = {comp: self.x[comp]() for comp in self.components}
        y_sym_dict = {comp: self.y[comp]() for comp in self.components}

        H_liq_expr = self.property_package.get_phase_enthalpy_expression(self.T(), self.P(), x_sym_dict, phase="liquid")
        H_vap_expr = self.property_package.get_phase_enthalpy_expression(self.T(), self.P(), y_sym_dict, phase="vapor")

        global_h_expr = self.H_molar() - (self.V_frac() * H_vap_expr + (1.0 - self.V_frac()) * H_liq_expr)
        self.createEquation("Global_Enthalpy_Closure", description="Two-phase enthalpy mix", expr=global_h_expr)


class EnergyStream(Model):
    """
    Energy Stream defining generic thermal or mechanical transfers.
    
    Used to topologically link heat duties (Q) or mechanical work (W) 
    between unit operations or utilities across the flowsheet.
    """
    
    def __init__(self, name, stream_type="heat", description=""):
        """
        Initializes the EnergyStream.

        :param str name: Unique identifier for the stream instance.
        :param str stream_type: The nature of energy transfer ('heat' or 'work').
        :param str description: Optional physical description.
        """
        self.stream_type = str(stream_type).lower()
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        """
        Declares the algebraic variables for the energy transfer.
        Injects a colossal baseline for Q to dilate the KINSOL solver's maximum step length.
        """
        if self.stream_type == "heat":
            # FIX: Colossal Q baseline to force the dilation of KINSOL's mxnewtstep
            self.Q = self.createVariable("Q", "W", value=1e6, exposure_type="algebraic")
            self.T_source = self.createVariable("T_source", "K", value=298.15, exposure_type="algebraic")
        elif self.stream_type == "work":
            self.W = self.createVariable("W", "W", value=0.0, exposure_type="algebraic")
            self.RPM = self.createVariable("RPM", "rev/min", value=0.0, exposure_type="algebraic")

    def DeclareEquations(self):
        """
        Energy streams generally act as free variables defined by connection 
        topology, hence no intrinsic residual equations are strictly required.
        """
        pass