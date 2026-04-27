# -*- coding: utf-8 -*-

"""
Coupled Plant Benchmark - ANTARES V5
Simulation of 4 Unit Operations (Mixer, 2x CSTR, Splitter) with a Recycle Loop.
Validates the Equation-Oriented (EO) resolution against the Analytical Solution.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model

# Framework Settings
cfg.VERBOSITY_LEVEL = 1
cfg.USE_C_CODE_COMPILATION = False  # Runs instantly without JIT (0D)

# =============================================================================
# 1. UNIT OPERATIONS LIBRARY (THE FRONTEND)
# =============================================================================

class Mixer(Model):
    """
    Ideal Fluid Mixer.
    Combines two inlet flows into a single outlet stream, assuming perfect
    and instantaneous mixing.
    """

    def __init__(self, name):
        """
        :param str name: Unique identifier for the mixer.
        """
        super().__init__(name, description="Ideal Mixer")
        self()

    def DeclareVariables(self):
        self.F_in1 = self.createVariable("F_in1", "L/min", exposure_type="algebraic")
        self.C_in1 = self.createVariable("C_in1", "mol/L", exposure_type="algebraic")
        self.F_in2 = self.createVariable("F_in2", "L/min", exposure_type="algebraic")
        self.C_in2 = self.createVariable("C_in2", "mol/L", exposure_type="algebraic")

        self.F_out = self.createVariable("F_out", "L/min", exposure_type="algebraic")
        self.C_out = self.createVariable("C_out", "mol/L", exposure_type="algebraic")

    def DeclareEquations(self):
        # Global Mass Balance
        eq_flow = self.F_out() - (self.F_in1() + self.F_in2())
        self.createEquation("flow_balance", expr=eq_flow)

        # Component Mass Balance
        eq_species = (self.F_out() * self.C_out()) - (
            self.F_in1() * self.C_in1() + self.F_in2() * self.C_in2()
        )
        self.createEquation("species_balance", expr=eq_species)


class CSTR(Model):
    """
    Continuous Stirred-Tank Reactor.
    Models a perfectly mixed tank with a first-order kinetic reaction.
    """

    def __init__(self, name):
        """
        :param str name: Unique identifier for the reactor.
        """
        super().__init__(name, description="Continuous Stirred-Tank Reactor")
        self()

    def DeclareVariables(self):
        self.F_in = self.createVariable("F_in", "L/min", exposure_type="algebraic")
        self.C_in = self.createVariable("C_in", "mol/L", exposure_type="algebraic")
        self.F_out = self.createVariable("F_out", "L/min", exposure_type="algebraic")

        # Dynamic State Variable
        self.C_out = self.createVariable("C_out", "mol/L", exposure_type="differential")
        self.setInitialCondition(self.C_out, 0.0)  # Starts empty (solvent only)

    def DeclareParameters(self):
        self.V = self.createParameter("V", "L", description="Reactor Volume", value=500.0)
        self.k = self.createParameter("k", "min^-1", description="Kinetic Constant", value=0.1)

    def DeclareEquations(self):
        # Assuming constant density (F_in = F_out)
        self.createEquation("flow_balance", expr=self.F_out() - self.F_in())

        # V * dC/dt = F_in*C_in - F_out*C_out - k*V*C_out
        eq_kinetic = (self.V() * self.C_out.Diff()) - (
            self.F_in() * self.C_in()
            - self.F_out() * self.C_out()
            - self.k() * self.V() * self.C_out()
        )
        self.createEquation("reaction_balance", expr=eq_kinetic)


class Splitter(Model):
    """
    Stream Splitter (Tee).
    Divides an incoming stream into two outgoing streams with identical compositions.
    """

    def __init__(self, name):
        """
        :param str name: Unique identifier for the splitter.
        """
        super().__init__(name, description="Stream Splitter (Tee)")
        self()

    def DeclareVariables(self):
        self.F_in = self.createVariable("F_in", "L/min", exposure_type="algebraic")
        self.C_in = self.createVariable("C_in", "mol/L", exposure_type="algebraic")

        self.F_out1 = self.createVariable("F_out1", "L/min", exposure_type="algebraic")
        self.C_out1 = self.createVariable("C_out1", "mol/L", exposure_type="algebraic")

        self.F_out2 = self.createVariable("F_out2", "L/min", exposure_type="algebraic")
        self.C_out2 = self.createVariable("C_out2", "mol/L", exposure_type="algebraic")

    def DeclareParameters(self):
        self.R = self.createParameter("R", "", description="Recycle Fraction", value=0.5)

    def DeclareEquations(self):
        self.createEquation("split_flow_1", expr=self.F_out1() - (self.R() * self.F_in()))
        self.createEquation("split_flow_2", expr=self.F_out2() - ((1.0 - self.R()) * self.F_in()))
        
        # Compositions remain unchanged
        self.createEquation("split_conc_1", expr=self.C_out1() - self.C_in())
        self.createEquation("split_conc_2", expr=self.C_out2() - self.C_in())


# =============================================================================
# 2. GLOBAL TOPOLOGY (THE MASTER FLOWSHEET)
# =============================================================================

class PlantBenchmark(Model):
    """
    Master Flowsheet assembling the interconnected unit operations.
    """

    def __init__(self, name):
        """
        :param str name: Unique identifier for the master flowsheet.
        """
        self.mix = Mixer("M1")
        self.R1 = CSTR("R1")
        self.R2 = CSTR("R2")
        self.spl = Splitter("S1")
        super().__init__(name, submodels=[self.mix, self.R1, self.R2, self.spl])
        self()

    def DeclareEquations(self):
        # 1. Fresh Feed Specification
        # ENCAPSULATION MAGIC: Using `.fix()` completely hides the residual 
        # equation logic (DOF closure) from the end-user.
        self.mix.F_in1.fix(100.0)
        self.mix.C_in1.fix(1.0)

        # 2. Topological Connections using V5 Connection Class
        # Link 1: Mixer -> R1
        Connection("L1_F", self.mix, self.R1, "F_out", "F_in").apply_to(self)
        Connection("L1_C", self.mix, self.R1, "C_out", "C_in").apply_to(self)

        # Link 2: R1 -> R2
        Connection("L2_F", self.R1, self.R2, "F_out", "F_in").apply_to(self)
        Connection("L2_C", self.R1, self.R2, "C_out", "C_in").apply_to(self)

        # Link 3: R2 -> Splitter
        Connection("L3_F", self.R2, self.spl, "F_out", "F_in").apply_to(self)
        Connection("L3_C", self.R2, self.spl, "C_out", "C_in").apply_to(self)

        # Link 4: THE GREAT CHALLENGE - Recycle Loop: Splitter -> Mixer
        Connection("L_Recycle_F", self.spl, self.mix, "F_out1", "F_in2").apply_to(self)
        Connection("L_Recycle_C", self.spl, self.mix, "C_out1", "C_in2").apply_to(self)


# =============================================================================
# 3. EXECUTION AND AUDIT
# =============================================================================

if __name__ == "__main__":
    plant = PlantBenchmark("CSTR_Complex")

    print("\n[INIT] Firing up the DAE Integration Engine (IDAS)...")
    simulator = Simulator(model=plant)

    # Simulate 150 minutes (more than enough to reach Steady-State)
    t_span = np.linspace(0, 150, 300)

    # -------------------------------------------------------------------------
    # SINGULAR JACOBIAN PREVENTION (Zero-Flow Bilinearity Trap)
    # We provide non-zero initial guesses for the algebraic variables using the
    # native `.setValue()` method. This guarantees that the derivatives of
    # bilinear terms (Flow * Concentration) do not collapse to zero during
    # the Algebraic Initial Conditions (calc_ic) stage at t=0.
    # -------------------------------------------------------------------------
    plant.mix.F_out.setValue(100.0)
    plant.R1.F_out.setValue(100.0)
    plant.R2.F_out.setValue(100.0)
    plant.spl.F_out1.setValue(50.0)
    plant.spl.F_out2.setValue(50.0)
    plant.mix.C_out.setValue(1.0)
    plant.R1.F_in.setValue(100.0)
    plant.R2.F_in.setValue(100.0)

    results = simulator.run(t_span)

    # -------------------------------------------------------------------------
    # 4. LITERATURE COMPARISON
    # -------------------------------------------------------------------------

    # Extract the final row of the results matrix (t = 150 min, Steady-State)
    C1_sim = results.get_variable(plant.R1.C_out)[-1]
    C2_sim = results.get_variable(plant.R2.C_out)[-1]

    # Exact Analytical Results (calculated from the global mass balance)
    C1_lit = 0.50 / 0.85
    C2_lit = 0.40 / 0.85

    error_C1 = abs(C1_sim - C1_lit) / C1_lit * 100.0
    error_C2 = abs(C2_sim - C2_lit) / C2_lit * 100.0

    print("\n" + "=" * 60)
    print("Simultaneous Recycle Solution (Tearing-Free / Equation-Oriented)")
    print("-" * 60)
    print(
        f"[{plant.R1.name} - C_out] | Analytical: {C1_lit:.6f} | ANTARES: {C1_sim:.6f} | Relative Error: {error_C1:.2e} %"
    )
    print(
        f"[{plant.R2.name} - C_out] | Analytical: {C2_lit:.6f} | ANTARES: {C2_sim:.6f} | Relative Error: {error_C2:.2e} %"
    )
    print("=" * 60)