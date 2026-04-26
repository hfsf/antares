# -*- coding: utf-8 -*-

"""
Master Flowsheet Benchmark - ANTARES V5
Demonstrates the integration of Topological Streams, Hybrid Thermodynamics
(LUT and EOS), 1D+t PDE Discretization, and Perfect Encapsulation in EO simulation.

Process Description:
An exothermic reaction occurs in a Tubular Reactor.
The feed is modeled using a Multicomponent Equation of State (EOS) fetching
real data from the `thermo` database.
The reactor dynamically creates its own thermal port (EnergyStream) and
material port (MaterialStream), completely hiding the boundary conditions
and integrals from the flowsheet user.
"""

import casadi as ca
import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.expression_evaluation import EquationNode
from antares.core.model import Model
from antares.core.stream import EnergyStream, MaterialStream
from antares.core.thermo_package import MulticomponentEOS, PureFluidLUT
from antares.core.unit import Unit
from antares.plotter import Plotter

# Framework Settings
cfg.VERBOSITY_LEVEL = 1
cfg.USE_C_CODE_COMPILATION = False


class TubularReactor(Model):
    """
    1D Distributed Tubular Reactor Model (Plug Flow with Dispersion).
    Demonstrates perfect encapsulation by generating its own ports
    (Material and Energy) to communicate with the external flowsheet.
    """

    def __init__(self, name, property_package, length=2.0, n_points=50, description=""):
        # O reator agora exige o pacote termodinâmico na sua construção
        self.property_package = property_package
        self.L = length
        self.N = n_points
        super().__init__(name, description)
        self()

    def DeclareVariables(self):
        # 1. Spatial Domain
        self.z = self.createDomain(
            "z", unit="m", length=self.L, n_points=self.N, diff_scheme="backward"
        )

        # 2. Distributed States
        self.C_A = self.createVariable(
            "C_A",
            "mol/m^3",
            description="Reactant Concentration",
            exposure_type="differential",
        )
        self.distributeVariable(self.C_A, self.z)

        self.T = self.createVariable(
            "T", "K", description="Reactor Temperature", exposure_type="differential"
        )
        self.distributeVariable(self.T, self.z)

        # =====================================================================
        # ENCAPSULATION MAGIC: The Reactor creates its own topological ports!
        # =====================================================================
        self.heat_port = EnergyStream("Jacket_Port", stream_type="heat")

        # A porta de entrada agora utiliza o pacote termodinâmico recebido!
        self.inlet = MaterialStream(
            "Inlet_Port", property_package=self.property_package
        )

        # Register the ports as submodels of the reactor
        self.submodels.extend([self.heat_port, self.inlet])

    def DeclareParameters(self):
        self.v = self.createParameter("v", "m/s", value=0.5)
        self.Deff = self.createParameter("Deff", "m^2/s", value=0.01)
        self.k0 = self.createParameter("k0", "s^-1", value=0.8)
        self.dH_rxn = self.createParameter("dH_rxn", "J/mol", value=-50000.0)
        self.rho_Cp = self.createParameter("rho_Cp", "J/(m^3*K)", value=4e6)
        self.U = self.createParameter("U", "W/(m^2*K)", value=1500.0)
        self.Diameter = self.createParameter("Diameter", "m", value=0.2)

    def DeclareEquations(self):
        # =====================================================================
        # 1. 1D+t PDEs (Mass and Energy Balances)
        # =====================================================================
        convection_mass = self.v() * self.z.apply_gradient(self.C_A)
        diffusion_mass = self.Deff() * self.z.apply_laplacian(self.C_A)
        reaction_rate = self.k0() * self.C_A()

        eq_mass = self.C_A.Diff() - (diffusion_mass - convection_mass - reaction_rate)
        self.addBulkEquation("Mass_Balance_A", eq_mass, self.z)

        convection_energy = self.v() * self.z.apply_gradient(self.T)
        heat_generation = (-self.dH_rxn()) * reaction_rate

        # The local driving force uses the Temperature from the heat_port
        heat_removal_local = (4.0 * self.U() / self.Diameter()) * (
            self.T() - self.heat_port.T_source()
        )

        eq_energy = (self.rho_Cp() * self.T.Diff()) - (
            -self.rho_Cp() * convection_energy + heat_generation - heat_removal_local
        )
        self.addBulkEquation("Energy_Balance", eq_energy, self.z)

        # =====================================================================
        # 2. INTERNAL HEAT INTEGRATION (Hidden from the end-user)
        # =====================================================================
        dz = self.L / (self.N - 1)

        T_sym = self.T.symbolic_object
        T_jack_sym = (
            self.heat_port.T_source.symbolic_object
            if hasattr(self.heat_port.T_source, "symbolic_object")
            else self.heat_port.T_source()
        )
        U_sym = (
            self.U.symbolic_object if hasattr(self.U, "symbolic_object") else self.U()
        )
        D_sym = (
            self.Diameter.symbolic_object
            if hasattr(self.Diameter, "symbolic_object")
            else self.Diameter()
        )

        area_per_node = np.pi * D_sym * dz

        # CasADi C++ Vectorized summation
        total_heat_sym = ca.sum1(U_sym * area_per_node * (T_sym - T_jack_sym))

        total_heat_expr = EquationNode(
            name="Q_total_sum",
            symbolic_object=total_heat_sym,
            unit_object=Unit("", "W"),
        )

        # The Reactor internally forces its heat_port to match the integrated Q
        self.createEquation(
            "Internal_Heat_Routing", expr=self.heat_port.Q() - total_heat_expr
        )

        # =====================================================================
        # 3. INTERNAL BOUNDARY CONDITIONS (Hidden from the end-user)
        # =====================================================================
        # The Reactor pulls the values directly from its own MaterialStream port!

        # 1. Área da secção transversal do tubo (m^2)
        A_cross = np.pi * (self.Diameter() ** 2) / 4.0

        # 2. Vazão volumétrica (m^3/s)
        q_vol = self.v() * A_cross

        # 3. Fluxo molar específico do etanol (mol/s)
        F_ethanol = self.inlet.F_molar() * self.inlet.z["ethanol"]()

        # 4. Concentração real na fronteira (mol/m^3)
        # O Guardião Dimensional validará automaticamente: (mol/s) / (m^3/s) = mol/m^3
        C_A_inlet = F_ethanol / q_vol

        self.setBoundaryCondition(
            self.C_A, self.z, "start", "dirichlet", value=C_A_inlet
        )
        self.setBoundaryCondition(
            self.T, self.z, "start", "dirichlet", value=self.inlet.T()
        )

        self.setBoundaryCondition(self.C_A, self.z, "end", "neumann", value=0.0)
        self.setBoundaryCondition(self.T, self.z, "end", "neumann", value=0.0)


class ProcessFlowsheet(Model):
    """
    Master Flowsheet Model.
    """

    def __init__(self, name):
        super().__init__(name, description="Complete Reactor Flowsheet")
        self()

    def DeclareVariables(self):
        # 1. PROPERTY PACKAGES
        self.pkg_reactants = MulticomponentEOS(components=["ethanol", "water"])
        self.pkg_reactants.fetch_parameters_from_db()

        self.pkg_utility = PureFluidLUT(fluid_name="water")
        self.pkg_utility.thermo_data = {
            "T_grid": [280.0, 300.0, 320.0, 340.0, 360.0],
            "P_grid": [1.0, 2.0, 3.0],
            "H_matrix": np.random.uniform(4000, 80000, (5, 3)),
        }

        # 2. TOPOLOGICAL STREAMS
        self.feed_stream = MaterialStream("Feed", property_package=self.pkg_reactants)
        self.utility_stream = EnergyStream("Cooling_Utility", stream_type="heat")

        # 3. UNIT OPERATIONS
        # Passamos o pacote termodinâmico para o Reator
        self.reactor = TubularReactor(
            "R1", property_package=self.pkg_reactants, length=5.0, n_points=50
        )

        # =====================================================================
        # INITIAL GUESSES (Fornece consistência ao vetor z0 do solver)
        # =====================================================================
        # Iteramos sobre as correntes e portas para garantir que o vetor
        # algébrico (z0) tenha todos os 20 elementos preenchidos.
        all_topological_units = [
            self.feed_stream,
            self.utility_stream,
            self.reactor.inlet,
            self.reactor.heat_port,
        ]

        for unit in all_topological_units:
            if hasattr(unit, "T"):
                unit.T.setValue(300.0)
            if hasattr(unit, "P"):
                unit.P.setValue(1.01325)
            if hasattr(unit, "F_molar"):
                unit.F_molar.setValue(100.0)
            if hasattr(unit, "Q"):
                unit.Q.setValue(0.0)
            if hasattr(unit, "T_source"):
                unit.T_source.setValue(300.0)

        # Register all topological entities to guarantee DOF closure!
        self.submodels.extend([self.feed_stream, self.utility_stream, self.reactor])

    def DeclareEquations(self):
        # =====================================================================
        # SPECIFICATIONS (Fixing Degrees of Freedom)
        # =====================================================================
        # 1. Feed Stream Specs (4 DOF for a binary mixture)
        self.createEquation("Spec_Feed_T", expr=self.feed_stream.T() - 350.0)
        self.createEquation("Spec_Feed_P", expr=self.feed_stream.P() - 1.01325)
        self.createEquation("Spec_Feed_F", expr=self.feed_stream.F_molar() - 100.0)
        self.createEquation(
            "Spec_Feed_z_ethanol", expr=self.feed_stream.z["ethanol"]() - 1.0
        )

        # 2. Utility Stream Spec (1 DOF)
        self.createEquation(
            "Spec_Utility_T", expr=self.utility_stream.T_source() - 300.0
        )

        # =====================================================================
        # TOPOLOGICAL ROUTING
        # =====================================================================
        # O fluxograma agora é extremamente limpo: ligar saídas às entradas
        Connection(
            "Connect_Utility", self.utility_stream, self.reactor.heat_port
        ).apply_to(self)
        Connection("Connect_Feed", self.feed_stream, self.reactor.inlet).apply_to(self)


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n[V5 ARCHITECTURE] Assembling Master Flowsheet...")
    flowsheet = ProcessFlowsheet("Master_Plant")

    simulador = Simulator(model=flowsheet)
    resultados = simulador.run(t_span=np.linspace(0, 50, 100))

    Q_total = resultados.get_variable(flowsheet.utility_stream.Q)[-1]
    H_feed = resultados.get_variable(flowsheet.feed_stream.H_molar)[-1]

    print("\n[RESULTS REPORT]")
    print(f" -> Analytical Feed Enthalpy (EOS with Real DB Data): {H_feed:.2f} J/mol")
    print(
        f" -> Total Heat transferred to Utility (EnergyStream): {Q_total / 1000:.2f} kW"
    )

    plotter = Plotter(resultados)

    # 1. Gráfico Transiente: PDE 1D+t revelada na Concentração!
    plotter.plot_spatial(
        variables=flowsheet.reactor.C_A,
        domain=flowsheet.reactor.z,
        time=[0.0, 1.0, 5.0, 15.0, 50.0],
        title="PDE 1D+t: Evolução Transiente da Concentração (C_A)",
    )

    # 2. Gráfico Transiente: PDE 1D+t revelada na Temperatura!
    plotter.plot_spatial(
        variables=flowsheet.reactor.T,
        domain=flowsheet.reactor.z,
        time=[0.0, 1.0, 5.0, 15.0, 50.0],
        title="PDE 1D+t: Evolução Transiente da Temperatura (T)",
    )

    # 3. O Gráfico do Estado Estacionário final
    plotter.plot_spatial(
        variables=[flowsheet.reactor.C_A, flowsheet.reactor.T],
        domain=flowsheet.reactor.z,
        time_index=-1,
        title="Steady-State Final do Reator Tubular",
    )
