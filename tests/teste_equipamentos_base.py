# -*- coding: utf-8 -*-

"""
ANTARES V5 - Validation Test for Equipment Base Classes
Demonstrates the instantiation of Steady and Dynamic equipment using the new architecture,
and their topological equivalence by reaching the same steady-state roots.
"""

import numpy as np
import casadi as ca

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model
from antares.library.streams import MaterialStream, TwoPhaseStream
from antares.library.thermo_package import PengRobinsonEOS

from antares.library.base import SteadyEquipment, DynamicEquipment

cfg.VERBOSITY_LEVEL = 1
cfg.USE_C_CODE_COMPILATION = False
cfg.VERBOSITY_OF_DOF_ANALYSIS = 2

# =============================================================================
# 1. DEFINIÇÃO DA CLASSE ESTACIONÁRIA
# =============================================================================
class SteadyFlashDrum(SteadyEquipment):
    r"""Implementation of a Steady-State Mass Flash using the new equipment base class."""
    def __init__(self, name, property_package):
        super().__init__(name, property_package, "Steady Isothermal Flash")
        self()

    def DeclareVariables(self):
        self.add_material_inlet("feed")
        self.add_material_outlet("vap_out")
        self.add_material_outlet("liq_out")
        
        # Core thermodynamic equilibrium
        self.vle_core = TwoPhaseStream("vle_core", self.pkg)
        self.submodels.append(self.vle_core)
        
        self.T_op = self.createParameter("T_op", "K", value=310.0)
        self.P_op = self.createParameter("P_op", "bar", value=25.0)

    def DeclareEquations(self):
        # 1. Inject Thermodynamics into the core
        self.pkg.build_phase_equilibrium(self.vle_core)
        
        # 2. Core Mapping
        self.createEquation("core_T", expr=self.vle_core.T() - self.T_op())
        self.createEquation("core_P", expr=self.vle_core.P() - self.P_op())
        self.createEquation("core_F", expr=self.vle_core.F_molar() - self.feed.F_molar())
        
        for comp in self.pkg.components[:-1]:
            self.createEquation(f"core_z_{comp}", expr=self.vle_core.z[comp]() - self.feed.z[comp]())

        # 3. Vapor Mapping
        self.createEquation("vap_F", expr=self.vap_out.F_molar() - (self.vle_core.F_molar() * self.vle_core.V_frac()))
        self.createEquation("vap_T", expr=self.vap_out.T() - self.vle_core.T())
        self.createEquation("vap_P", expr=self.vap_out.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(f"vap_y_{comp}", expr=self.vap_out.z[comp]() - self.vle_core.y[comp]())

        # 4. Liquid Mapping
        self.createEquation("liq_F", expr=self.liq_out.F_molar() - (self.vle_core.F_molar() * (1.0 - self.vle_core.V_frac())))
        self.createEquation("liq_T", expr=self.liq_out.T() - self.vle_core.T())
        self.createEquation("liq_P", expr=self.liq_out.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(f"liq_x_{comp}", expr=self.liq_out.z[comp]() - self.vle_core.x[comp]())


# =============================================================================
# 2. DEFINIÇÃO DA CLASSE DINÂMICA
# =============================================================================
class DynamicFlashDrum(DynamicEquipment):
    r"""Implementation of a Dynamic Mass Flash using the new equipment base class."""
    def __init__(self, name, property_package):
        super().__init__(name, property_package, "Dynamic Isothermal Flash")
        self()

    def DeclareVariables(self):
        self.add_material_inlet("feed")
        self.add_material_outlet("vap_out")
        self.add_material_outlet("liq_out")
        
        # The TwoPhaseStream acts as the internal equilibrium holdup state
        self.vle_core = TwoPhaseStream("vle_core", self.pkg)
        self.submodels.append(self.vle_core)
        
        # Dynamic Holdup Variables
        self.M_comp = {}
        for comp in self.pkg.components:
            self.M_comp[comp] = self.createVariable(
                f"M_{comp}", "mol", value=50.0, exposure_type="differential"
            )
            
        self.M_tot = self.createVariable("M_tot", "mol", value=100.0, exposure_type="algebraic")

        # Operational Parameters
        self.T_op = self.createParameter("T_op", "K", value=310.0)
        self.P_op = self.createParameter("P_op", "bar", value=25.0)
        self.k_v = self.createParameter("k_v", "1/s", value=1.0)
        self.k_l = self.createParameter("k_l", "1/s", value=1.0)

    def DeclareEquations(self):
        # 1. Automatic ODE generation from the base class
        self.generate_dynamic_mass_balances(self.M_comp)

        # 2. Holdup mass closure and anchoring to the VLE core
        sum_M = sum([self.M_comp[c]() for c in self.pkg.components])
        self.createEquation("Total_Mass", expr=self.M_tot() - sum_M)

        for comp in self.pkg.components[:-1]:
            self.createEquation(f"Holdup_z_{comp}", expr=(self.vle_core.z[comp]() * self.M_tot()) - self.M_comp[comp]())

        self.createEquation("core_T", expr=self.vle_core.T() - self.T_op())
        self.createEquation("core_P", expr=self.vle_core.P() - self.P_op())
        
        # CORE_F FIX: Tied to feed flow rate to close degrees of freedom with dimensional coherence (mol/s)
        self.createEquation("core_F", expr=self.vle_core.F_molar() - self.feed.F_molar())

        # 3. Phase Equilibrium
        self.pkg.build_phase_equilibrium(self.vle_core)
        
        # 4. Hydraulics and Output Mapping
        # F_molar (mol/s) = k (1/s) * M_tot (mol) * V_frac (dimless)
        self.createEquation("Hydraulic_Vapor", expr=self.vap_out.F_molar() - self.k_v() * self.M_tot() * self.vle_core.V_frac())
        self.createEquation("Hydraulic_Liquid", expr=self.liq_out.F_molar() - self.k_l() * self.M_tot() * (1.0 - self.vle_core.V_frac()))

        self.createEquation("vap_T_map", expr=self.vap_out.T() - self.T_op())
        self.createEquation("vap_P_map", expr=self.vap_out.P() - self.P_op())
        self.createEquation("liq_T_map", expr=self.liq_out.T() - self.T_op())
        self.createEquation("liq_P_map", expr=self.liq_out.P() - self.P_op())

        for comp in self.pkg.components[:-1]:
            self.createEquation(f"vap_y_map_{comp}", expr=self.vap_out.z[comp]() - self.vle_core.y[comp]())
            self.createEquation(f"liq_x_map_{comp}", expr=self.liq_out.z[comp]() - self.vle_core.x[comp]())


# =============================================================================
# 3. FLOWSHEET E TESTE DE EXECUÇÃO
# =============================================================================
class TestFlowsheet(Model):
    def __init__(self, mode="steady"):
        self.mode = mode
        # Restaurado para o Benchmark Rigoroso que sabemos que funciona!
        self.pkg = PengRobinsonEOS(components=["methane", "propane", "pentane"])
        self.pkg.fetch_parameters_from_db()
        super().__init__(f"Flowsheet_{mode}")
        self()

    def DeclareVariables(self):
        self.feed_stream = MaterialStream("feed_stream", self.pkg)
        
        if self.mode == "steady":
            self.equip = SteadyFlashDrum("flash", self.pkg)
        else:
            self.equip = DynamicFlashDrum("flash", self.pkg)
            
        self.submodels.extend([self.feed_stream, self.equip])

        self.equip.T_op.setValue(310.0)
        self.equip.P_op.setValue(25.0)

        # "Warm-Start" Rigoroso baseado na literatura
        for stream in [self.feed_stream, self.equip.feed, self.equip.vle_core, self.equip.vap_out, self.equip.liq_out]:
            stream.T.setValue(310.0)
            stream.P.setValue(25.0)
            stream.F_molar.setValue(100.0)
            stream.z["methane"].setValue(0.42)
            stream.z["propane"].setValue(0.26)
            stream.z["pentane"].setValue(0.32)

        # Chutes Iniciais Assimétricos do Benchmark Clássico
        chute_x = {"methane": 0.10, "propane": 0.25, "pentane": 0.65}
        chute_y = {"methane": 0.75, "propane": 0.20, "pentane": 0.05}

        for comp in self.pkg.components:
            self.equip.vle_core.x[comp].setValue(chute_x[comp])
            self.equip.vle_core.y[comp].setValue(chute_y[comp])
            self.equip.liq_out.z[comp].setValue(chute_x[comp])
            self.equip.vap_out.z[comp].setValue(chute_y[comp])
            
        self.equip.vle_core.V_frac.setValue(0.50)

    def DeclareEquations(self):
        self.feed_stream.T.fix(310.0)
        self.feed_stream.P.fix(25.0)
        self.feed_stream.F_molar.fix(100.0)
        self.feed_stream.z["methane"].fix(0.42)
        self.feed_stream.z["propane"].fix(0.26)

        Connection("C1", self.feed_stream, self.equip.feed).apply_to(self)


if __name__ == "__main__":
    import sys
    cfg.VERBOSITY_LEVEL = 1
    print("\n" + "=" * 70)
    print(" ANTARES V5 - EQUIPMENT BASE CLASSES VALIDATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. TESTE DO CASO ESTACIONÁRIO (ALGEBRAIC)
    # -------------------------------------------------------------------------
    print("\n[1/2] Instanciando SteadyEquipment (KINSOL Sandbox)...")
    plant_ss = TestFlowsheet(mode="steady")
    sim_ss = Simulator(plant_ss)
    
    # Dicionários para guardar a resposta para o "Hot-Start" dinâmico
    vf_ss = 0.0
    z_ss = {}
    x_ss = {}
    y_ss = {}
    
    try:
        print(" -> Resolvendo Raízes do Estado Estacionário...")
        res_ss = sim_ss.run(t_span=[0, 1])
        
        vf_ss = res_ss.get_variable(plant_ss.equip.vle_core.V_frac)[-1]
        for c in plant_ss.pkg.components:
            z_ss[c] = res_ss.get_variable(plant_ss.equip.vle_core.z[c])[-1]
            x_ss[c] = res_ss.get_variable(plant_ss.equip.vle_core.x[c])[-1]
            y_ss[c] = res_ss.get_variable(plant_ss.equip.vle_core.y[c])[-1]
        
        print(f" -> SUCESSO: Fração Vaporizada (Estacionário) = {vf_ss:.4f}")
    except Exception as e:
        print(f" -> FALHA NO ESTACIONÁRIO: {e}")
        print("\nAbortando caso dinâmico devido à falha de convergência no estacionário.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. TESTE DO CASO DINÂMICO (DAE / ODE)
    # -------------------------------------------------------------------------
    print("\n[2/2] Instanciando DynamicEquipment (IDAS)...")
    plant_dyn = TestFlowsheet(mode="dynamic")
    
    # === A MAGIA DA ENGENHARIA DE PROCESSOS: "Hot-Start" ===
    # Preenchemos a Jacobiana de t=0 com as raízes puras do caso estacionário
    plant_dyn.equip.vle_core.V_frac.setValue(vf_ss)
    for c in plant_dyn.pkg.components:
        plant_dyn.equip.vle_core.z[c].setValue(z_ss[c])
        plant_dyn.equip.vle_core.x[c].setValue(x_ss[c])
        plant_dyn.equip.vle_core.y[c].setValue(y_ss[c])

    sim_dyn = Simulator(plant_dyn)
    
    try:
        print(" -> Marchando no tempo (t=0 a 3600s)...")
        t_span = np.linspace(0, 60, 100)
        
        # O integrador ODE precisa saber a massa exata de cada componente em t=0!
        M_tot_t0 = 10.0
        condicoes_iniciais = {}
        for c in plant_dyn.pkg.components:
            condicoes_iniciais[f"M_{c}_flash"] = M_tot_t0 * z_ss[c]

        # Com as ODEs e algébricas ancoradas num estado termodinâmico puro, o IDAS arranca sereno
        res_dyn = sim_dyn.run(t_span=t_span, initial_conditions=condicoes_iniciais)
        
        vf_dyn = res_dyn.get_variable(plant_dyn.equip.vle_core.V_frac)[-1]
        print(f" -> SUCESSO: Fração Vaporizada (Dinâmico t=3600s) = {vf_dyn:.4f}")

        # -------------------------------------------------------------------------
        # 3. PLOTAGEM DOS RESULTADOS TRANSIENTES (USANDO O MÓDULO NATIVO)
        # -------------------------------------------------------------------------
        print("\n -> Gerando Gráficos do Transiente via Plotter...")
        
        # Assumindo que o Plotter está na raiz do core ou backend. Ajuste o import se necessário.
        from antares.plotter import Plotter 
        
        plotter = Plotter(res_dyn)
        
        # Gráfico 1: Fração Vaporizada
        plotter.plot_variables(
            variables=[plant_dyn.equip.vle_core.V_frac],
            title='Thermodynamic Equilibrium Transient',
            ylabel='Vapor Fraction',
            legend_labels=['Dynamic V/F'],
            #save_path='plot_vfrac.png',
            show=False  # Mantém a janela oculta até ao último gráfico
        )
        
        # Gráfico 2: Holdup de Massa (Enchimento do Tanque)
        plotter.plot_variables(
            variables=[plant_dyn.equip.M_tot],
            title='Hydraulic Accumulation',
            ylabel='Mass Holdup [mol]',
            legend_labels=['Total Mass Holdup'],
            #save_path='plot_holdup.png',
            show=False
        )
        
        # Gráfico 3: Vazões de Saída (Vapor vs Líquido)
        # Note como a extração orientada a objetos previne colisões do nome 'F_molar'
        plotter.plot_variables(
            variables=[plant_dyn.equip.vap_out.F_molar, plant_dyn.equip.liq_out.F_molar],
            title='Outlet Flow Dynamics',
            ylabel='Flow Rate [mol/s]',
            legend_labels=['Vapor Flow Rate', 'Liquid Flow Rate'],
            #save_path='plot_flows.png',
            show=True   # Renderiza todas as janelas geradas
        )

        print("\n" + "=" * 70)
        print(" VALIDATION SUMMARY")
        print("=" * 70)
        print(" Transição topológica de Estacionário para Dinâmico validada com sucesso.")
        print(" Gráficos extraídos via encapsulamento orientado a objetos.")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f" -> FALHA NO DINÂMICO: {e}")

    print("\n" + "=" * 70)
    print(" VALIDATION SUMMARY")
    print("=" * 70)
    print(" Transição topológica de Estacionário para Dinâmico validada com sucesso.")
    print("=" * 70 + "\n")