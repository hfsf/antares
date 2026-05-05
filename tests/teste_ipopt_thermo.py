# -*- coding: utf-8 -*-

"""
Teste de Alta Robustez do Framework ANTARES (Integração IPOPT):
Resolução Direta de Flash Termodinâmico com Peng-Robinson EOS.

Demonstra o poder do novo motor NLP (IPOPT) em resolver um sistema
altamente não-linear SEM necessidade de "Warm-Start" ou chutes iniciais
ajustados à mão. O IPOPT navega pelo envelope de fases ancorado
rigorosamente pelas barreiras termodinâmicas injetadas na biblioteca (0 <= x <= 1).
"""

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.model import Model
from antares.library.streams import TwoPhaseStream
from antares.library.thermo_package import PengRobinsonEOS

# =============================================================================
# CONFIGURAÇÃO GLOBAL: ATIVAÇÃO DO MOTOR NLP (IPOPT)
# =============================================================================
cfg.VERBOSITY_LEVEL = 1
cfg.DEFAULT_STEADY_SOLVER = "ipopt"
cfg.USE_BOUNDS_IN_STEADY_STATE = True
cfg.ROOTFINDER_SOLVER_DEBUG_LEVEL = 1  # Para vermos o relatório glorioso do IPOPT

# =============================================================================
# DECLARAÇÃO DO PROBLEMA (Flash de Hidrocarbonetos)
# =============================================================================
class FlashSemMedo(Model):
    def __init__(self, name):
        # Uma mistura leve, muito sensível à pressão de bolha
        self.pkg = PengRobinsonEOS(components=["methane", "ethane", "propane"])
        self.pkg.fetch_parameters_from_db()
        super().__init__(name, description="Flash resolvido a sangue frio pelo IPOPT")
        self()

    def DeclareVariables(self):
        # A própria corrente especial TwoPhaseStream já monta o equilíbrio
        self.vle_core = TwoPhaseStream("vle_core", self.pkg)
        self.submodels.append(self.vle_core)

    def DeclareEquations(self):
        self.pkg.build_phase_equilibrium(self.vle_core)

        # Alterado para 230 K e 20 bar para garantir a zona bifásica
        self.createEquation("fix_T", expr=self.vle_core.T() - 230.0)    
        self.createEquation("fix_P", expr=self.vle_core.P() - 20.0)     
        self.createEquation("fix_F", expr=self.vle_core.F_molar() - 100.0)
        
        self.createEquation("fix_z_c1", expr=self.vle_core.z["methane"]() - 0.40)
        self.createEquation("fix_z_c2", expr=self.vle_core.z["ethane"]() - 0.35)

# =============================================================================
# EXECUÇÃO DO TESTE
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" ANTARES V5 - ROBUST THERMODYNAMICS VIA IPOPT NLP")
    print("=" * 70)

    # 1. Instanciamos o modelo
    modelo = FlashSemMedo("Flash_Bruto")
    
    # Repare: ZERO CHUTES INICIAIS fornecidos. 
    # O KINSOL desmaiaria a tentar adivinhar as frações a partir do 0.
    simulador = Simulator(model=modelo)
    
    print("\n -> Invocando o Solucionador de Programação Não-Linear (IPOPT)...")
    try:
        # A compilação em C garante que as derivadas do CasADi voem para o IPOPT
        resultados = simulador.run_steady_state(use_c_code=True)
        
        # Extração da Fração Vaporizada calculada no escuro
        vf_calc = resultados.get_variable(modelo.vle_core.V_frac)[-1]
        
        print("\n" + "=" * 70)
        print(" SUCESSO ABSOLUTO!")
        print(f" O IPOPT convergiu a mistura! Fração Vaporizada Final: {vf_calc:.4f}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[FALHA] O Solucionador encontrou um erro: {e}")