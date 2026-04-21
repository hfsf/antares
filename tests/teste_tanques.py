# -*- coding: utf-8 -*-

"""
Teste End-to-End do Framework ANTARES:
Simulação Dinâmica de Dois Tanques em Série.
Padrão rigoroso de Orientação a Objetos e Encapsulamento.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model
from antares.core.template_units import m, s
from antares.plotter import Plotter

# Criando unidades compostas
m2 = m**2
m3 = m**3

# =============================================================================
# FASE 1: O FRONTEND (A Biblioteca de Equipamentos)
# =============================================================================

class Tanque(Model):
    def __init__(self, name):
        super().__init__(name, description="Tanque de armazenamento com dreno")
        self()

    def DeclareVariables(self):
        self.h = self.createVariable("h", m, "Nível do tanque", exposure_type="differential")
        self.F_in = self.createVariable("F_in", m3 / s, "Vazão de entrada", exposure_type="algebraic")
        self.F_out = self.createVariable("F_out", m3 / s, "Vazão de saída", exposure_type="algebraic")

    def DeclareParameters(self):
        self.A = self.createParameter("A", m2, "Área transversal", value=2.0)
        self.Cv = self.createParameter("Cv", "m ^ 2 / s", "Constante da válvula", value=0.5)

    def DeclareEquations(self):
        eq_massa = (self.A() * self.h.Diff()) - (self.F_in() - self.F_out())
        self.createEquation("balanco_massa", expr=eq_massa)

        eq_valvula = self.F_out() - (self.Cv() * self.h())
        self.createEquation("vazao_saida", expr=eq_valvula)

# =============================================================================
# FASE 2: A TOPOLOGIA (Master Flowsheet)
# =============================================================================

class PlantaPiloto(Model):
    def __init__(self, name):
        self.T1 = Tanque("T1")
        self.T2 = Tanque("T2")
        super().__init__(name, description="Dois tanques em série", submodels=[self.T1, self.T2])
        self()

    def DeclareVariables(self):
        # ENCAPSULAMENTO PERFEITO: Acessamos as variáveis instanciadas diretamente
        self.setInitialCondition(self.T1.h, value=0.0)
        self.setInitialCondition(self.T2.h, value=5.0)

    def DeclareParameters(self):
        # Sobrescrevendo o parâmetro do submodelo sem usar strings!
        self.T2.A.value = 3.5

    def DeclareEquations(self):
        eq_alimentacao = self.T1.F_in() - 1.5
        self.createEquation("eq_alim_global", expr=eq_alimentacao)

        link = Connection(
            name="Linha_Transferencia",
            source_port=self.T1,
            sink_port=self.T2,
            source_var_name="F_out",
            sink_var_name="F_in",
        )
        link.apply_to(self)

# =============================================================================
# FASE 3: EXECUÇÃO (Backend CasADi + Pós-Processamento)
# =============================================================================

if __name__ == "__main__":
    cfg.VERBOSITY_LEVEL = 1

    processo = PlantaPiloto("SistemaGlobal")
    
    # Execução limpa: O simulador puxa as CIs automaticamente do Master Flowsheet
    simulador = Simulator(model=processo)
    resultados = simulador.run(t_span=np.linspace(0, 30, 200))

    plotador = Plotter(resultados)
    
    # Extração elegante de nomes para o plot sem "advinhar" strings
    var_T1 = processo.T1.h.name
    var_T2 = processo.T2.h.name

    plotador.plot(
        variables=[var_T1, var_T2],
        title="Dinâmica dos Tanques em Série",
        xlabel="Tempo Decorrido (s)",
        ylabel="Altura do Fluido (m)",
        legend_labels={var_T1: "Tanque de Alimentação", var_T2: "Tanque de Retenção"},
        show=True
    )