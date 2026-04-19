# -*- coding: utf-8 -*-

"""
Teste End-to-End do Framework ANTARES:
Simulação Dinâmica de Dois Tanques em Série.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection

# Agora o Python encontra o 'antares' automaticamente em qualquer lugar do seu PC!
from antares.core.model import Model
from antares.core.template_units import m, s
from antares.plotter import Plotter

# Criando unidades compostas rapidamente para o teste
m2 = m**2
m3 = m**3

# =============================================================================
# FASE 1: O FRONTEND (A Biblioteca de Equipamentos)
# =============================================================================


class Tanque(Model):
    def __init__(self, name):
        super().__init__(name, description="Tanque de armazenamento com dreno")
        self()  # Aciona a rotina declarativa

    def DeclareVariables(self):
        # A EDO exigirá uma condição inicial estrita no run()
        self.h = self.createVariable(
            "h", m, "Nível do tanque", exposure_type="differential"
        )

        # Variáveis algébricas
        self.F_in = self.createVariable(
            "F_in", m3 / s, "Vazão de entrada", exposure_type="algebraic"
        )
        self.F_out = self.createVariable(
            "F_out", m3 / s, "Vazão de saída", exposure_type="algebraic"
        )

    def DeclareParameters(self):
        self.A = self.createParameter("A", m2, "Área transversal", value=2.0)
        self.Cv = self.createParameter(
            "Cv", "m ^ 2 / s", "Constante da válvula", value=0.5
        )

    def DeclareEquations(self):
        # Balanço de Massa: A * dh/dt = F_in - F_out
        # Repare na beleza do operador Diff()!
        eq_massa = (self.A() * self.h.Diff()) - (self.F_in() - self.F_out())
        self.createEquation("balanco_massa", expr=eq_massa)

        # Equação da Válvula: F_out = Cv * h
        eq_valvula = self.F_out() - (self.Cv() * self.h())
        self.createEquation("vazao_saida", expr=eq_valvula)


# =============================================================================
# FASE 2: A TOPOLOGIA (Master Flowsheet)
# =============================================================================


class PlantaPiloto(Model):
    def __init__(self, name):
        # Instancia os blocos do sistema
        self.T1 = Tanque("T1")
        self.T2 = Tanque("T2")

        # Passa os blocos para a classe mãe incorporar e "achatar" a hierarquia
        super().__init__(
            name, description="Dois tanques em série", submodels=[self.T1, self.T2]
        )
        self()

    def DeclareEquations(self):
        # Restrição de Contorno: Alimentação fixa do primeiro tanque (1.5 m3/s)
        eq_alimentacao = self.T1.F_in() - 1.5
        self.createEquation("eq_alim_global", expr=eq_alimentacao)

        # Restrição Topológica: Conecta a saída de T1 à entrada de T2
        link = Connection(
            name="Linha_Transferencia",
            source_port=self.T1,
            sink_port=self.T2,
            source_var_name="F_out",
            sink_var_name="F_in",
        )
        link.apply_to(self)  # Injeta as equações algébricas no flowsheet


# =============================================================================
# FASE 3: EXECUÇÃO (Backend CasADi + Pós-Processamento)
# =============================================================================

if __name__ == "__main__":
    # Configurações Globais (Vamos ligar os logs para ver a mágica a acontecer)
    cfg.VERBOSITY_LEVEL = 2

    print(">>> INICIANDO O FRAMEWORK ANTARES <<<\n")

    # 1. Compilação Simbólica
    processo = PlantaPiloto("SistemaGlobal")
    processo.print_dof_report()

    # 2. Inicialização do Simulador (Transpilação Analítica ocorre aqui)
    simulador = Simulator(model=processo)

    # 3. Definição do Setup da Simulação
    tempo = np.linspace(0, 30, 200)  # 30 segundos

    # A Salvaguarda: Se o utilizador não passar h_T1, o sistema lança erro!
    condicoes_iniciais = {
        "h_T1": 0.0,  # Tanque 1 começa vazio e vai encher
        "h_T2": 5.0,  # Tanque 2 começa cheio e vai drenar
    }

    # Sobrescrevendo a área do Tanque 2 on-the-fly (sem alterar a classe original)
    parametros = {"A_T2": 3.5}

    # 4. Resolução Numérica (C++ JIT Integration)
    print("\n>>> INICIANDO INTEGRAÇÃO NUMÉRICA <<<")
    resultados = simulador.run(
        t_span=tempo, initial_conditions=condicoes_iniciais, parameters_dict=parametros
    )

    # 5. Visualização Nativas com o novo Plotter
    print("\n>>> GERANDO GRÁFICOS <<<")
    plotador = Plotter(resultados)

    plotador.plot(
        variables=["h_T1", "h_T2"],
        title="Análise Dinâmica",
        xlabel="Tempo Decorrido (s)",  # Você decide o idioma aqui
        ylabel="Altura do Fluido (m)",
        legend_labels={"h_T1": "Tanque de Alimentação", "h_T2": "Tanque de Retenção"},
    )
