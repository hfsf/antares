# -*- coding: utf-8 -*-

"""
Teste End-to-End do Framework ANTARES:
Simulação de um Reator Tubular 1D (Advecção-Difusão) usando o Método das Linhas.
"""

import matplotlib.pyplot as plt
import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.model import Model
from antares.core.template_units import m
from antares.plotter import Plotter

# Opcional: Aumentar a verbosidade para ver o CasADi a compilar as 50 EDOs
cfg.VERBOSITY_LEVEL = 2


# =============================================================================
# 1. DECLARAÇÃO DO MODELO FENOMENOLÓGICO
# =============================================================================
class ReatorTubular(Model):
    def __init__(self, name):
        super().__init__(name)
        # O maestro: dispara as rotinas declarativas na ordem certa!
        self()

    def DeclareVariables(self):
        # 1. Cria o domínio espacial (Eixo Z) usando a nova fábrica unificada
        self.z = self.createDomain(
            "Eixo_Z",
            unit=m,
            length=10.0,
            n_points=50,
            method="mol",
            diff_scheme="backward",
        )

        # 2. Cria a variável de Temperatura
        self.T = self.createVariable(
            "T", "K", "Temperatura", exposure_type="differential"
        )

        # A CORREÇÃO: Damos a ordem diretamente ao Modelo!
        # Isto força a execução do `self.variables[node.name] = node`
        #self.distributeVariable(self.T, self.z)
        self.T.distributeOn(self.z)

    def DeclareParameters(self):
        # Parâmetros de transporte (vazão e difusão térmica)
        self.v = self.createParameter("v", "m/s", "Velocidade", value=0.5)
        self.alpha = self.createParameter("alpha", "m^2/s", "Difusividade", value=0.05)

    def DeclareEquations(self):
        # 1. VETORES (PDEs): Usamos a forma residual (-) para que o Numpy preserve a AST
        pde_energia = self.T.Diff() - (
            -(self.v() * self.T.Grad()) + (self.alpha() * self.T.Div())
        )

        # Equações do "Volume" e Saída (Nós 1 até 49)
        self.createEquation("eq_energia_bulk", expr=pde_energia[1:])

        # 2. ESCALARES: O nó 0 não é uma matriz. Aqui, o operador lógico "==" funciona perfeitamente!
        # Fixamos a derivada em zero. A temperatura de 350K será dada na Condição Inicial.
        self.createEquation("bc_entrada", expr=self.T.Diff()[0] == 0.0)


# =============================================================================
# 2. INSTANCIAÇÃO E SETUP
# =============================================================================
if __name__ == "__main__":
    # Instancia o reator
    reator = ReatorTubular("PFR_Aquecimento")

    # Dicionário de Condições Iniciais usando os nomes reais (com o prefixo do modelo)
    ic_dict = {}
    for i in range(reator.z.n_points):
        nome_no = f"T_PFR_Aquecimento_Eixo_Z_{i}"
        if i == 0:
            ic_dict[nome_no] = 350.0  # T de entrada cravada em 350K
        else:
            ic_dict[nome_no] = 300.0  # O restante do reator começa frio (300K)

    # Grelha de tempo de simulação (0 a 30 segundos)
    t_span = np.linspace(0, 30, 150)

    # =============================================================================
    # 3. EXECUÇÃO DA SIMULAÇÃO
    # =============================================================================
    simulador = Simulator(model=reator)
    resultados = simulador.run(t_span, initial_conditions=ic_dict)

    # =============================================================================
    # 4. GERAÇÃO DE GRÁFICOS (DINÂMICO E ESPACIAL)
    # =============================================================================
    # Gráfico 1: Evolução no Tempo em pontos específicos do reator
    plotador = Plotter(resultados)

    # Chamamos as variáveis pelo nome de registo completo
    variaveis_tempo = [
        "T_PFR_Aquecimento_Eixo_Z_0",
        "T_PFR_Aquecimento_Eixo_Z_12",
        "T_PFR_Aquecimento_Eixo_Z_25",
        "T_PFR_Aquecimento_Eixo_Z_49",
    ]

    legendas_tempo = {
        "T_PFR_Aquecimento_Eixo_Z_0": "Z = 0.0m (Entrada)",
        "T_PFR_Aquecimento_Eixo_Z_12": "Z = 2.5m",
        "T_PFR_Aquecimento_Eixo_Z_25": "Z = 5.0m",
        "T_PFR_Aquecimento_Eixo_Z_49": "Z = 10.0m (Saída)",
    }

    plotador.plot(
        variables=variaveis_tempo,
        title="Dinâmica de Aquecimento do Reator",
        xlabel="Tempo de Simulação (s)",
        ylabel="Temperatura (K)",
        legend_labels=legendas_tempo,
        show=True,
    )

    # Gráfico 2: Perfil Espacial no Tempo Final
    plotador._apply_aesthetics()

    estado_final = resultados.history.iloc[-1]

    # Extração elegante do perfil usando os prefixos
    nomes_T = [f"T_PFR_Aquecimento_Eixo_Z_{i}" for i in range(reator.z.n_points)]
    perfil_T = estado_final[nomes_T].values

    plt.figure(figsize=(10, 6))
    plt.plot(
        reator.z.grid, perfil_T, linewidth=3, color="#d55e00", marker="o", markersize=4
    )
    plt.title(
        f"Perfil Espacial de Temperatura (t = {t_span[-1]} s)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Posição Z (m)", fontsize=12)
    plt.ylabel("Temperatura (K)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
