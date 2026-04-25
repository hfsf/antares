# -*- coding: utf-8 -*-

"""
Benchmark de Planta Acoplada - ANTARES V5
Simulação de 4 Operações Unitárias (Misturador, 2x CSTR, Splitter) com Loop de Reciclo.
Validação da resolução Orientada a Equações contra a Solução Analítica.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model

# Configurações do Framework
cfg.VERBOSITY_LEVEL = 1
cfg.USE_C_CODE_COMPILATION = False  # Roda instantaneamente sem JIT (0D)

# =============================================================================
# 1. BIBLIOTECA DE OPERAÇÕES UNITÁRIAS (O FRONTEND)
# =============================================================================


class Mixer(Model):
    def __init__(self, name):
        super().__init__(name, description="Misturador Ideal")
        self()

    def DeclareVariables(self):
        self.F_in1 = self.createVariable("F_in1", "L/min", exposure_type="algebraic")
        self.C_in1 = self.createVariable("C_in1", "mol/L", exposure_type="algebraic")
        self.F_in2 = self.createVariable("F_in2", "L/min", exposure_type="algebraic")
        self.C_in2 = self.createVariable("C_in2", "mol/L", exposure_type="algebraic")

        self.F_out = self.createVariable("F_out", "L/min", exposure_type="algebraic")
        self.C_out = self.createVariable("C_out", "mol/L", exposure_type="algebraic")

    def DeclareEquations(self):
        # Balanço de Massa Global
        eq_fluxo = self.F_out() - (self.F_in1() + self.F_in2())
        self.createEquation("balanco_fluxo", expr=eq_fluxo)

        # Balanço por Espécie (Massa)
        eq_especie = (self.F_out() * self.C_out()) - (
            self.F_in1() * self.C_in1() + self.F_in2() * self.C_in2()
        )
        self.createEquation("balanco_especies", expr=eq_especie)


class CSTR(Model):
    def __init__(self, name):
        super().__init__(name, description="Reator Contínuo de Mistura Perfeita")
        self()

    def DeclareVariables(self):
        self.F_in = self.createVariable("F_in", "L/min", exposure_type="algebraic")
        self.C_in = self.createVariable("C_in", "mol/L", exposure_type="algebraic")
        self.F_out = self.createVariable("F_out", "L/min", exposure_type="algebraic")

        # Variável Dinâmica
        self.C_out = self.createVariable("C_out", "mol/L", exposure_type="differential")
        self.setInitialCondition(self.C_out, 0.0)  # Inicia vazio (apenas com solvente)

    def DeclareParameters(self):
        self.V = self.createParameter("V", "L", value=500.0)
        self.k = self.createParameter("k", "min^-1", value=0.1)

    def DeclareEquations(self):
        # Assumindo densidade constante (F_in = F_out)
        self.createEquation("balanco_fluxo", expr=self.F_out() - self.F_in())

        # V * dC/dt = F_in*C_in - F_out*C_out - k*V*C_out
        eq_cinetica = (self.V() * self.C_out.Diff()) - (
            self.F_in() * self.C_in()
            - self.F_out() * self.C_out()
            - self.k() * self.V() * self.C_out()
        )
        self.createEquation("balanco_reacao", expr=eq_cinetica)


class Splitter(Model):
    def __init__(self, name):
        super().__init__(name, description="Divisor de Correntes (Tee)")
        self()

    def DeclareVariables(self):
        self.F_in = self.createVariable("F_in", "L/min", exposure_type="algebraic")
        self.C_in = self.createVariable("C_in", "mol/L", exposure_type="algebraic")

        self.F_out1 = self.createVariable("F_out1", "L/min", exposure_type="algebraic")
        self.C_out1 = self.createVariable("C_out1", "mol/L", exposure_type="algebraic")

        self.F_out2 = self.createVariable("F_out2", "L/min", exposure_type="algebraic")
        self.C_out2 = self.createVariable("C_out2", "mol/L", exposure_type="algebraic")

    def DeclareParameters(self):
        self.R = self.createParameter("R", "", "Fração de Reciclo", value=0.5)

    def DeclareEquations(self):
        self.createEquation(
            "split_fluxo_1", expr=self.F_out1() - (self.R() * self.F_in())
        )
        self.createEquation(
            "split_fluxo_2", expr=self.F_out2() - ((1.0 - self.R()) * self.F_in())
        )
        self.createEquation("split_conc_1", expr=self.C_out1() - self.C_in())
        self.createEquation("split_conc_2", expr=self.C_out2() - self.C_in())


# =============================================================================
# 2. A TOPOLOGIA GLOBAL (O MASTER FLOWSHEET)
# =============================================================================


class PlantaBenchmark(Model):
    def __init__(self, name):
        self.mix = Mixer("M1")
        self.R1 = CSTR("R1")
        self.R2 = CSTR("R2")
        self.spl = Splitter("S1")
        super().__init__(name, submodels=[self.mix, self.R1, self.R2, self.spl])
        self()

    def DeclareEquations(self):
        # 1. Alimentação Fresca (Forçando a entrada do Misturador)
        self.createEquation("Alimentacao_F", expr=self.mix.F_in1() - 100.0)
        self.createEquation("Alimentacao_C", expr=self.mix.C_in1() - 1.0)

        # 2. Conexões Topológicas usando a Classe Connection V5
        # Ligação 1: Mixer -> R1
        Connection("L1_F", self.mix, self.R1, "F_out", "F_in").apply_to(self)
        Connection("L1_C", self.mix, self.R1, "C_out", "C_in").apply_to(self)

        # Ligação 2: R1 -> R2
        Connection("L2_F", self.R1, self.R2, "F_out", "F_in").apply_to(self)
        Connection("L2_C", self.R1, self.R2, "C_out", "C_in").apply_to(self)

        # Ligação 3: R2 -> Splitter
        Connection("L3_F", self.R2, self.spl, "F_out", "F_in").apply_to(self)
        Connection("L3_C", self.R2, self.spl, "C_out", "C_in").apply_to(self)

        # Ligação 4: O GRANDE DESAFIO - Loop de Reciclo: Splitter -> Mixer
        Connection("L_Recycle_F", self.spl, self.mix, "F_out1", "F_in2").apply_to(self)
        Connection("L_Recycle_C", self.spl, self.mix, "C_out1", "C_in2").apply_to(self)


# =============================================================================
# 3. EXECUÇÃO E AUDITORIA
# =============================================================================

if __name__ == "__main__":
    planta = PlantaBenchmark("Complexo_CSTRs")

    print("\n[INIT] Disparando o Motor de Integração DAE (IDAS)...")
    simulador = Simulator(model=planta)

    # Simulamos 150 minutos (mais que suficiente para a planta atingir o Estado Estacionário)
    t_span = np.linspace(0, 150, 300)

    # -------------------------------------------------------------------------
    # PREVENÇÃO DE JACOBIANO SINGULAR (Zero-Flow Bilinearity Trap)
    # Fornecemos "chutes" não-nulos para as variáveis algébricas. Isso garante
    # que as derivadas dos termos bilineares (Fluxo * Concentração) não zerem
    # no cálculo das Condições Iniciais Algébricas (calc_ic) no t=0.
    # -------------------------------------------------------------------------
    chutes_iniciais = {
        planta.mix.F_out.name: 100.0,
        planta.R1.F_out.name: 100.0,
        planta.R2.F_out.name: 100.0,
        planta.spl.F_out1.name: 50.0,
        planta.spl.F_out2.name: 50.0,
        planta.mix.C_out.name: 1.0,
        planta.R1.F_in.name: 100.0,
        planta.R2.F_in.name: 100.0,
    }

    resultados = simulador.run(t_span, initial_conditions=chutes_iniciais)

    # -------------------------------------------------------------------------
    # 4. COMPARAÇÃO COM A LITERATURA
    # -------------------------------------------------------------------------

    # Extraímos a última linha da matriz de resultados (t = 150 min, Steady-State)
    C1_sim = resultados.get_variable(planta.R1.C_out.name)[-1]
    C2_sim = resultados.get_variable(planta.R2.C_out.name)[-1]

    # Resultados Analíticos Exatos (calculados do balanço global de massa)
    C1_lit = 0.50 / 0.85
    C2_lit = 0.40 / 0.85

    erro_C1 = abs(C1_sim - C1_lit) / C1_lit * 100.0
    erro_C2 = abs(C2_sim - C2_lit) / C2_lit * 100.0

    print("\n" + "=" * 60)
    print("Solução de Reciclo Simultâneo (Sem Tearing / Equation-Oriented)")
    print("-" * 60)
    print(
        f"[{planta.R1.name} - C_out] | Analítico: {C1_lit:.6f} | ANTARES: {C1_sim:.6f} | Erro Relativo: {erro_C1:.2e} %"
    )
    print(
        f"[{planta.R2.name} - C_out] | Analítico: {C2_lit:.6f} | ANTARES: {C2_sim:.6f} | Erro Relativo: {erro_C2:.2e} %"
    )
    print("=" * 60)
