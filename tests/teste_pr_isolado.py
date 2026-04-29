# -*- coding: utf-8 -*-

"""
Sandbox de Diagnóstico Termodinâmico - Peng-Robinson EOS
Isola a matemática do CasADi sem o peso do KINSOL ou da topologia do Flowsheet.
"""

import casadi as ca
import numpy as np

try:
    from thermo.chemical import Chemical
except ImportError:
    raise ImportError("Instale a biblioteca 'thermo' (pip install thermo)")

print("=" * 70)
print(" DIAGNÓSTICO CASADI ISOLADO - PENG-ROBINSON EOS")
print("=" * 70)

# 1. Recuperando Propriedades Críticas reais
components = ["methane", "propane", "pentane"]
data = {}
print("[1/4] Extraindo dados termodinâmicos da base...")
for comp in components:
    chem = Chemical(comp)
    data[comp] = {"Tc": chem.Tc, "Pc": chem.Pc, "omega": chem.omega}
    print(
        f"  -> {comp.capitalize()}: Tc={chem.Tc:.2f} K, Pc={chem.Pc / 1e5:.2f} bar, w={chem.omega:.4f}"
    )

# 2. Criando as Variáveis Simbólicas do CasADi
print("\n[2/4] Criando Variáveis Simbólicas (CasADi SX)...")
T = ca.SX.sym("T")
P = ca.SX.sym("P")
x = ca.SX.sym("x", 3)
y = ca.SX.sym("y", 3)
Z_L = ca.SX.sym("Z_L")
Z_V = ca.SX.sym("Z_V")

variables = ca.vertcat(T, P, x, y, Z_L, Z_V)

# 3. Reconstruindo a Matemática do Peng-Robinson
print("[3/4] Montando Árvore Algébrica do Peng-Robinson...")
R = 8.314

a_comp = []
b_comp = []
for comp in components:
    Tc = data[comp]["Tc"]
    Pc = data[comp]["Pc"]
    w = data[comp]["omega"]

    kappa = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alpha = (1.0 + kappa * (1.0 - ca.sqrt(T / Tc))) ** 2
    a_comp.append(0.45724 * (R**2 * Tc**2 / Pc) * alpha)
    b_comp.append(0.07780 * (R * Tc / Pc))

# Regras de Mistura
aL, bL, aV, bV = 0.0, 0.0, 0.0, 0.0
for i in range(3):
    bL += x[i] * b_comp[i]
    bV += y[i] * b_comp[i]
    for j in range(3):
        aL += x[i] * x[j] * ca.sqrt(a_comp[i] * a_comp[j])
        aV += y[i] * y[j] * ca.sqrt(a_comp[i] * a_comp[j])

A_L = (aL * P) / (R**2 * T**2)
B_L = (bL * P) / (R * T)
A_V = (aV * P) / (R**2 * T**2)
B_V = (bV * P) / (R * T)

# Equações Residuais (Raízes Cúbicas)
res_L = (
    Z_L**3
    - (1.0 - B_L) * Z_L**2
    + (A_L - 2.0 * B_L - 3.0 * B_L**2) * Z_L
    - (A_L * B_L - B_L**2 - B_L**3)
)
res_V = (
    Z_V**3
    - (1.0 - B_V) * Z_V**2
    + (A_V - 2.0 * B_V - 3.0 * B_V**2) * Z_V
    - (A_V * B_V - B_V**2 - B_V**3)
)

# Logaritmos de Fugacidade
ln_phi_L = []
ln_phi_V = []
for i in range(3):
    # LÍQUIDO
    bi_b_L = b_comp[i] / bL
    sum_x_aij_L = ca.sqrt(a_comp[i] * aL)
    term_a_L = (2.0 * sum_x_aij_L / aL) - bi_b_L

    arg1_L = Z_L - B_L
    arg2_L = Z_L + (1.0 + np.sqrt(2.0)) * B_L
    arg3_L = Z_L + (1.0 - np.sqrt(2.0)) * B_L

    phi_L = (
        bi_b_L * (Z_L - 1.0)
        - ca.log(arg1_L)
        - (A_L / (2.0 * np.sqrt(2.0) * B_L)) * term_a_L * ca.log(arg2_L / arg3_L)
    )
    ln_phi_L.append(phi_L)

    # VAPOR
    bi_b_V = b_comp[i] / bV
    sum_x_aij_V = ca.sqrt(a_comp[i] * aV)
    term_a_V = (2.0 * sum_x_aij_V / aV) - bi_b_V

    arg1_V = Z_V - B_V
    arg2_V = Z_V + (1.0 + np.sqrt(2.0)) * B_V
    arg3_V = Z_V + (1.0 - np.sqrt(2.0)) * B_V

    phi_V = (
        bi_b_V * (Z_V - 1.0)
        - ca.log(arg1_V)
        - (A_V / (2.0 * np.sqrt(2.0) * B_V)) * term_a_V * ca.log(arg2_V / arg3_V)
    )
    ln_phi_V.append(phi_V)

# Balanço Isofugacidade Logarítmica
fug_eqs = []
for i in range(3):
    eq = ca.log(x[i]) + ln_phi_L[i] - ca.log(y[i]) - ln_phi_V[i]
    fug_eqs.append(eq)

# Agrupando todas as equações
equations = ca.vertcat(res_L, res_V, *fug_eqs)

# Criando as Funções de Avaliação CasADi
f_eval = ca.Function("f_eval", [variables], [equations])
jac_eval = ca.Function("jac_eval", [variables], [ca.jacobian(equations, variables)])

# Funções de Diagnóstico Intermediário
f_diag = ca.Function(
    "f_diag", [variables], [A_L, B_L, arg1_L, arg3_L, A_V, B_V, arg1_V, arg3_V]
)

# =============================================================================
# 4. AVALIAÇÃO NUMÉRICA (O TESTE DE FOGO)
# =============================================================================
print("\n[4/4] Injetando Condições do Benchmark...")
T_num = 310.0
P_num = 25.0 * 1e5
x_num = [0.1, 0.2, 0.7]  # Fração assimétrica para quebrar singularidade
y_num = [0.7, 0.2, 0.1]
Z_L_num = 0.1  # Chute de Compressibilidade Líquida
Z_V_num = 0.9  # Chute de Compressibilidade Vapor

inputs = ca.vertcat(T_num, P_num, *x_num, *y_num, Z_L_num, Z_V_num)

print("\n--- AVALIANDO VARIÁVEIS INTERMEDIÁRIAS (PERIGO DE NaN) ---")
try:
    AL_n, BL_n, arg1L_n, arg3L_n, AV_n, BV_n, arg1V_n, arg3V_n = f_diag(inputs)
    print(f"Líquido -> A: {float(AL_n):.5f}, B: {float(BL_n):.5f}")
    print(f"Líquido -> (Z - B): {float(arg1L_n):.5f} (SE < 0 -> GERA NaN NO LOG!)")
    print(f"Líquido -> Denominador do Log: {float(arg3L_n):.5f} (SE <= 0 -> GERA NaN!)")
    print(f"\nVapor   -> A: {float(AV_n):.5f}, B: {float(BV_n):.5f}")
    print(f"Vapor   -> (Z - B): {float(arg1V_n):.5f} (SE < 0 -> GERA NaN NO LOG!)")
    print(f"Vapor   -> Denominador do Log: {float(arg3V_n):.5f} (SE <= 0 -> GERA NaN!)")
except RuntimeError as e:
    print("FALHA CATASTRÓFICA NA AVALIAÇÃO INTERMEDIÁRIA:")
    print(e)

print("\n--- AVALIANDO RESÍDUOS DAS EQUAÇÕES (f = 0) ---")
try:
    res_num = f_eval(inputs)
    print(res_num)
except RuntimeError as e:
    print("FALHA CATASTRÓFICA NOS RESÍDUOS:")
    print(e)

print("\n--- AVALIANDO JACOBIANO ANALÍTICO (J = df/dx) ---")
try:
    jac_num = jac_eval(inputs)
    print(np.array(jac_num))
except RuntimeError as e:
    print("FALHA CATASTRÓFICA NO JACOBIANO (AQUI ESTÁ O ERRO DO SOLVER):")
    print(e)

print("=" * 70)
