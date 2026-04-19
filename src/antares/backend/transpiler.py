# -*- coding: utf-8 -*-

"""
ANTARES Transpiler Module.
Responsible for converting the symbolic expression tree (SymPy) declared
in the model into a high-performance computational graph (CasADi).
"""

import casadi as ca
import sympy as sp

import antares.core.GLOBAL_CFG as cfg
from antares.core.error_definitions import UnexpectedValueError


class CasadiTranspiler:
    """
    Definition of the CasadiTranspiler class. Handles the translation of symbolic
    equations to CasADi MX objects, ensuring strict vector alignment and derivative isolation.
    """

    def __init__(self, model):
        """
        Initializes the transpiler with a flat mathematical model.

        :param Model model: Instance of the Master Flowsheet containing all equations.
        """
        self.model = model

        # Fundamental mapping: {SymPy_Symbol: CasADi_MX_Symbol}
        self.sym_map = {}

        # Lists to maintain strict variable ordering (crucial for CasADi solvers)
        self.x_vars = []  # Differential states (x)
        self.z_vars = []  # Algebraic variables (z)
        self.p_vars = []  # Parameters (p)

        # Store the names of X variables in exact order for ODE alignment
        self.x_names_order = []

        # Final dictionary structure to be delivered to the solver
        self.dae_dict = {}

    def transpile(self):
        """
        Executes the translation of the entire model.

        :return: A structured DAE dictionary ready for casadi.integrator.
        :rtype: dict
        """
        self._map_variables_and_parameters()
        self._translate_equations()

        if cfg.VERBOSITY_LEVEL >= 2:
            print(
                f"[DEBUG] Transpilation of '{self.model.name}' finished successfully."
            )

        return self.dae_dict

    def _map_variables_and_parameters(self):
        """
        Creates CasADi symbols and maps them exactly to their SymPy counterparts.
        """
        # 1. Map Variables
        for var_name, var_obj in self.model.variables.items():
            ca_sym = ca.MX.sym(var_name)
            self.sym_map[sp.Symbol(var_name)] = ca_sym  # <--- CORRIGIDO AQUI

            if var_obj.type == "differential":
                self.x_vars.append(ca_sym)
                self.x_names_order.append(var_name)
            else:
                self.z_vars.append(ca_sym)

        # 2. Map Parameters
        for par_name, par_obj in self.model.parameters.items():
            ca_sym = ca.MX.sym(par_name)
            self.sym_map[sp.Symbol(par_name)] = ca_sym  # <--- CORRIGIDO AQUI
            self.p_vars.append(ca_sym)

        # 3. Map Constants
        for const_name, const_obj in self.model.constants.items():
            self.sym_map[sp.Symbol(const_name)] = const_obj.value  # <--- CORRIGIDO AQUI

    def _translate_equations(self):
        """
        Uses SymPy to analytically isolate derivatives and lambdify to convert
        mathematical expressions into CasADi operations.
        """
        # Temporary dictionary to guarantee strict ODE ordering
        ode_eqs_dict = {}
        alg_eqs = []

        sympy_symbols_list = list(self.sym_map.keys())
        casadi_symbols_list = list(self.sym_map.values())

        for eq_name, eq_obj in self.model.equations.items():
            expr_sympy = eq_obj.equation_expression.repr_symbolic

            if eq_obj.type == "differential":
                # --- STEP A: ISOLATE THE DERIVATIVE ---

                # 1. Find which symbol is the "_dot" marker inside the equation
                dot_symbols = [
                    s for s in expr_sympy.free_symbols if str(s).endswith("_dot")
                ]

                if not dot_symbols:
                    raise UnexpectedValueError(
                        f"Equation '{eq_name}' was marked as differential but does not "
                        f"contain a Diff() operator."
                    )
                if len(dot_symbols) > 1:
                    raise UnexpectedValueError(
                        f"Equation '{eq_name}' contains more than one derivative. "
                        f"Please write one differential equation per state variable."
                    )

                dot_sym = dot_symbols[0]

                if cfg.VERBOSITY_LEVEL >= 2:
                    print(
                        f"[DEBUG] Isolating derivative {dot_sym} for equation '{eq_name}'..."
                    )

                # 2. Isolate the derivative mathematically (expr_sympy = 0)
                # Returns the right-hand side (RHS): x_dot = RHS
                isolated_expr_list = sp.solve(expr_sympy, dot_sym)

                if not isolated_expr_list:
                    raise ValueError(
                        f"Could not analytically isolate the derivative in equation '{eq_name}'."
                    )

                isolated_rhs = isolated_expr_list[0]

                # --- STEP B: TRANSLATE TO CASADI ---
                # NOVO: O Cão de Guarda de Símbolos Fantasmas!
                unmapped = [
                    s for s in isolated_rhs.free_symbols if s not in self.sym_map
                ]
                if unmapped:
                    raise ValueError(
                        f"\n[ANTARES AST ERROR] Símbolos não mapeados na equação '{eq_name}': {unmapped}.\n"
                        f"Estes símbolos apareceram na árvore matemática mas nunca foram declarados no modelo!"
                    )

                type_mapping = {
                    "Float": float,
                    "Integer": int,
                    "Rational": lambda n, d: float(n) / float(d),
                }

                # 1. CORREÇÃO: Usar sympy_symbols_list e isolated_rhs!
                func_casadi_gen = sp.lambdify(
                    sympy_symbols_list,
                    isolated_rhs,
                    modules=[type_mapping, ca],
                )
                expr_casadi = func_casadi_gen(*casadi_symbols_list)

                # Store in the dictionary using the base variable name (removing '_dot')
                base_var_name = str(dot_sym).replace("_dot", "")
                ode_eqs_dict[base_var_name] = expr_casadi

            else:
                # Algebraic equations do not need isolation, evaluate directly
                type_mapping = {
                    "Float": float,
                    "Integer": int,
                    "Rational": lambda n, d: float(n) / float(d),
                }

                # AQUI DEVE SER expr_sympy!
                func_casadi_gen = sp.lambdify(
                    sympy_symbols_list, expr_sympy, modules=[type_mapping, ca]
                )
                expr_casadi = func_casadi_gen(*casadi_symbols_list)
                alg_eqs.append(expr_casadi)

        # --- STEP C: STRICT VECTOR ALIGNMENT ---
        ode_eqs = []
        for x_name in self.x_names_order:
            if x_name not in ode_eqs_dict:
                raise ValueError(
                    f"Differential variable '{x_name}' was declared, but no equation "
                    f"containing its Diff() operator was found."
                )
            ode_eqs.append(ode_eqs_dict[x_name])

        # Assemble the final DAE structure
        self.dae_dict = {
            "x": ca.vertcat(*self.x_vars) if self.x_vars else ca.MX(),
            "z": ca.vertcat(*self.z_vars) if self.z_vars else ca.MX(),
            "p": ca.vertcat(*self.p_vars) if self.p_vars else ca.MX(),
            "ode": ca.vertcat(*ode_eqs) if ode_eqs else ca.MX(),
            "alg": ca.vertcat(*alg_eqs) if alg_eqs else ca.MX(),
        }
