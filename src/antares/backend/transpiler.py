# -*- coding: utf-8 -*-

"""
ANTARES Transpiler Module.

Responsible for converting the symbolic expression tree (SymPy) declared
in the model into a high-performance computational graph (CasADi).
Incorporates advanced compilation techniques such as Batch Lambdification 
and Smart Algebraic Isolation to ensure instant setup times even for 
massive multidimensional PDE grids.
"""

import warnings
import casadi as ca
import sympy as sp

import antares.core.GLOBAL_CFG as cfg
from antares.core.error_definitions import UnexpectedValueError


class CasadiTranspiler:
    """
    Handles the translation of symbolic equations to CasADi MX objects, 
    ensuring strict vector alignment, automatic derivative isolation, 
    and batched JIT compilation.
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
        Executes the translation of the entire model by mapping variables 
        and batch-compiling the equations.

        :return: A structured DAE dictionary ready for CasADi integration.
        :rtype: dict
        """
        self._map_variables_and_parameters()
        self._translate_equations()

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
            print(f"[DEBUG] Transpilation of '{self.model.name}' finished successfully.")

        return self.dae_dict

    def _map_variables_and_parameters(self):
        """
        Creates CasADi symbols and maps them exactly to their SymPy counterparts.
        Classifies variables into differential, algebraic, and parameter vectors.
        """
        # 1. Map Variables
        for var_name, var_obj in self.model.variables.items():
            ca_sym = ca.MX.sym(var_name)
            self.sym_map[sp.Symbol(var_name)] = ca_sym

            if var_obj.type == "differential":
                self.x_vars.append(ca_sym)
                self.x_names_order.append(var_name)
            else:
                self.z_vars.append(ca_sym)

        # 2. Map Parameters
        for par_name, par_obj in self.model.parameters.items():
            ca_sym = ca.MX.sym(par_name)
            self.sym_map[sp.Symbol(par_name)] = ca_sym
            self.p_vars.append(ca_sym)

        # 3. Map Constants
        for const_name, const_obj in self.model.constants.items():
            self.sym_map[sp.Symbol(const_name)] = const_obj.value

    def _translate_equations(self):
        """
        Uses high-performance algebraic substitution to isolate derivatives and 
        batch-lambdify to convert entire expression arrays into CasADi operations
        in a single call, eliminating massive loop overheads. Includes an automatic 
        fallback to heavy symbolic solvers if highly non-linear implicit ODEs are detected.

        :raises UnexpectedValueError: If a differential equation lacks a derivative or has multiples.
        :raises ValueError: If analytical isolation fails or unmapped symbols exist.
        """
        sympy_symbols_list = list(self.sym_map.keys())
        casadi_symbols_list = list(self.sym_map.values())

        ode_sympy_list = []
        ode_base_names = []
        alg_sympy_list = []

        # =====================================================================
        # 1. MATHEMATICAL ISOLATION AND BATCH PREPARATION
        # =====================================================================
        for eq_name, eq_obj in self.model.equations.items():
            expr_sympy = eq_obj.equation_expression.repr_symbolic

            if eq_obj.type == "differential":
                dot_symbols = [s for s in expr_sympy.free_symbols if str(s).endswith("_dot")]

                if not dot_symbols:
                    raise UnexpectedValueError(
                        f"Equation '{eq_name}' was marked as differential but does not contain a Diff() operator."
                    )
                if len(dot_symbols) > 1:
                    raise UnexpectedValueError(
                        f"Equation '{eq_name}' contains more than one derivative."
                    )

                dot_sym = dot_symbols[0]

                # -------------------------------------------------------------
                # THE SMART ISOLATOR ENGINE
                # -------------------------------------------------------------
                term_B = expr_sympy.subs(dot_sym, 0.0)
                term_A = expr_sympy.diff(dot_sym)

                # ARCHITECTURAL SAFEGUARD: Check for strict linearity
                # If the second derivative w.r.t dot_sym is NOT zero, 
                # the equation is a highly non-linear implicit ODE.
                is_strictly_linear = sp.simplify(term_A.diff(dot_sym)) == 0

                if is_strictly_linear:
                    # The Fast Track (99.9% of phenomenological models)
                    isolated_rhs = -term_B / term_A
                else:
                    # The Heavy Fallback (Mathematical abstract models)
                    if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                        warnings.warn(
                            f"[ANTARES] Highly non-linear implicit derivative detected in '{eq_name}'. "
                            f"Falling back to heavy symbolic isolation. Compilation may take longer."
                        )
                    
                    isolated_expr_list = sp.solve(expr_sympy, dot_sym)
                    if not isolated_expr_list:
                        raise ValueError(
                            f"Could not analytically isolate the derivative in equation '{eq_name}'. "
                            f"The implicit system might be unresolvable symbolically."
                        )
                    # We take the first mathematical root
                    isolated_rhs = isolated_expr_list[0]

                # -------------------------------------------------------------
                # The Phantom Symbol Guard
                unmapped = [s for s in isolated_rhs.free_symbols if s not in self.sym_map]
                if unmapped:
                    raise ValueError(
                        f"\n[ANTARES AST ERROR] Unmapped symbols in equation '{eq_name}': {unmapped}.\n"
                        f"These symbols appeared in the mathematical tree but were never declared in the model!"
                    )

                ode_sympy_list.append(isolated_rhs)
                ode_base_names.append(str(dot_sym).replace("_dot", ""))

            else:
                alg_sympy_list.append(expr_sympy)

        type_mapping = {
            "Float": float,
            "Integer": int,
            "Rational": lambda n, d: float(n) / float(d),
        }

        # =====================================================================
        # 2. BATCH TRANSLATION (The Performance Engine)
        # =====================================================================
        ode_eqs_dict = {}
        
        # Batch Lambdify ODEs
        if ode_sympy_list:
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
                print(f"[DEBUG] Batch transpiling {len(ode_sympy_list)} ODEs...")
                
            # Compiles the whole array at once
            func_casadi_ode = sp.lambdify(sympy_symbols_list, ode_sympy_list, modules=[type_mapping, ca])
            ode_casadi_results = func_casadi_ode(*casadi_symbols_list)
            
            # Unpack results safely (lambdify returns a scalar if list length is 1)
            if not isinstance(ode_casadi_results, (list, tuple)):
                ode_casadi_results = [ode_casadi_results]
                
            for name, ca_expr in zip(ode_base_names, ode_casadi_results):
                ode_eqs_dict[name] = ca_expr

        # Batch Lambdify ALGs
        alg_casadi_results = []
        if alg_sympy_list:
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
                print(f"[DEBUG] Batch transpiling {len(alg_sympy_list)} Algebraic equations...")
                
            func_casadi_alg = sp.lambdify(sympy_symbols_list, alg_sympy_list, modules=[type_mapping, ca])
            alg_casadi_results = func_casadi_alg(*casadi_symbols_list)
            
            if not isinstance(alg_casadi_results, (list, tuple)):
                alg_casadi_results = [alg_casadi_results]

        # =====================================================================
        # 3. STRICT VECTOR ALIGNMENT
        # =====================================================================
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
            "alg": ca.vertcat(*alg_casadi_results) if alg_casadi_results else ca.MX(),
        }