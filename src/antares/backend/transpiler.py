# -*- coding: utf-8 -*-

"""
ANTARES Transpiler Module.

Responsible for converting the symbolic expression tree (SymPy) declared
in the model into a high-performance computational graph (CasADi).
Incorporates advanced compilation techniques such as Batch Lambdification 
and Smart Algebraic Isolation. Now featuring tqdm progress tracking for UX.
"""

import warnings
import casadi as ca
import sympy as sp

import antares.core.GLOBAL_CFG as cfg
from antares.core.error_definitions import UnexpectedValueError

# Graceful import for tqdm (Industrial standard for progress bars)
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm if the user hasn't installed it
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Override progress bar if user has tqdm but has disabled in in GLOBAL_CFG
# It is a dirty fix? Perhaps. But should work.
if getattr(cfg, "SHOW_LOADING_BARS", True) is False:
    HAS_TQDM = False

class CasadiTranspiler:
    """
    Handles the translation of symbolic equations to CasADi MX objects, 
    ensuring strict vector alignment, automatic derivative isolation, 
    and batched JIT compilation.
    """

    def __init__(self, model):
        self.model = model
        self.sym_map = {}
        self.x_vars = []  
        self.z_vars = []  
        self.p_vars = []  
        self.x_names_order = []
        self.dae_dict = {}

    def transpile(self):
        """Executes the translation with progress tracking."""
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"\n[ANTARES TRANSPILER] Starting compilation for '{self.model.name}'...")
            if not HAS_TQDM:
                print("  -> Tip: Install 'tqdm' (pip install tqdm) for progress bars.")

        self._map_variables_and_parameters()
        self._translate_equations()

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"[ANTARES TRANSPILER] Setup finished successfully.\n")

        return self.dae_dict

    def _map_variables_and_parameters(self):
        """Creates CasADi symbols with a fast progress bar."""
        
        # 1. Map Variables
        var_items = self.model.variables.items()
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            var_items = tqdm(var_items, desc="Mapping Variables", unit=" node", leave=False)
            
        for var_name, var_obj in var_items:
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
        Uses algebraic substitution to isolate derivatives. This is the 
        heaviest Python loop, so it gets a detailed progress bar.
        """
        sympy_symbols_list = list(self.sym_map.keys())
        casadi_symbols_list = list(self.sym_map.values())

        ode_sympy_list = []
        ode_base_names = []
        alg_sympy_list = []

        # =====================================================================
        # 1. MATHEMATICAL ISOLATION (With Progress Bar)
        # =====================================================================
        eq_items = self.model.equations.items()
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            eq_items = tqdm(eq_items, desc="Isolating Derivatives", unit=" eq")

        for eq_name, eq_obj in eq_items:
            expr_sympy = eq_obj.equation_expression.repr_symbolic

            if eq_obj.type == "differential":
                dot_symbols = [s for s in expr_sympy.free_symbols if str(s).endswith("_dot")]

                if not dot_symbols:
                    raise UnexpectedValueError(f"Equation '{eq_name}' missing Diff() operator.")
                if len(dot_symbols) > 1:
                    raise UnexpectedValueError(f"Equation '{eq_name}' has >1 derivative.")

                dot_sym = dot_symbols[0]

                # The Smart Isolator Engine
                term_B = expr_sympy.subs(dot_sym, 0.0)
                term_A = expr_sympy.diff(dot_sym)

                is_strictly_linear = sp.simplify(term_A.diff(dot_sym)) == 0

                if is_strictly_linear:
                    isolated_rhs = -term_B / term_A
                else:
                    isolated_expr_list = sp.solve(expr_sympy, dot_sym)
                    isolated_rhs = isolated_expr_list[0]

                ode_sympy_list.append(isolated_rhs)
                ode_base_names.append(str(dot_sym).replace("_dot", ""))

            else:
                alg_sympy_list.append(expr_sympy)

        type_mapping = {"Float": float, "Integer": int, "Rational": lambda n, d: float(n) / float(d)}

        # =====================================================================
        # 2. BATCH TRANSLATION (The Performance Engine)
        # =====================================================================
        ode_eqs_dict = {}
        
        if ode_sympy_list:
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print(f" -> Batch Lambdifying {len(ode_sympy_list)} ODEs (Please wait...)")
                
            func_casadi_ode = sp.lambdify(sympy_symbols_list, ode_sympy_list, modules=[type_mapping, ca])
            ode_casadi_results = func_casadi_ode(*casadi_symbols_list)
            
            if not isinstance(ode_casadi_results, (list, tuple)):
                ode_casadi_results = [ode_casadi_results]
                
            for name, ca_expr in zip(ode_base_names, ode_casadi_results):
                ode_eqs_dict[name] = ca_expr

        alg_casadi_results = []
        if alg_sympy_list:
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print(f" -> Batch Lambdifying {len(alg_sympy_list)} Algebraic constraints...")
                
            func_casadi_alg = sp.lambdify(sympy_symbols_list, alg_sympy_list, modules=[type_mapping, ca])
            alg_casadi_results = func_casadi_alg(*casadi_symbols_list)
            
            if not isinstance(alg_casadi_results, (list, tuple)):
                alg_casadi_results = [alg_casadi_results]

        # =====================================================================
        # 3. STRICT VECTOR ALIGNMENT
        # =====================================================================
        ode_eqs = []
        for x_name in self.x_names_order:
            ode_eqs.append(ode_eqs_dict[x_name])

        self.dae_dict = {
            "x": ca.vertcat(*self.x_vars) if self.x_vars else ca.MX(),
            "z": ca.vertcat(*self.z_vars) if self.z_vars else ca.MX(),
            "p": ca.vertcat(*self.p_vars) if self.p_vars else ca.MX(),
            "ode": ca.vertcat(*ode_eqs) if ode_eqs else ca.MX(),
            "alg": ca.vertcat(*alg_casadi_results) if alg_casadi_results else ca.MX(),
        }