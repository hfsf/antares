# -*- coding: utf-8 -*-

"""
ANTARES Transpiler Module.

V4 UPDATE: Vectorial Transpiler.
Translates mathematical block tensors into native CasADi symbolic vectors,
injecting finite difference Sparse Matrices dynamically and fully bypassing
the Scalar Unrolling graph explosion. Fully shielded against SymPy-CasADi type conflicts.
"""

import casadi as ca
import numpy as np
import scipy.sparse as sps
import sympy as sp

import antares.core.GLOBAL_CFG as cfg
from antares.core.error_definitions import UnexpectedValueError

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable


if getattr(cfg, "SHOW_LOADING_BARS", True) is False:
    HAS_TQDM = False


class CasadiTranspiler:
    def __init__(self, model):
        self.model = model
        self.sym_map = {}

        self.x_vars = []
        self.z_vars = []
        self.p_vars = []
        self.x_names_order = []

        self.dae_dict = {}
        self.distributed_var_objects_list = []

    def transpile(self):
        """Executes the Vectorial Translation Engine."""
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(
                f"\n[ANTARES TRANSPILER V4] Starting Block Vectorization for '{self.model.name}'..."
            )

        self._map_variables_and_parameters()
        self._translate_equations()

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"[ANTARES TRANSPILER V4] Setup finished successfully.\n")

        return self.dae_dict

    def _map_variables_and_parameters(self):
        """Maps distributed arrays directly to MX sparse vectors avoiding loop unpacking."""
        for var_name, var_obj in self.model.variables.items():
            if getattr(var_obj, "is_distributed", False):
                n_points = var_obj.n_points

                idx_diff = np.where(var_obj.node_types.flatten() == "differential")[
                    0
                ].tolist()
                idx_alg = np.where(var_obj.node_types.flatten() == "algebraic")[
                    0
                ].tolist()

                var_obj.idx_diff = idx_diff
                var_obj.idx_alg = idx_alg

                C_full = ca.MX.zeros(n_points)
                if idx_diff:
                    C_x = ca.MX.sym(var_name + "_x", len(idx_diff))
                    self.x_vars.append(C_x)
                    self.x_names_order.append(var_name)
                    C_full[idx_diff] = C_x

                    C_dot_full = ca.MX.zeros(n_points)
                    C_x_dot = ca.MX.sym(var_name + "_dot", len(idx_diff))
                    C_dot_full[idx_diff] = C_x_dot
                    self.sym_map[sp.Symbol(var_name + "_dot")] = C_dot_full

                if idx_alg:
                    C_z = ca.MX.sym(var_name + "_z", len(idx_alg))
                    self.z_vars.append(C_z)
                    C_full[idx_alg] = C_z

                # Fix V4: Força a amarração correta do nome exato do modelo!
                self.sym_map[sp.Symbol(var_name)] = C_full
                self.distributed_var_objects_list.append(var_obj)
            else:
                ca_sym = ca.MX.sym(var_name)
                # Fix V4: Força a amarração correta do nome exato do modelo!
                self.sym_map[sp.Symbol(var_name)] = ca_sym

                if var_obj.type == "differential":
                    self.x_vars.append(ca_sym)
                    self.x_names_order.append(var_name)
                    self.sym_map[sp.Symbol(var_name + "_dot")] = ca.MX.sym(
                        var_name + "_dot"
                    )
                else:
                    self.z_vars.append(ca_sym)

        for par_name, par_obj in self.model.parameters.items():
            ca_sym = ca.MX.sym(par_name)
            self.sym_map[sp.Symbol(par_name)] = ca_sym
            self.p_vars.append(ca_sym)

        for const_name, const_obj in self.model.constants.items():
            self.sym_map[sp.Symbol(const_name)] = const_obj.value

    # =========================================================================
    # MATRIX DELEGATION GENERATORS
    # =========================================================================

    def _scipy_to_casadi(self, mat):
        mat = mat.tocsc()
        colind = mat.indptr.astype(int).tolist()
        row = mat.indices.astype(int).tolist()
        data = mat.data.tolist()
        sparsity = ca.Sparsity(mat.shape[0], mat.shape[1], colind, row)
        return ca.DM(sparsity, data)

    def _build_laplacian_sparse(self, dom_obj):
        if hasattr(dom_obj, "z"):
            Lx, Ly, Lz = (
                sps.csr_matrix(dom_obj.x.B_matrix),
                sps.csr_matrix(dom_obj.y.B_matrix),
                sps.csr_matrix(dom_obj.z.B_matrix),
            )
            Ix, Iy, Iz = (
                sps.eye(dom_obj.x.n_points),
                sps.eye(dom_obj.y.n_points),
                sps.eye(dom_obj.z.n_points),
            )
            return (
                sps.kron(sps.kron(Lx, Iy), Iz)
                + sps.kron(sps.kron(Ix, Ly), Iz)
                + sps.kron(sps.kron(Ix, Iy), Lz)
            )
        elif hasattr(dom_obj, "y"):
            Lx, Ly = (
                sps.csr_matrix(dom_obj.x.B_matrix),
                sps.csr_matrix(dom_obj.y.B_matrix),
            )
            Ix, Iy = sps.eye(dom_obj.x.n_points), sps.eye(dom_obj.y.n_points)
            return sps.kron(Lx, Iy) + sps.kron(Ix, Ly)
        return sps.csr_matrix(dom_obj.B_matrix)

    def _build_gradient_sparse(self, dom_obj, axis_name=None):
        if hasattr(dom_obj, "z"):
            Ix, Iy, Iz = (
                sps.eye(dom_obj.x.n_points),
                sps.eye(dom_obj.y.n_points),
                sps.eye(dom_obj.z.n_points),
            )
            if axis_name == dom_obj.x.name:
                return sps.kron(sps.kron(sps.csr_matrix(dom_obj.x.A_matrix), Iy), Iz)
            elif axis_name == dom_obj.y.name:
                return sps.kron(sps.kron(Ix, sps.csr_matrix(dom_obj.y.A_matrix)), Iz)
            elif axis_name == dom_obj.z.name:
                return sps.kron(sps.kron(Ix, Iy), sps.csr_matrix(dom_obj.z.A_matrix))
        elif hasattr(dom_obj, "y"):
            Ix, Iy = sps.eye(dom_obj.x.n_points), sps.eye(dom_obj.y.n_points)
            if axis_name == dom_obj.x.name:
                return sps.kron(sps.csr_matrix(dom_obj.x.A_matrix), Iy)
            elif axis_name == dom_obj.y.name:
                return sps.kron(Ix, sps.csr_matrix(dom_obj.y.A_matrix))
        return sps.csr_matrix(dom_obj.A_matrix)

    # =========================================================================
    # VECTORIAL TRANSLATION ENGINE
    # =========================================================================

    def _translate_equations(self):
        # 1. Segurança Máxima: Catch stray atoms (Mapeia símbolos órfãos)
        for eq_obj in self.model.equations.values():
            for atom in eq_obj.equation_expression.repr_symbolic.atoms(sp.Symbol):
                if atom not in self.sym_map:
                    self.sym_map[atom] = ca.MX.sym(atom.name)

        sympy_symbols_list = list(self.sym_map.keys())
        casadi_symbols_list = list(self.sym_map.values())

        # 2. Dicionário Unificado
        eval_dict = {
            "Float": float,
            "Integer": int,
            "Rational": lambda n, d: float(n) / float(d),
        }

        for var in self.distributed_var_objects_list:
            dom = var.domain
            B_ca = self._scipy_to_casadi(self._build_laplacian_sparse(dom))
            eval_dict[f"Laplacian_{dom.name}"] = lambda x, B=B_ca: ca.mtimes(B, x)

            for loc in [
                "left",
                "right",
                "west",
                "east",
                "x_start",
                "x_end",
                "top",
                "bottom",
                "north",
                "south",
                "y_start",
                "y_end",
                "front",
                "back",
                "z_start",
                "z_end",
            ]:
                axis = None
                if loc in ["left", "right", "west", "east", "x_start", "x_end"]:
                    axis = dom.x.name if hasattr(dom, "x") else None
                elif loc in ["bottom", "top", "south", "north", "y_start", "y_end"]:
                    axis = dom.y.name if hasattr(dom, "y") else None
                elif loc in ["front", "back", "z_start", "z_end"]:
                    axis = dom.z.name if hasattr(dom, "z") else None
                if axis:
                    A_n_ca = self._scipy_to_casadi(
                        self._build_gradient_sparse(dom, axis_name=axis)
                    )
                    eval_dict[f"NormalGradient_{dom.name}_{loc}"] = lambda x, A=A_n_ca: (
                        ca.mtimes(A, x)
                    )

        ode_acc = {
            v.name: ca.MX.zeros(v.n_points) for v in self.distributed_var_objects_list
        }
        ode_lumped = {}
        alg_acc = []

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(" -> Lambdifying Block Tensors into CasADi Graph...")

        for eq_name, eq_obj in self.model.equations.items():
            expr = eq_obj.equation_expression.repr_symbolic

            if eq_obj.type == "differential":
                dot_sym = [s for s in expr.free_symbols if str(s).endswith("_dot")][0]
                var_name = str(dot_sym).replace("_dot", "")

                term_B = expr.subs(dot_sym, 0)
                term_A = expr.diff(dot_sym)
                isolated_rhs = -term_B / term_A

                ca_func = sp.lambdify(
                    sympy_symbols_list, isolated_rhs, modules=[eval_dict, ca, "numpy"]
                )
                ca_val = ca_func(*casadi_symbols_list)

                if getattr(eq_obj, "is_distributed", False):
                    if hasattr(ca_val, "is_scalar") and ca_val.is_scalar():
                        ca_val = ca.repmat(
                            ca_val, self.model.variables[var_name].n_points, 1
                        )
                    ode_acc[var_name][eq_obj.flat_indices] = ca_val[eq_obj.flat_indices]
                else:
                    ode_lumped[var_name] = ca_val

            else:
                ca_func = sp.lambdify(
                    sympy_symbols_list, expr, modules=[eval_dict, ca, "numpy"]
                )
                ca_val = ca_func(*casadi_symbols_list)

                if getattr(eq_obj, "is_distributed", False):
                    if hasattr(ca_val, "is_scalar") and ca_val.is_scalar():
                        alg_acc.append(ca.repmat(ca_val, len(eq_obj.flat_indices), 1))
                    else:
                        alg_acc.append(ca_val[eq_obj.flat_indices])
                else:
                    alg_acc.append(ca_val)

        # Assemble strictly aligned DAE arrays
        ode_list = []
        for x_name in self.x_names_order:
            if x_name in ode_lumped:
                ode_list.append(ode_lumped[x_name])
            elif x_name in self.model.variables and getattr(
                self.model.variables[x_name], "is_distributed", False
            ):
                var_obj = self.model.variables[x_name]
                if var_obj.idx_diff:
                    ode_list.append(ode_acc[x_name][var_obj.idx_diff])

        self.dae_dict = {
            "x": ca.vertcat(*self.x_vars) if self.x_vars else ca.MX(),
            "z": ca.vertcat(*self.z_vars) if self.z_vars else ca.MX(),
            "p": ca.vertcat(*self.p_vars) if self.p_vars else ca.MX(),
            "ode": ca.vertcat(*ode_list) if ode_list else ca.MX(),
            "alg": ca.vertcat(*alg_acc) if alg_acc else ca.MX(),
        }
