# -*- coding: utf-8 -*-

"""
DaeAssembler Module (Legacy name: Transpiler) - V5 Native CasADi Architecture.

In the V5 Architecture, the concept of "transpilation" is technically obsolete.
Equations are already constructed as native CasADi MX graphs within the Model.
This module acts as an Assembler: it gathers the native graphs, maps topologies,
automatically isolates the Jacobian for ODEs, and concatenates the final DAE
system for the SUNDIALS integrators.
"""

import casadi as ca
import numpy as np

import antares.core.GLOBAL_CFG as cfg


class CasadiTranspiler:
    """
    System Assembler (Maintained as CasadiTranspiler for API backward compatibility).
    Assembles the pre-compiled CasADi graphs, applies topological slicing, and
    uses exact Jacobian linear algebra to isolate time derivatives.
    """

    def __init__(self, model):
        """
        Initializes the Assembler with the formulated mathematical model.

        :param Model model: The fully declared ANTARES Model.
        """
        self.model = model

        # CasADi core vectors for the solver
        self.x_vars = []
        self.z_vars = []
        self.p_vars = []

        # Order tracking to ensure strict block alignment
        self.x_names_order = []

        self.dae_dict = {}
        self.distributed_var_objects_list = []

    def transpile(self):
        """
        Executes the Native Graph Assembly Engine.

        :return: A dictionary containing the structured DAE system (x, z, p, ode, alg).
        :rtype: dict
        """
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(
                f"\n[ANTARES V5 ASSEMBLER] Compiling DAE System for '{self.model.name}'..."
            )

        self._map_and_assemble_topology()
        self._isolate_and_concatenate_graphs()

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"[ANTARES V5 ASSEMBLER] System closed and assembled successfully.\n")

        return self.dae_dict

    def _map_and_assemble_topology(self):
        """
        Generates the exact solver variables (x, z, p) based on topological rules.
        Builds the substitution mapping lists to bind the Model's monolithic
        variables into the partitioned DAE arrays.
        """
        self.old_syms = []
        self.new_syms = []

        # 1. Map Variables
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

                if idx_alg:
                    C_z = ca.MX.sym(var_name + "_z", len(idx_alg))
                    self.z_vars.append(C_z)
                    C_full[idx_alg] = C_z

                self.old_syms.append(var_obj.symbolic_object)
                self.new_syms.append(C_full)
                self.distributed_var_objects_list.append(var_obj)
            else:
                if var_obj.type == "differential":
                    ca_sym = ca.MX.sym(var_name + "_x")
                    self.x_vars.append(ca_sym)
                    self.x_names_order.append(var_name)
                else:
                    ca_sym = ca.MX.sym(var_name + "_z")
                    self.z_vars.append(ca_sym)

                self.old_syms.append(var_obj.symbolic_object)
                self.new_syms.append(ca_sym)

        # 2. Map Parameters
        for par_name, par_obj in self.model.parameters.items():
            p_sym = ca.MX.sym(par_name)
            self.p_vars.append(p_sym)

            if hasattr(par_obj, "symbolic_object"):
                self.old_syms.append(par_obj.symbolic_object)
                self.new_syms.append(p_sym)

        # 3. Map Constants
        for const_name, const_obj in self.model.constants.items():
            if hasattr(const_obj, "symbolic_object"):
                self.old_syms.append(const_obj.symbolic_object)
                self.new_syms.append(ca.MX(const_obj.value))

    def _isolate_and_concatenate_graphs(self):
        """
        Processes the native CasADi equations.
        Applies exact Jacobian algebra (M * x_dot = -F) to isolate time
        derivatives automatically, ensuring C++ signature compliance during substitution.
        """
        ode_acc = {
            v.name: ca.MX.zeros(v.n_points) for v in self.distributed_var_objects_list
        }
        ode_lumped = {}
        alg_acc = []

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(" -> Applying Native Jacobian Isolation and Concatenation...")

        for eq_name, eq_obj in self.model.equations.items():
            expr = eq_obj.equation_expression.symbolic_object

            # FIX: Bind the native expression in a list to match the C++ signature: ([MX], [MX], [MX])
            expr_sub = ca.substitute([expr], self.old_syms, self.new_syms)[0]

            if eq_obj.type == "differential":
                all_syms = ca.symvar(expr_sub)
                dot_syms = [s for s in all_syms if s.name().endswith("_dot")]

                if not dot_syms:
                    raise RuntimeError(
                        f"Equation '{eq_name}' is differential but lacks a temporal derivative (_dot)."
                    )

                dot_sym = dot_syms[0]
                var_name = dot_sym.name().replace("_dot", "")

                # FIX: Consistent list encapsulation for Jacobian zero-state evaluation
                F = ca.substitute(
                    [expr_sub], [dot_sym], [ca.MX.zeros(dot_sym.size1(), 1)]
                )[0]
                M = ca.jacobian(expr_sub, dot_sym)

                isolated_rhs = ca.solve(M, -F)

                if getattr(eq_obj, "is_distributed", False):
                    ode_acc[var_name][eq_obj.flat_indices] = isolated_rhs[
                        eq_obj.flat_indices
                    ]
                else:
                    ode_lumped[var_name] = isolated_rhs

            else:
                if getattr(eq_obj, "is_distributed", False):
                    alg_acc.append(expr_sub[eq_obj.flat_indices])
                else:
                    alg_acc.append(expr_sub)

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
