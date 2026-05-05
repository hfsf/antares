# -*- coding: utf-8 -*-

r"""
DaeAssembler Module (Legacy name: Transpiler) - V5 Native CasADi Architecture.

In the V5 Architecture, the concept of "transpilation" is technically obsolete.
Equations are already constructed as native CasADi MX graphs within the Model.
This module acts as the Root Assembler: it gathers the native graphs, maps topologies,
automatically isolates the Jacobian for ODEs, and crucially, acts as the 
**Numerical Scaling Guardian** by injecting residual conditioning factors (e.g., $10^{-5}$) 
to prevent singular Jacobians without polluting the user's physical abstractions.
"""

import casadi as ca
import numpy as np

import antares.core.GLOBAL_CFG as cfg


class CasadiTranspiler:
    r"""
    System Assembler Engine (Maintained as CasadiTranspiler for API backward compatibility).
    
    Assembles the pre-compiled CasADi graphs, applies topological slicing, 
    isolates temporal derivatives via exact Jacobian linear algebra, and 
    silently enforces residual scaling on ill-conditioned thermodynamic states.
    """

    def __init__(self, model):
        self.model = model
        self.x_vars = []
        self.z_vars = []
        self.p_vars = []
        self.x_names_order = []
        self.dae_dict = {}
        self.distributed_var_objects_list = []

    def transpile(self):
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"\n[ANTARES V5 ASSEMBLER] Compiling DAE System for '{self.model.name}'...")

        self._map_and_assemble_topology()
        self._isolate_and_concatenate_graphs()

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"[ANTARES V5 ASSEMBLER] System closed and assembled successfully.\n")

        return self.dae_dict

    def _map_and_assemble_topology(self):
        self.old_syms = []
        self.new_syms = []

        for var_name, var_obj in self.model.variables.items():
            if getattr(var_obj, "is_distributed", False):
                n_points = var_obj.n_points
                idx_diff = np.where(var_obj.node_types.flatten() == "differential")[0].tolist()
                idx_alg = np.where(var_obj.node_types.flatten() == "algebraic")[0].tolist()

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

        for par_name, par_obj in self.model.parameters.items():
            p_sym = ca.MX.sym(par_name)
            self.p_vars.append(p_sym)
            if hasattr(par_obj, "symbolic_object"):
                self.old_syms.append(par_obj.symbolic_object)
                self.new_syms.append(p_sym)

        for const_name, const_obj in self.model.constants.items():
            if hasattr(const_obj, "symbolic_object"):
                self.old_syms.append(const_obj.symbolic_object)
                self.new_syms.append(ca.MX(const_obj.value))

    def _determine_residual_scale(self, eq_name, expr):
        eq_lower = str(eq_name).lower()
        if "enthalpy" in eq_lower or "drum_energy" in eq_lower:
            return 1e5
        if "energy_flow" in eq_lower or "eq_energy" in eq_lower or "duty" in eq_lower:
            return 1e5
        if "pressure_pa" in eq_lower:
            return 1e5
        return 1.0

    def _isolate_and_concatenate_graphs(self):
        r"""
        Processes the native CasADi equations and applies automated scaling.
        Applies exact Jacobian algebra ($M \cdot \dot{x} = -F$) to isolate time
        derivatives automatically.
        """
        ode_acc = {v.name: ca.MX.zeros(v.n_points) for v in self.distributed_var_objects_list}
        ode_lumped = {}
        alg_acc = []

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(" -> Applying Native Jacobian Isolation, Residual Scaling, and Concatenation...")

        for eq_name, eq_obj in self.model.equations.items():
            expr = eq_obj.equation_expression.symbolic_object
            expr_sub = ca.substitute([expr], self.old_syms, self.new_syms)[0]

            if eq_obj.type == "differential":
                all_syms = ca.symvar(expr_sub)
                dot_syms = [s for s in all_syms if s.name().endswith("_dot")]

                if not dot_syms:
                    raise RuntimeError(f"Equation '{eq_name}' is differential but lacks a temporal derivative (_dot).")

                dot_sym = dot_syms[0]
                var_name = dot_sym.name().replace("_dot", "")
                F = ca.substitute([expr_sub], [dot_sym], [ca.MX.zeros(dot_sym.size1(), 1)])[0]
                M = ca.jacobian(expr_sub, dot_sym)

                isolated_rhs = ca.solve(M, -F)

                if getattr(eq_obj, "is_distributed", False):
                    ode_acc[var_name][eq_obj.flat_indices] = isolated_rhs[eq_obj.flat_indices]
                else:
                    ode_lumped[var_name] = isolated_rhs

            else:
                scale_factor = self._determine_residual_scale(eq_name, expr_sub)
                scaled_expr = expr_sub / scale_factor

                if getattr(eq_obj, "is_distributed", False):
                    alg_acc.append(scaled_expr[eq_obj.flat_indices])
                else:
                    alg_acc.append(scaled_expr)

        ode_list = []
        for x_name in self.x_names_order:
            if x_name in ode_lumped:
                ode_list.append(ode_lumped[x_name])
            elif x_name in self.model.variables and getattr(self.model.variables[x_name], "is_distributed", False):
                var_obj = self.model.variables[x_name]
                if var_obj.idx_diff:
                    ode_list.append(ode_acc[x_name][var_obj.idx_diff])

        if "alg_names" not in self.dae_dict:
            self.dae_dict["alg_names"] = []
        for eq_name, eq_obj in self.model.equations.items():
            if eq_obj.type != "differential":
                self.dae_dict["alg_names"].append(eq_name)

        self.dae_dict.update({
            "x": ca.vertcat(*self.x_vars) if self.x_vars else ca.MX(),
            "z": ca.vertcat(*self.z_vars) if self.z_vars else ca.MX(),
            "p": ca.vertcat(*self.p_vars) if self.p_vars else ca.MX(),
            "ode": ca.vertcat(*ode_list) if ode_list else ca.MX(),
            "alg": ca.vertcat(*alg_acc) if alg_acc else ca.MX(),
        })