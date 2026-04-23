# -*- coding: utf-8 -*-

"""
Simulator Module.

Defines the Simulator class.
Receives the phenomenological model, calls the CasadiTranspiler, and solves
the Differential-Algebraic Equation (DAE) system over time using CasADi's
native integrators, or resolves steady-state conditions using rootfinders.
V4 UPDATE: Fully integrated with Block Vectorization arrays and Smart Solver Mapping.
"""

import glob
import os
import warnings

import casadi as ca
import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.transpiler import CasadiTranspiler
from antares.core.error_definitions import (
    AbsentRequiredObjectError,
    DegreesOfFreedomError,
    UnexpectedValueError,
)
from antares.core.results import Results


class Simulator:
    """
    Orchestrates the numerical resolution of the mathematical model.
    Bridges the declarative Model abstractions with the high-performance
    CasADi backend. Supports both dynamic integration and steady-state.
    """

    def __init__(self, model, solver_type=None, check_dof=None):
        self.model = model
        self.solver_type = (
            solver_type
            if solver_type is not None
            else getattr(cfg, "DEFAULT_INTEGRATOR", "idas")
        )

        self.transpiler = CasadiTranspiler(model)
        self.dae_structure = self.transpiler.transpile()

        do_dof_check = (
            check_dof
            if check_dof is not None
            else getattr(cfg, "PERFORM_DOF_CHECK", True)
        )
        if do_dof_check:
            self._check_degrees_of_freedom()

        self.solver_opts = {
            "calc_ic": True,
            "reltol": getattr(cfg, "DEFAULT_RELATIVE_TOLERANCE", 1e-6),
            "abstol": getattr(cfg, "DEFAULT_ABSOLUTE_TOLERANCE", 1e-8),
        }

        self._integrator = None
        self._rootfinder = None
        self._last_t_span = None
        self._compilation_prefixes = ["jit_", "tmp_casadi_"]

    def _cleanup_compilation_files(self):
        keep_files = getattr(cfg, "KEEP_TEMPORARY_COMPILATION_FILES", False)
        if keep_files:
            return

        extensions = ["*.c", "*.o", "*.so", "*.dll"]
        for prefix in self._compilation_prefixes:
            for ext in extensions:
                for filepath in glob.glob(f"{prefix}{ext}"):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass

    def _check_degrees_of_freedom(self):
        n_x = self.dae_structure["x"].size1() if "x" in self.dae_structure else 0
        n_z = self.dae_structure["z"].size1() if "z" in self.dae_structure else 0
        total_vars = n_x + n_z

        n_ode = self.dae_structure["ode"].size1() if "ode" in self.dae_structure else 0
        n_alg = self.dae_structure["alg"].size1() if "alg" in self.dae_structure else 0
        total_eqs = n_ode + n_alg

        dof = total_vars - total_eqs

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"\n[{self.model.name} COMPILATION REPORT]")
            print(
                f"  -> Total Variables: {total_vars} (Differential: {n_x}, Algebraic: {n_z})"
            )
            print(
                f"  -> Total Equations: {total_eqs} (ODE: {n_ode}, Algebraic: {n_alg})"
            )
            print(f"  -> Degrees of Freedom: {dof}")
            if dof == 0:
                print("  -> Status: System is mathematically closed. OK!\n")

        if dof != 0:
            msg = (
                f"\n[ANTARES MATHEMATICAL KERNEL] Degrees of Freedom Violation!\n"
                f"The transpiled model '{self.model.name}' is not mathematically closed.\n\n"
                f"  -> Total Variables: {total_vars}\n"
                f"  -> Total Equations: {total_eqs}\n"
                f"  -> DOF (Vars - Eqs): {dof}\n\n"
            )
            raise DegreesOfFreedomError(msg)

    def _resolve_linear_solver(self, requested_solver):
        """
        Translates generic ANTARES intents ('direct', 'iterative') into
        exact CasADi C++ plugin names, keeping the framework generalist.
        """
        solver_map = {"direct": "csparse", "iterative": "gmres"}
        return solver_map.get(requested_solver.lower(), requested_solver.lower())

    def _compile_integrator(self, t_span, use_c_code=False, linear_solver="direct"):
        if self._integrator is None or not np.array_equal(t_span, self._last_t_span):
            opts = self.solver_opts.copy()
            t0 = t_span[0]

            # Aplica o Mapeamento Inteligente
            opts["linear_solver"] = self._resolve_linear_solver(linear_solver)

            if use_c_code:
                opts["jit"] = True
                opts["compiler"] = "shell"
                if (
                    getattr(cfg, "C_COMPILATION_OPTIMIZATION_LEVEL", "basic")
                    == "aggressive"
                ):
                    opts["jit_options"] = {"flags": ["-O3", "-ffast-math"]}
                else:
                    opts["jit_options"] = {"flags": ["-O1", "-pipe"]}
                try:
                    self._integrator = ca.integrator(
                        self._compilation_prefixes[0],
                        self.solver_type,
                        self.dae_structure,
                        t0,
                        t_span,
                        opts,
                    )
                except Exception:
                    opts["jit"] = False
                    self._integrator = ca.integrator(
                        self._compilation_prefixes[0],
                        self.solver_type,
                        self.dae_structure,
                        t0,
                        t_span,
                        opts,
                    )
            else:
                self._integrator = ca.integrator(
                    self._compilation_prefixes[0],
                    self.solver_type,
                    self.dae_structure,
                    t0,
                    t_span,
                    opts,
                )

            self._last_t_span = t_span

    def _compile_rootfinder(self, use_c_code=False, linear_solver="direct"):
        if self._rootfinder is None:
            dae = self.dae_structure

            v_vars = []
            if "x" in dae and dae["x"].size1() > 0:
                v_vars.append(dae["x"])
            if "z" in dae and dae["z"].size1() > 0:
                v_vars.append(dae["z"])
            v = ca.vertcat(*v_vars) if v_vars else ca.MX()

            eq_vars = []
            if "ode" in dae and dae["ode"].size1() > 0:
                eq_vars.append(dae["ode"])
            if "alg" in dae and dae["alg"].size1() > 0:
                eq_vars.append(dae["alg"])
            eqs = ca.vertcat(*eq_vars) if eq_vars else ca.MX()

            p = dae["p"] if "p" in dae and dae["p"].size1() > 0 else ca.MX()
            problem = {"x": v, "p": p, "g": eqs}

            opts = {"abstol": getattr(cfg, "DEFAULT_ABSOLUTE_TOLERANCE", 1e-8)}

            # Aplica o Mapeamento Inteligente
            opts["linear_solver"] = self._resolve_linear_solver(linear_solver)

            if use_c_code:
                opts["jit"] = True
                opts["compiler"] = "shell"
                opts["jit_options"] = {"flags": ["-O1", "-pipe"]}
                try:
                    self._rootfinder = ca.rootfinder(
                        self._compilation_prefixes[1], "kinsol", problem, opts
                    )
                except Exception:
                    opts["jit"] = False
                    self._rootfinder = ca.rootfinder(
                        self._compilation_prefixes[1], "kinsol", problem, opts
                    )
            else:
                self._rootfinder = ca.rootfinder(
                    self._compilation_prefixes[1], "kinsol", problem, opts
                )

    def _get_initial_vector(self, input_dict, category="differential"):
        """Extracts the numerical Initial Conditions natively mapping the Vector Blocks."""
        vec = []
        for var_name, var_obj in self.model.variables.items():
            if getattr(var_obj, "is_distributed", False):
                ic_arr = var_obj.initial_condition_array
                if category == "differential" and var_obj.idx_diff:
                    vec.extend(ic_arr[var_obj.idx_diff].tolist())
                elif category == "algebraic" and var_obj.idx_alg:
                    vec.extend(ic_arr[var_obj.idx_alg].tolist())
            else:
                if var_obj.type == category:
                    if var_name in input_dict:
                        vec.append(input_dict[var_name])
                    elif category == "differential":
                        if getattr(var_obj, "is_specified", False):
                            vec.append(var_obj.value)
                        else:
                            raise AbsentRequiredObjectError(
                                f"Differential variable '{var_name}' requires an Initial Condition."
                            )
                    else:
                        vec.append(var_obj.value)
        return vec

    def _get_parameter_vector(self, p_dict):
        """Extracts parameters strictly preserving the transpiler order."""
        vec = []
        for par_name, par_obj in self.model.parameters.items():
            if par_name in p_dict:
                vec.append(p_dict[par_name])
            else:
                vec.append(par_obj.value)
        return vec

    def _generate_names_list(self):
        """Flattens the multi-dimensional coordinates back into explicit string names."""
        x_names, z_names = [], []
        for var_name, var_obj in self.model.variables.items():
            if getattr(var_obj, "is_distributed", False):
                for i in range(var_obj.n_points):
                    idx_tuple = np.unravel_index(i, var_obj.tensor_shape)
                    idx_str = "_".join(map(str, idx_tuple))
                    node_name = f"{var_name}_{var_obj.domain.name}_{idx_str}"
                    if i in var_obj.idx_diff:
                        x_names.append(node_name)
                    if i in var_obj.idx_alg:
                        z_names.append(node_name)
            else:
                if var_obj.type == "differential":
                    x_names.append(var_name)
                else:
                    z_names.append(var_name)
        return x_names, z_names

    def run(
        self,
        t_span,
        initial_conditions=None,
        parameters_dict=None,
        use_c_code=None,
        linear_solver=None,
    ):
        do_c_code = (
            use_c_code
            if use_c_code is not None
            else getattr(cfg, "USE_C_CODE_COMPILATION", False)
        )
        do_lin_sol = (
            linear_solver
            if linear_solver is not None
            else getattr(cfg, "DEFAULT_LINEAR_SOLVER", "direct")
        )

        ic_dict = initial_conditions if initial_conditions is not None else {}
        p_dict = parameters_dict if parameters_dict is not None else {}

        self._compile_integrator(t_span, use_c_code=do_c_code, linear_solver=do_lin_sol)

        x0_vec = self._get_initial_vector(ic_dict, category="differential")
        z0_vec = self._get_initial_vector(ic_dict, category="algebraic")
        p_vec = self._get_parameter_vector(p_dict)

        sim_args = {"x0": x0_vec, "p": p_vec}
        if z0_vec:
            sim_args["z0"] = z0_vec

        res = self._integrator(**sim_args)

        x_res = res["xf"].full().T
        z_res = res["zf"].full().T if z0_vec else None

        results_container = Results(
            name=f"Sim_{self.model.name}",
            time_units=getattr(cfg, "DEFAULT_TIME_UNIT", "s"),
        )
        x_names, z_names = self._generate_names_list()

        results_container.load_from_simulator(t_span, x_res, z_res, x_names, z_names)

        if do_c_code:
            self._cleanup_compilation_files()
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print("Dynamic Simulation successfully completed!")
        return results_container

    def run_steady_state(
        self,
        initial_guesses=None,
        parameters_dict=None,
        use_c_code=None,
        linear_solver=None,
    ):
        do_c_code = (
            use_c_code
            if use_c_code is not None
            else getattr(cfg, "USE_C_CODE_COMPILATION", False)
        )
        do_lin_sol = (
            linear_solver
            if linear_solver is not None
            else getattr(cfg, "DEFAULT_LINEAR_SOLVER", "direct")
        )

        ig_dict = initial_guesses if initial_guesses is not None else {}
        p_dict = parameters_dict if parameters_dict is not None else {}

        self._compile_rootfinder(use_c_code=do_c_code, linear_solver=do_lin_sol)

        x0_vec = self._get_initial_vector(ig_dict, category="differential")
        z0_vec = self._get_initial_vector(ig_dict, category="algebraic")

        v0_vec = []
        if x0_vec:
            v0_vec.extend(x0_vec)
        if z0_vec:
            v0_vec.extend(z0_vec)

        p_vec = self._get_parameter_vector(p_dict)

        res = self._rootfinder(x0=v0_vec, p=p_vec)

        v_sol = res["x"].full().flatten()
        n_x = len(x0_vec)
        x_sol = v_sol[:n_x]
        z_sol = v_sol[n_x:]

        results_container = Results(
            name=f"Steady_{self.model.name}",
            time_units=getattr(cfg, "DEFAULT_TIME_UNIT", "s"),
        )
        x_names, z_names = self._generate_names_list()

        t_ss = np.array([0.0])
        x_res = np.array([x_sol]) if n_x > 0 else np.array([[]])
        z_res = np.array([z_sol]) if len(z_sol) > 0 else None

        results_container.load_from_simulator(t_ss, x_res, z_res, x_names, z_names)

        if do_c_code:
            self._cleanup_compilation_files()
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print("Steady-State solution successfully found!")
        return results_container
