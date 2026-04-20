# -*- coding: utf-8 -*-

"""
Simulator Module.

Defines the Simulator class.
Receives the phenomenological model, calls the CasadiTranspiler, and solves
the Differential-Algebraic Equation (DAE) system over time using CasADi's
native integrators. Optimized with Lazy Compilation, Auto-Fallback rules,
and Global Configurations.
"""

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
    Orchestrates the numerical integration of the mathematical model.
    Bridges the declarative Model abstractions with the high-performance 
    CasADi backend.
    """

    def __init__(self, model, solver_type=None, check_dof=None):
        """
        Instantiates the Simulator. Prepares the simulation by transpiling the model
        and setting up the ground for the CasADi solver.

        :param Model model: The model object to be simulated. Must be fully declared.
        :param str solver_type: The type of CasADi integrator to use. If None, falls 
                                back to the global DEFAULT_INTEGRATOR (usually 'idas').
        :param bool check_dof: Flag to enable/disable DOF validation at compile time. 
                               If None, delegates to GLOBAL_CFG.PERFORM_DOF_CHECK.
        """
        self.model = model

        # Applies global configuration if no specific solver is requested
        self.solver_type = (
            solver_type if solver_type is not None else getattr(cfg, "DEFAULT_INTEGRATOR", "idas")
        )

        # Checking if the solver type is within the standard expected ones
        if self.solver_type not in ["idas", "cvodes", "collocation", "rk"]:
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                warnings.warn(
                    f"Solver type '{self.solver_type}' might not be natively supported. "
                    f"Proceed with caution."
                )

        self.transpiler = CasadiTranspiler(model)

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"Transpiling model '{self.model.name}' to CasADi representation...")

        # 1. Transpile the mathematical AST into CasADi graph
        self.dae_structure = self.transpiler.transpile()

        # 2. VERIFICATION OF DEGREES OF FREEDOM (Fail-Fast: Compile time)
        do_dof_check = check_dof if check_dof is not None else getattr(cfg, "PERFORM_DOF_CHECK", True)
        if do_dof_check:
            self._check_degrees_of_freedom()

        # Integrator options injected from GLOBAL_CFG
        self.solver_opts = {
            "calc_ic": True,  # Automatically calculates consistent initial conditions
            "calc_icB": False,
            "reltol": getattr(cfg, "DEFAULT_RELATIVE_TOLERANCE", 1e-6),
            "abstol": getattr(cfg, "DEFAULT_ABSOLUTE_TOLERANCE", 1e-8),
        }

        # Attributes for Lazy Compilation (Performance optimization)
        self._integrator = None
        self._last_t_span = None

    def _check_degrees_of_freedom(self):
        """
        Validates the mathematical closure of the system by calculating 
        the Degrees of Freedom (DOF) of the transpiled computational graph.
        Prints a compilation report if verbosity allows.

        :raises DegreesOfFreedomError: If total variables and total equations do not match.
        """
        # Count explicit transpiled variables
        n_x = self.dae_structure["x"].size1() if "x" in self.dae_structure else 0
        n_z = self.dae_structure["z"].size1() if "z" in self.dae_structure else 0
        total_vars = n_x + n_z

        # Count explicit transpiled equations
        n_ode = self.dae_structure["ode"].size1() if "ode" in self.dae_structure else 0
        n_alg = self.dae_structure["alg"].size1() if "alg" in self.dae_structure else 0
        total_eqs = n_ode + n_alg

        dof = total_vars - total_eqs

        # ALWAYS PRINT THE COMPILATION REPORT (If verbosity is active)
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"\n[{self.model.name} COMPILATION REPORT]")
            print(f"  -> Total Variables: {total_vars} (Differential: {n_x}, Algebraic: {n_z})")
            print(f"  -> Total Equations: {total_eqs} (ODE: {n_ode}, Algebraic: {n_alg})")
            print(f"  -> Degrees of Freedom: {dof}")
            if dof == 0:
                print("  -> Status: System is mathematically closed. OK!\n")

        # THROW ERROR IF SYSTEM IS OPEN
        if dof != 0:
            msg = (
                f"\n[ANTARES MATHEMATICAL KERNEL] Degrees of Freedom Violation!\n"
                f"The transpiled model '{self.model.name}' is not mathematically closed.\n\n"
                f"  -> Total Variables: {total_vars} (Differential: {n_x}, Algebraic: {n_z})\n"
                f"  -> Total Equations: {total_eqs} (ODE: {n_ode}, Algebraic: {n_alg})\n"
                f"  -> DOF (Vars - Eqs): {dof}\n\n"
            )
            if dof > 0:
                msg += f"SOLUTION: The system is underspecified. You are missing {dof} equation(s) or Boundary Condition(s)."
            else:
                msg += f"SOLUTION: The system is overspecified. You have {abs(dof)} redundant or conflicting equation(s)."
            
            raise DegreesOfFreedomError(msg)

    def _compile_integrator(self, t_span, use_c_code=False):
        """
        Compiles the CasADi JIT integrator. If use_c_code is True, invokes GCC/Clang 
        to compile the DAE system into native C machine code. Includes an automatic 
        fallback to the CasADi Virtual Machine if the user lacks a C compiler.

        :param array-like t_span: 1D array containing the time grid for the simulation.
        :param bool use_c_code: Internal flag resolving the hierarchy of C code compilation.
        """
        if self._integrator is None or not np.array_equal(t_span, self._last_t_span):
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
                print("Compiling CasADi integrator (This occurs only once per time grid)...")

            opts = self.solver_opts.copy()

            # Remove deprecated flags strictly to prevent CasADi warnings
            opts.pop("grid", None)
            opts.pop("output_t0", None)

            t0 = t_span[0]

            # -----------------------------------------------------------------
            # COMPILATION CORE (C-CODE vs VIRTUAL MACHINE)
            # -----------------------------------------------------------------
            if use_c_code:
                if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                    print(f"[{self.model.name}] Generating and compiling pure C code for extreme performance.")
                
                # Activate Just-In-Time C Code Compilation
                opts["jit"] = True
                opts["compiler"] = "shell"
                if getattr(cfg, "C_COMPILATION_OPTIMIZATION_LEVEL", "basic"):
                    opts["jit_options"] = {"flags": ["-O1", "-pipe"]} # Basic optimization
                elif getattr(cfg, "C_COMPILATION_OPTIMIZATION_LEVEL", "aggressive"):
                    opts["jit_options"] = {"flags": ["-O3", "-ffast-math"]}  # Aggressive optimization
                else:
                    if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
                        print("[WARNING] Optimization level of C code for compilation not declared. Fall back to 'basic'")
                        opts["jit_options"] = {"flags": ["-O1", "-pipe"]} # Basic optimization                
                try:
                    self._integrator = ca.integrator(
                        "sim_integrator", self.solver_type, self.dae_structure, t0, t_span, opts
                    )
                except Exception as e:
                    # FALLBACK SAFETY NET: If GCC is not found in PATH, downgrade gracefully
                    if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                        warnings.warn(
                            "Native C code compilation failed (Is a C compiler like GCC installed "
                            "and added to PATH?). Falling back to the CasADi Virtual Machine."
                        )
                    opts["jit"] = False
                    self._integrator = ca.integrator(
                        "sim_integrator", self.solver_type, self.dae_structure, t0, t_span, opts
                    )
            else:
                # Standard Virtual Machine compilation
                self._integrator = ca.integrator(
                    "sim_integrator", self.solver_type, self.dae_structure, t0, t_span, opts
                )

            self._last_t_span = t_span

    def run(self, t_span, initial_conditions=None, parameters_dict=None, use_c_code=None):
        """
        Executes the simulation. 

        By default, it uses the Initial Conditions and Parameters defined internally 
        in the Model object. The optional dictionaries serve as runtime overrides.

        :param array-like t_span: Time grid for the simulation.
        :param dict initial_conditions: Runtime override for variables ICs. Defaults to None.
        :param dict parameters_dict: Runtime override for parameters. Defaults to None.
        :param bool use_c_code: Runtime override to force native C compilation. 
                                If None, delegates to GLOBAL_CFG.USE_C_CODE_COMPILATION.
        :return: Results object containing the structured data of the simulation.
        :rtype: Results
        """
        # Resolve compilation hierarchy (Parameter > Global Config > Default)
        do_c_code = use_c_code if use_c_code is not None else getattr(cfg, "USE_C_CODE_COMPILATION", False)

        # Ensure arguments are dictionaries, even if None is passed
        ic_dict = initial_conditions if initial_conditions is not None else {}
        p_dict = parameters_dict if parameters_dict is not None else {}

        # 1. Smart Compilation (Lazy Evaluation + C-Code Generation)
        self._compile_integrator(t_span, use_c_code=do_c_code)

        # 2. Extract input vectors (Resolving Encapsulation vs Override logic)
        x0_vec = self._build_input_vector(
            ic_dict,
            self.transpiler.x_vars,
            self.model.variables,
            category="differential",
        )
        z0_vec = self._build_input_vector(
            ic_dict, 
            self.transpiler.z_vars, 
            self.model.variables, 
            category="algebraic"
        )
        p_vec = self._build_input_vector(
            p_dict, 
            self.transpiler.p_vars, 
            self.model.parameters, 
            category="parameter"
        )

        # 3. Fast native execution
        sim_args = {"x0": x0_vec, "p": p_vec}
        if self.transpiler.z_vars:
            sim_args["z0"] = z0_vec

        res = self._integrator(**sim_args)

        # 4. Results processing
        x_res = res["xf"].full().T
        z_res = res["zf"].full().T if self.transpiler.z_vars else None

        # 5. Packaging into Results container
        results_container = Results(
            name=f"Sim_{self.model.name}", 
            time_units=getattr(cfg, "DEFAULT_TIME_UNIT", "s")
        )

        x_names = [var.name() for var in self.transpiler.x_vars]
        z_names = [var.name() for var in self.transpiler.z_vars]

        results_container.load_from_simulator(t_span, x_res, z_res, x_names, z_names)

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print("Simulation successfully completed!")

        return results_container

    def _build_input_vector(self, input_dict, ca_vars_list, model_dict, category="algebraic"):
        """
        Builds the input numerical vector expected by CasADi, resolving the 
        hierarchy between runtime overrides and model-encapsulated definitions.
        """
        vec = []
        for ca_var in ca_vars_list:
            var_name = ca_var.name()

            # Priority 1: User runtime override (passed via the run() method)
            if var_name in input_dict:
                vec.append(input_dict[var_name])

            # Priority 2: Model Encapsulated Initial Conditions
            elif category == "differential":
                if var_name in model_dict and getattr(model_dict[var_name], "is_specified", False):
                    vec.append(model_dict[var_name].value)
                else:
                    raise AbsentRequiredObjectError(
                        f"The differential variable '{var_name}' requires an explicit "
                        f"Initial Condition. Please use 'self.setInitialCondition()' "
                        f"in the Model declaration, or pass it via the 'initial_conditions' "
                        f"dictionary in the run() method."
                    )

            # Priority 3: Parameters or Algebraic variable guesses
            elif var_name in model_dict:
                vec.append(model_dict[var_name].value)

            else:
                # Priority 4: Final fallback
                if category == "algebraic":
                    if getattr(cfg, "STRICT_MODE", False):
                        raise AbsentRequiredObjectError(
                            f"[STRICT MODE] Missing initial guess for algebraic variable '{var_name}'."
                        )
                    vec.append(0.0)
                else:
                    raise UnexpectedValueError(
                        f"Unable to resolve a numerical value for {category}: '{var_name}'"
                    )

        return vec