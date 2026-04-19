# *coding:utf-8*

"""
Define Simulator class.
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
    UnexpectedValueError,
)
from antares.core.results import Results


class Simulator:
    """
    Definition of Simulator class. Orchestrates the integration of the mathematical model.
    """

    def __init__(self, model, solver_type=None):
        """
        Instantiate Simulator. Prepares the simulation by transpiling the model
        and setting up the ground for the CasADi solver.

        :param Model model:
            The model object to be simulated. Must be fully declared.

        :param str solver_type:
            The type of CasADi integrator to use. If None, falls back to the
            global DEFAULT_INTEGRATOR (usually 'idas').
        """
        self.model = model

        # Applies global configuration if no specific solver is requested
        self.solver_type = (
            solver_type if solver_type is not None else cfg.DEFAULT_INTEGRATOR
        )

        # Checking if the solver type is within the standard expected ones
        if self.solver_type not in ["idas", "cvodes", "collocation", "rk"]:
            if cfg.VERBOSITY_LEVEL >= 1:
                warnings.warn(
                    f"Solver type '{self.solver_type}' might not be fully supported. Proceed with caution."
                )

        self.transpiler = CasadiTranspiler(model)

        if cfg.VERBOSITY_LEVEL >= 1:
            print(f"Transpiling model '{self.model.name}' to CasADi representation...")

        self.dae_structure = self.transpiler.transpile()

        # Integrator options injected from GLOBAL_CFG
        self.solver_opts = {
            "calc_ic": True,  # Automatically calculates consistent initial conditions (useful for DAEs)
            "calc_icB": False,
            "reltol": cfg.DEFAULT_RELATIVE_TOLERANCE,
            "abstol": cfg.DEFAULT_ABSOLUTE_TOLERANCE,
        }

        # Attributes for Lazy Compilation (Performance optimization)
        self._integrator = None
        self._last_t_span = None

    def _compile_integrator(self, t_span):
        """
        Compiles the CasADi JIT integrator only if it's the first run or if
        the time grid has changed. This avoids massive performance drops in optimization loops.

        :param array-like t_span:
            1D array containing the time grid for the simulation.
        """
        # Check if arrays are exactly equal to avoid useless recompilation
        if self._integrator is None or not np.array_equal(t_span, self._last_t_span):
            if cfg.VERBOSITY_LEVEL >= 2:
                print(
                    "Compiling CasADi JIT integrator (This occurs only once per time grid)..."
                )

            opts = self.solver_opts.copy()

            # Removemos antigas flags do dicionário (para limpar os warnings)
            opts.pop("grid", None)
            opts.pop("output_t0", None)

            t0 = t_span[0]

            # NOVO PADRÃO CASADI: Passamos t0 e t_span como argumentos posicionais diretamente
            self._integrator = ca.integrator(
                "sim_integrator", self.solver_type, self.dae_structure, t0, t_span, opts
            )

            self._last_t_span = t_span

    def run(self, t_span, initial_conditions=None, parameters_dict=None):
        """
        Executes the simulation. If arguments are omitted, it uses the default
        values defined in the Model, except for differential variables which
        strictly require explicit initial conditions.

        :param array-like t_span:
            Time grid for the simulation (e.g., np.linspace(0, 10, 100)).

        :param dict initial_conditions:
            Dictionary containing initial conditions for the variables.
            Format: {'variable_name': value}. Defaults to None.

        :param dict parameters_dict:
            Dictionary containing values for the parameters.
            Format: {'parameter_name': value}. Defaults to None.

        :return results_container:
            Results object containing the structured data of the simulation.
        :rtype Results:
        """
        # Ensure arguments are dictionaries, even if None is passed
        ic_dict = initial_conditions if initial_conditions is not None else {}
        p_dict = parameters_dict if parameters_dict is not None else {}

        # 1. Smart Compilation (Lazy)
        self._compile_integrator(t_span)

        # 2. Extract input vectors (With category-based safety rules)
        x0_vec = self._build_input_vector(
            ic_dict,
            self.transpiler.x_vars,
            self.model.variables,
            category="differential",
        )
        z0_vec = self._build_input_vector(
            ic_dict, self.transpiler.z_vars, self.model.variables, category="algebraic"
        )
        p_vec = self._build_input_vector(
            p_dict, self.transpiler.p_vars, self.model.parameters, category="parameter"
        )

        # 3. Fast native execution
        sim_args = {"x0": x0_vec, "p": p_vec}
        if self.transpiler.z_vars:
            sim_args["z0"] = z0_vec

        res = self._integrator(**sim_args)

        # 4. Results processing
        x_res = res["xf"].full().T
        z_res = res["zf"].full().T if self.transpiler.z_vars else None

        # 5. Packaging into Results container (Phase 3)
        results_container = Results(
            name=f"Sim_{self.model.name}", time_units=cfg.DEFAULT_TIME_UNIT
        )

        # Extract original names for the DataFrame columns
        x_names = [var.name() for var in self.transpiler.x_vars]
        z_names = [var.name() for var in self.transpiler.z_vars]

        results_container.load_from_simulator(t_span, x_res, z_res, x_names, z_names)

        if cfg.VERBOSITY_LEVEL >= 1:
            print("Simulation successfully completed!")

        return results_container

    def _build_input_vector(
        self, input_dict, ca_vars_list, model_dict, category="algebraic"
    ):
        """
        Builds the input vector applying strict safety rules depending on
        the variable type (differential, algebraic, or parameter).

        :param dict input_dict:
            Dictionary provided by the user in the run() method.
        :param list ca_vars_list:
            List of CasADi symbolic variables from the transpiler.
        :param dict model_dict:
            Dictionary of objects (Variables or Parameters) from the Model.
        :param str category:
            Category of the variables ("differential", "algebraic", or "parameter").

        :return vec_:
            List of numerical values ordered exactly as CasADi expects.
        :rtype list:

        :raises AbsentRequiredObjectError:
            If a differential variable is missing its initial condition.
        :raises UnexpectedValueError:
            If a parameter or algebraic variable cannot be resolved.
        """
        vec = []
        for ca_var in ca_vars_list:
            var_name = ca_var.name()

            # 1st Priority: The user explicitly provided it in the run() call
            if var_name in input_dict:
                vec.append(input_dict[var_name])

            # 2nd Priority: Behavior depends on the nature of the variable
            elif category == "differential":
                # STRICT SAFEGUARD: State variables (dx/dt) demand Initial Conditions.
                raise AbsentRequiredObjectError(
                    f"The differential variable '{var_name}' requires an explicit "
                    f"Initial Condition. Please provide its value in the 'initial_conditions' "
                    f"dictionary when calling the run() method."
                )

            elif var_name in model_dict:
                # Parameters or algebraic "guesses" can fallback to the .value defined in the Model
                vec.append(model_dict[var_name].value)

            else:
                # 3rd Priority: Last resort for algebraic variables
                if category == "algebraic":
                    # If strict mode is enabled, we could force the user to provide algebraic guesses too
                    if cfg.STRICT_MODE:
                        raise AbsentRequiredObjectError(
                            f"[STRICT MODE] Missing initial guess for algebraic variable '{var_name}'."
                        )
                    vec.append(0.0)
                else:
                    raise UnexpectedValueError(
                        f"Unable to find a valid numerical value for {category}: '{var_name}'"
                    )

        return vec
