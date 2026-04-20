# -*- coding: utf-8 -*-

"""
Model Module.

Defines the Model class for the ANTARES framework.
This class acts as a mathematical container, storing equations, variables,
parameters, and constants. It provides a clean, object-oriented API for users
to define phenomenological models before they are transpiled to the CasADi backend.
"""

import numpy as np

from . import GLOBAL_CFG as cfg
from .constant import Constant
from .equation import Equation
from .error_definitions import UnexpectedObjectDeclarationError
from .parameter import Parameter
from .print_headings import print_heading
from .variable import Variable


class Model:
    """
    Base Model class definition.

    Users should inherit from this class and implement the declarative methods:
    - DeclareVariables()
    - DeclareParameters()
    - DeclareConstants()
    - DeclareEquations()
    """

    def __init__(self, name, description="", submodels=None):
        """
        Instantiate the Model.

        :param str name: Name of the current model. Must be unambiguous.
        :param str description: Short description of the current model.
        :param list submodels: Optional list of instantiated child models to be automatically incorporated.
        """
        # Print banner
        print_heading()

        self.name = name
        self.description = description

        # Stores the list of submodels passed during initialization
        self.submodels = submodels if submodels is not None else []

        # Core mathematical dictionaries
        self.variables = {}
        self.parameters = {}
        self.constants = {}
        self.equations = {}

        # Dictionary to store exposed variables (useful for flowsheet connections)
        self.exposed_vars = {"input": [], "output": []}

    def __call__(self):
        """
        Overloaded function used to resolve all configurations necessary for
        model definition. It calls the user-defined declarative methods in the
        correct execution order.
        """
        self.DeclareConstants()
        self.DeclareParameters()
        self.DeclareVariables()
        self.DeclareEquations()

        # Automatic incorporation of submodels (Hierarchical flattening)
        for child in self.submodels:
            self.incorporateFromExternalModel(child)

        # Validation diagnostics linked to GLOBAL_CFG
        if len(self.variables) == 0:
            if cfg.STRICT_MODE:
                raise UnexpectedObjectDeclarationError(
                    f"[STRICT MODE] Fatal Error: No variables were declared in model '{self.name}'.",
                    self.variables
                )
            elif getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print(f"Warning: No variables were declared in model '{self.name}'.")

        if len(self.equations) == 0:
            if cfg.STRICT_MODE:
                raise UnexpectedObjectDeclarationError(
                    f"[STRICT MODE] Fatal Error: No equations were declared in model '{self.name}'.",
                    self.variables
                )
            elif getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print(f"Warning: No equations were declared in model '{self.name}'.")

    # =========================================================================
    # HIERARCHY & COMPOSITION (Master Flowsheet Logic)
    # =========================================================================

    def incorporateFromExternalModel(self, child_model):
        """
        Absorbs variables, parameters, constants, and equations from a submodel
        (child_model) into this master model.

        This flattens the hierarchy, allowing the CasADi transpiler to find
        all system equations in a single, unified dictionary.

        :param Model child_model: The submodel instance to be absorbed.
        """
        # 1. Absorb Variables
        for var_name, var_obj in child_model.variables.items():
            self.variables[var_name] = var_obj

        # 2. Absorb Parameters
        for par_name, par_obj in child_model.parameters.items():
            self.parameters[par_name] = par_obj

        # 3. Absorb Constants
        for const_name, const_obj in child_model.constants.items():
            self.constants[const_name] = const_obj

        # 4. Absorb Equations
        for eq_name, eq_obj in child_model.equations.items():
            self.equations[eq_name] = eq_obj

    # =========================================================================
    # FACTORY METHODS (The Syntax Sugar)
    # =========================================================================

    def createVariable(
        self,
        name,
        units,
        description="",
        is_lower_bounded=False,
        is_upper_bounded=False,
        lower_bound=None,
        upper_bound=None,
        is_exposed=False,
        exposure_type="",
        latex_text="",
        value=0.0,
    ):
        """
        Creates a Variable object, stores it in the model's dictionary,
        and makes it accessible as an attribute.

        :param str name: Identifier for the variable.
        :param str/Unit units: Physical unit of the variable.
        :param str exposure_type: 'algebraic' or 'differential'.
        :return: The instantiated Variable object.
        :rtype: Variable
        """
        if not latex_text:
            latex_text = name

        var = Variable(
            name,
            units,
            description,
            is_lower_bounded,
            is_upper_bounded,
            lower_bound,
            upper_bound,
            value,
            is_exposed,
            exposure_type,
            latex_text,
            owner_model_name=self.name,
        )

        # Ensures unique naming across the flowsheet
        var.name = f"{var.name}_{self.name}"

        # Dependency Injection: Bind variable to its parent model
        var._owner_model_instance = self
        
        self.variables[var.name] = var
        setattr(self, name, var)

        if is_exposed:
            self.exposed_vars[exposure_type].append(var)

        return var

    def createParameter(self, name, units, description="", value=0.0, latex_text=""):
        """Creates a Parameter object and binds it to the model."""
        if not latex_text:
            latex_text = name

        par = Parameter(
            name, units, description, value, latex_text, owner_model_name=self.name
        )
        par.name = f"{par.name}_{self.name}"

        par._owner_model_instance = self
        self.parameters[par.name] = par
        setattr(self, name, par)
        
        return par

    def createConstant(self, name, units, description="", value=0.0, latex_text=""):
        """Creates a Constant object and binds it to the model."""
        if not latex_text:
            latex_text = name

        con = Constant(
            name, units, description, value, latex_text, owner_model_name=self.name
        )
        con.name = f"{con.name}_{self.name}"

        con._owner_model_instance = self
        self.constants[con.name] = con
        setattr(self, name, con)
        
        return con

    def createEquation(self, name, description="", expr=None):
        """
        Creates an Equation object and binds it to the model.
        Handles both scalar expressions and arrays of expressions natively.
        """
        if isinstance(expr, (np.ndarray, list)):
            for i, eq_expr in enumerate(expr):
                # THE MAGIC OF ROBUSTNESS: If Numpy converted the (LHS, RHS) tuple
                # into a 1D matrix row during broadcasting, we cast it back to a Tuple!
                if isinstance(eq_expr, np.ndarray) and eq_expr.shape == (2,):
                    eq_expr = tuple(eq_expr)

                eq_name = f"{name}_{i}"
                eq = Equation(
                    name=eq_name,
                    description=f"{description} (Node {i})",
                    fast_expr=eq_expr,
                    owner_model_name=self.name,
                )
                self.equations[eq_name] = eq
                setattr(self, eq_name, eq)
        else:
            eq = Equation(name, description, expr, owner_model_name=self.name)
            eq.name = f"{eq.name}_{self.name}"

            self.equations[eq.name] = eq
            setattr(self, name, eq)
            return eq

    def createDomain(
        self,
        name,
        unit,
        description="",
        length=1.0,
        n_points=10,
        method="mol",
        diff_scheme="central",
    ):
        """
        Creates a discretization domain (spatial, temporal, population, etc.).
        Acts as an independent variable over which other variables can be distributed.
        """
        from .domain import Domain1D

        domain_obj = Domain1D(name, length, n_points, unit, description, method, diff_scheme)

        # Dependency Injection
        domain_obj._owner_model_instance = self

        setattr(self, name, domain_obj)
        return domain_obj

    def distributeVariable(self, variable, domain):
        """
        Discretizes the variable along the given domain and registers the resulting 
        mathematical nodes in the master solver dictionary for transpilation mapping.
        
        :param Variable variable: The variable to be distributed.
        :param Domain1D domain: The spatial domain to distribute upon.
        """
        # 1. The variable internally generates its discrete nodes
        variable.distributeOnDomain(domain)

        # 2. Register each node in the official simulator dictionary
        for node in variable.discrete_nodes:
            # Ensure the model is recognized as the node's owner
            node._owner_model_instance = self

            # Add to the main dictionary for the CasADi transpiler mapping
            self.variables[node.name] = node

        # 3. Clean up the "parent" abstract variable from the solver dictionary.
        # It remains accessible via self.var_name to generate derivatives,
        # but CasADi will no longer try to create a useless MX symbol for it.
        if variable.name in self.variables:
            del self.variables[variable.name]

    # =========================================================================
    # INITIAL & BOUNDARY CONDITIONS (Dimension-Agnostic Abstractions)
    # =========================================================================

    def setInitialCondition(self, variable, value, location=None):
        """
        Sets initial conditions safely for both Lumped (ODEs) and Distributed (PDEs) variables.
        Encapsulates the initialization logic, preventing abstraction leaks to the execution layer.

        :param Variable variable: The state variable to initialize.
        :param float/Quantity value: The numerical value for the initial condition.
        :param str/tuple/slice location: The specific node location to apply the IC. 
            If None or 'all', the value is applied uniformly.
            Accepts 1D semantics ('start', 'end') or N-D array slicing. Defaults to None.
        :raises ValueError: If an unsupported locator is provided.
        """
        # 1. LUMPED VARIABLE (Pure ODE): No discrete nodes exist.
        if not variable.is_distributed:
            variable.setValue(value)
            return

        # 2. DISTRIBUTED VARIABLE (PDE): Has discrete nodes.
        if location is None or location == 'all':
            for node in variable.discrete_nodes:
                node.setValue(value)
        elif location == 'start':
            variable.discrete_nodes[0].setValue(value)
        elif location == 'end':
            variable.discrete_nodes[-1].setValue(value)
        else:
            # Handles Numpy advanced slicing (e.g., tuple of slices for 2D/3D grids)
            node_array = np.array(variable.discrete_nodes)
            try:
                target_nodes = node_array[location]
                # If a single node is returned, make it iterable
                if not isinstance(target_nodes, np.ndarray):
                    target_nodes = [target_nodes]
                for node in target_nodes.flatten():
                    node.setValue(value)
            except Exception as e:
                raise ValueError(
                    f"Failed to apply initial condition to locator slice '{location}'. Error: {e}"
                )
    
    def setBoundaryCondition(self, variable, domain, boundary_locator, bc_type, value=0.0):
        """
        Automatically generates and registers boundary condition (BC) equations.
        Delegates the geometric parsing to the Domain and automatically re-casts 
        boundary nodes from 'differential' to 'algebraic' since physical boundaries 
        are algebraic constraints.
        """
        # 1. DELEGATE BOUNDARY PARSING TO THE DOMAIN
        idx, node_suffix = domain.get_boundary(boundary_locator)

        # 2. GENERATE UNIQUE SOLVER IDENTIFIERS
        eq_name = f"bc_{variable.name}_{domain.name}_{node_suffix}"
        description = f"{bc_type.capitalize()} boundary condition at {node_suffix}"

        # Helper function to safely downgrade boundary nodes to algebraic states
        def _cast_to_algebraic(var, index):
            nodes = np.array(var.discrete_nodes, dtype=object)[index]
            if not isinstance(nodes, np.ndarray):
                nodes = [nodes]
            else:
                nodes = nodes.flatten()
                
            for n in nodes:
                # Transforma a EDO numa variável Algébrica para o Solver DAE
                n.type = "algebraic"  

        # 3. APPLY MATHEMATICAL CONSTRAINTS
        bc_lower = str(bc_type).lower()
        
        if bc_lower == "dirichlet":
            # Dirichlet conditions prescribe the value of the state. 
            _cast_to_algebraic(variable, idx)
            self.createEquation(
                name=eq_name, 
                description=description, 
                expr=(variable()[idx] == value)
            )

        elif bc_lower == "neumann":
            # Neumann applies to the spatial gradient. Purely algebraic!
            _cast_to_algebraic(variable, idx)
            self.createEquation(
                name=eq_name, 
                description=description, 
                expr=(variable.Grad(domain)[idx] == value)
            )

        else:
            raise ValueError(
                f"Boundary condition type '{bc_type}' is not supported by ANTARES. "
                f"Supported types are: 'dirichlet', 'neumann'."
            )
          
    def addBulkEquation(self, name, expression, domain, description=""):
        """
        Registers an equation strictly to the interior (bulk) nodes of a domain.
        Delegates the logic of finding the "bulk" indices to the Domain object,
        ensuring compatibility across 1D, 2D, or 3D coordinate systems without 
        abstraction leaks (like manual [1:] slicing).

        :param str name: Unique identifier for the equation.
        :param np.ndarray expression: The vectorized mathematical expression (PDE or Algebraic).
        :param Domain domain: The spatial domain defining the geometry.
        :param str description: Optional equation description.
        """
        # Dimension-Agnostic Bulk Slicing
        if hasattr(domain, 'get_bulk_slice'):
            bulk_idx = domain.get_bulk_slice()
        else:
            # Safety fallback for older domain definitions
            bulk_idx = slice(1, -1) 
            
        interior_expr = expression[bulk_idx]
        self.createEquation(name, description=description, expr=interior_expr)

    # =========================================================================
    # DECLARATIVE INTERFACES (To be overridden by the user)
    # =========================================================================

    def DeclareVariables(self):
        """User-defined hook to declare all system variables."""
        pass

    def DeclareParameters(self):
        """User-defined hook to declare all system parameters."""
        pass

    def DeclareConstants(self):
        """User-defined hook to declare all universal constants."""
        pass

    def DeclareEquations(self):
        """User-defined hook to construct the governing equations."""
        pass

    # =========================================================================
    # DIAGNOSTICS & UTILITIES
    # =========================================================================

    def print_dof_report(self):
        """
        Prints a basic Degrees of Freedom (DOF) report for the model.
        Output is suppressed if VERBOSITY_LEVEL is set to 0.
        """
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            n_eq = len(self.equations)
            n_var = len(self.variables)
            n_par = len(self.parameters)

            print(f"\n--- Model Report: {self.name} ---")
            print(f"Variables:  {n_var}")
            print(f"Equations:  {n_eq}")
            print(f"Parameters: {n_par}")
            print(f"Degrees of Freedom (Vars - Eqs): {n_var - n_eq}")
            print("-" * 30)