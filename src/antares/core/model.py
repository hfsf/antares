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

# Graceful import for tqdm
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


class Model:
    """
    Base Model class definition.

    Users should inherit from this class and implement the declarative methods:
    
    * :meth:`DeclareVariables`
    * :meth:`DeclareParameters`
    * :meth:`DeclareConstants`
    * :meth:`DeclareEquations`
    """

    def __init__(self, name, description="", submodels=None):
        """
        Instantiate the Model.

        :param str name: Name of the current model. Must be unambiguous.
        :param str description: Short description of the current model. Defaults to "".
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
        correct execution order and incorporates any defined submodels.

        :raises UnexpectedObjectDeclarationError: If the model is evaluated without variables or equations in strict mode.
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
            if getattr(cfg, "STRICT_MODE", False):
                raise UnexpectedObjectDeclarationError(
                    f"[STRICT MODE] Fatal Error: No variables were declared in model '{self.name}'."
                )
            elif getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print(f"Warning: No variables were declared in model '{self.name}'.")

        if len(self.equations) == 0:
            if getattr(cfg, "STRICT_MODE", False):
                raise UnexpectedObjectDeclarationError(
                    f"[STRICT MODE] Fatal Error: No equations were declared in model '{self.name}'."
                )
            elif getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
                print(f"Warning: No equations were declared in model '{self.name}'.")

    # =========================================================================
    # HIERARCHY & COMPOSITION (Master Flowsheet Logic)
    # =========================================================================

    def incorporateFromExternalModel(self, child_model):
        """
        Absorbs variables, parameters, constants, and equations from a submodel
        into this master model. This flattens the hierarchy for the transpiler.

        :param Model child_model: The instantiated submodel to be incorporated.
        """
        for var_name, var_obj in child_model.variables.items():
            self.variables[var_name] = var_obj

        for par_name, par_obj in child_model.parameters.items():
            self.parameters[par_name] = par_obj

        for const_name, const_obj in child_model.constants.items():
            self.constants[const_name] = const_obj

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
        Creates a Variable object, binds it to the model, and registers it in the internal dictionary.

        :param str name: Internal and symbolic name of the variable.
        :param str units: Physical unit of the variable (e.g., 'K', 'mol/L').
        :param str description: Physical description of the variable.
        :param bool is_lower_bounded: True if the variable has a minimum value.
        :param bool is_upper_bounded: True if the variable has a maximum value.
        :param float lower_bound: The minimum numerical limit for the solver.
        :param float upper_bound: The maximum numerical limit for the solver.
        :param bool is_exposed: Indicates if the variable acts as a port for Flowsheet connections.
        :param str exposure_type: Specifies the mathematical nature, typically 'differential' or 'algebraic'.
        :param str latex_text: LaTeX string for report generation. Defaults to the variable name.
        :param float value: The nominal or initial value of the variable.
        :return: The instantiated Variable object.
        :rtype: Variable
        """
        if not latex_text:
            latex_text = name

        var = Variable(
            name, units, description, is_lower_bounded, is_upper_bounded,
            lower_bound, upper_bound, value, is_exposed, exposure_type,
            latex_text, owner_model_name=self.name,
        )

        var.name = f"{var.name}_{self.name}"
        var._owner_model_instance = self
        
        self.variables[var.name] = var
        setattr(self, name, var)

        if is_exposed:
            self.exposed_vars[exposure_type].append(var)

        return var

    def createParameter(self, name, units, description="", value=0.0, latex_text=""):
        """
        Creates a Parameter object (fixed or tunable scalar) and binds it to the model.

        :param str name: Internal and symbolic name of the parameter.
        :param str units: Physical unit.
        :param str description: Physical description.
        :param float value: Default numerical value.
        :param str latex_text: LaTeX representation. Defaults to the parameter name.
        :return: The instantiated Parameter object.
        :rtype: Parameter
        """
        if not latex_text:
            latex_text = name

        par = Parameter(name, units, description, value, latex_text, owner_model_name=self.name)
        par.name = f"{par.name}_{self.name}"
        par._owner_model_instance = self
        
        self.parameters[par.name] = par
        setattr(self, name, par)
        return par

    def createConstant(self, name, units, description="", value=0.0, latex_text=""):
        """
        Creates a Constant object (immutable physical constant) and binds it to the model.

        :param str name: Internal and symbolic name of the constant.
        :param str units: Physical unit.
        :param str description: Physical description.
        :param float value: Exact numerical value.
        :param str latex_text: LaTeX representation. Defaults to the constant name.
        :return: The instantiated Constant object.
        :rtype: Constant
        """
        if not latex_text:
            latex_text = name

        con = Constant(name, units, description, value, latex_text, owner_model_name=self.name)
        con.name = f"{con.name}_{self.name}"
        con._owner_model_instance = self
        
        self.constants[con.name] = con
        setattr(self, name, con)
        return con

    def createEquation(self, name, description="", expr=None):
        """
        Creates an Equation object and binds it to the model.
        Safely handles scalars, 1D arrays, and N-Dimensional tensors by flattening them.
        """
        if isinstance(expr, np.ndarray):
            expr = expr.flatten().tolist()

        if isinstance(expr, list):
            # Barra de carregamento para geração massiva de equações
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1 and HAS_TQDM and getattr(cfg, "SHOW_LOADING_BARS", True) :
                iter_expr = tqdm(enumerate(expr), total=len(expr), desc=f"Compiling '{name}'", unit=" eq", leave=False)
            else:
                iter_expr = enumerate(expr)

            for i, eq_expr in iter_expr:
                if isinstance(eq_expr, np.ndarray) and eq_expr.shape == (2,):
                    eq_expr = tuple(eq_expr)
                elif isinstance(eq_expr, list) and len(eq_expr) == 2:
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
        Creates a generic 1D discretization domain for Method of Lines (MoL).

        :param str name: Name of the spatial domain (e.g., 'Z_axis').
        :param Unit unit: Physical unit object (e.g., meters).
        :param str description: Physical description.
        :param float length: Total length of the domain.
        :param int n_points: Number of discrete grid points.
        :param str method: Discretization method. Defaults to 'mol'.
        :param str diff_scheme: Finite difference scheme ('central', 'backward', 'forward').
        :return: The instantiated 1D Domain.
        :rtype: Domain1D
        """
        from .domain import Domain1D
        domain_obj = Domain1D(name, length, n_points, unit, description, method, diff_scheme)
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)
        return domain_obj

    def distributeVariable(self, variable, domain):
        """
        Discretizes the variable along the given domain and registers the resulting 
        mathematical nodes. Flattens N-Dimensional arrays automatically.
        """
        variable.distributeOnDomain(domain)

        # Flatten the matrix to register each node properly
        flat_nodes = np.array(variable.discrete_nodes, dtype=object).flatten()

        # Barra de carregamento
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1 and HAS_TQDM and getattr(cfg, "SHOW_LOADING_BARS", True):
            iter_nodes = tqdm(flat_nodes, desc=f"Distributing '{variable.name}'", unit=" node", leave=False)
        else:
            iter_nodes = flat_nodes

        for node in iter_nodes:
            node._owner_model_instance = self
            self.variables[node.name] = node

        if variable.name in self.variables:
            del self.variables[variable.name]

    # =========================================================================
    # INITIAL & BOUNDARY CONDITIONS (Tensor-Safe)
    # =========================================================================

    def setInitialCondition(self, variable, value, location=None):
        """
        Sets initial conditions safely for both Lumped (ODEs) and Distributed (PDEs) variables.
        Supports multi-dimensional Numpy slicing to target specific spatial regions.

        :param Variable variable: The state variable to initialize.
        :param float value: The numerical value for the initial condition.
        :param str or slice or tuple location: The specific node location to apply the IC. 
                                               If None or 'all', applied uniformly.
        :raises ValueError: If an unsupported locator slice is provided.
        """
        if not variable.is_distributed:
            variable.setValue(value)
            return

        node_array = np.array(variable.discrete_nodes, dtype=object)

        if location is None or location == 'all':
            for node in node_array.flatten():
                node.setValue(value)
        elif location == 'start':
            node_array.flatten()[0].setValue(value)
        elif location == 'end':
            node_array.flatten()[-1].setValue(value)
        else:
            try:
                target_nodes = node_array[location]
                if not isinstance(target_nodes, np.ndarray):
                    target_nodes = [target_nodes]
                for node in np.array(target_nodes, dtype=object).flatten():
                    node.setValue(value)
            except Exception as e:
                raise ValueError(f"Failed to apply IC to locator slice '{location}'. Error: {e}")

    def setBoundaryCondition(self, variable, domain, boundary_locator, bc_type, value=0.0):
        """
        Automatically generates boundary conditions converting tensor slices into explicit 
        lists of scalar equations. 
        
        Uses variable node names to uniquely identify boundary equations, ensuring that 
        overlapping corners in 2D/3D domains naturally overwrite each other and prevent 
        Degrees of Freedom overspecification.

        :param Variable variable: The distributed state variable receiving the condition.
        :param Domain domain: The spatial domain over which the boundary is applied.
        :param str boundary_locator: Semantic identifier of the boundary (e.g., 'top', 'left', 'start').
        :param str bc_type: Type of boundary condition ('dirichlet' or 'neumann').
        :param float value: The prescribed value for the state or flux. Defaults to 0.0.
        :raises ValueError: If the requested boundary type is unsupported.
        """
        idx, node_suffix = domain.get_boundary(boundary_locator)
        description = f"{bc_type.capitalize()} boundary condition at {node_suffix}"

        def _cast_to_algebraic(var, index):
            nodes = np.array(var.discrete_nodes, dtype=object)[index]
            if not isinstance(nodes, np.ndarray):
                nodes = [nodes]
            for n in np.array(nodes, dtype=object).flatten():
                n.type = "algebraic"

        bc_lower = str(bc_type).lower()
        
        # ALL physical boundaries are pure algebraic constraints in DAEs
        _cast_to_algebraic(variable, idx)
        
        if bc_lower == "dirichlet":
            target_sym = variable()
        elif bc_lower == "neumann":
            target_sym = domain.get_normal_gradient(variable, boundary_locator)
        else:
            raise ValueError(f"Unsupported BC type '{bc_type}'.")

        # Safely extract specific symbolic elements and their parent variables
        extracted_sym = np.array(target_sym, dtype=object)[idx]
        extracted_vars = np.array(variable.discrete_nodes, dtype=object)[idx]
        
        if not isinstance(extracted_sym, np.ndarray):
            extracted_sym = [extracted_sym]
            extracted_vars = [extracted_vars]
            
        flat_syms = np.array(extracted_sym, dtype=object).flatten()
        flat_vars = np.array(extracted_vars, dtype=object).flatten()

        # Create uniquely identified equations for each node to resolve corners
        for sym, var in zip(flat_syms, flat_vars):
            unique_eq_name = f"bc_eq_{var.name}"
            
            self.createEquation(
                name=unique_eq_name, 
                description=description, 
                expr=(sym, value)
            )

    def addBulkEquation(self, name, expression, domain, description=""):
        """
        Registers a governing equation strictly to the interior (bulk) of N-D domains,
        automatically ignoring the boundaries to avoid system overspecification.

        :param str name: Base unique identifier for the equation suite.
        :param np.ndarray expression: The symbolic expression tensor.
        :param Domain domain: The domain dictating the bulk slicing logic.
        :param str description: Physical description of the equation. Defaults to "".
        """
        if hasattr(domain, 'get_bulk_slice'):
            bulk_idx = domain.get_bulk_slice()
        else:
            bulk_idx = slice(1, -1) 
            
        interior_expr = expression[bulk_idx]
        
        if isinstance(interior_expr, np.ndarray):
            flat_expr = interior_expr.flatten().tolist()
        else:
            flat_expr = [interior_expr]
            
        self.createEquation(name, description=description, expr=flat_expr)

    # =========================================================================
    # DECLARATIVE INTERFACES
    # =========================================================================

    def DeclareVariables(self):
        """User-defined hook to declare all system variables. Must be overridden."""
        pass

    def DeclareParameters(self):
        """User-defined hook to declare all system parameters. Must be overridden."""
        pass

    def DeclareConstants(self):
        """User-defined hook to declare all universal constants. Must be overridden."""
        pass

    def DeclareEquations(self):
        """User-defined hook to construct the governing equations. Must be overridden."""
        pass

    # =========================================================================
    # DIAGNOSTICS & UTILITIES
    # =========================================================================

    def print_dof_report(self):
        """
        Prints a basic Degrees of Freedom (DOF) report for the abstract model.
        Output is suppressed if GLOBAL_CFG.VERBOSITY_LEVEL is less than 1.
        Note: The definitive DOF validation is performed by the Simulator.
        """
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            n_eq = len(self.equations)
            n_var = len(self.variables)
            n_par = len(self.parameters)

            print(f"\n--- Model Report: {self.name} ---")
            print(f"Variables:  {n_var}")
            print(f"Equations:  {n_eq}")
            print(f"Parameters: {n_par}")
            print(f"Degrees of Freedom: {n_var - n_eq}")
            print("-" * 30)