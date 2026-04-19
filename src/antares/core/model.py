# -*- coding: utf-8 -*-

"""
Define the Model class for the ANTARES framework.
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

        # Dictionary to store exposed variables (useful for connections)
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
                    f"[STRICT MODE] Fatal Error: No variables were declared in model '{self.name}'."
                )
            elif cfg.VERBOSITY_LEVEL >= 1:
                print(f"Warning: No variables were declared in model '{self.name}'.")

        if len(self.equations) == 0:
            if cfg.STRICT_MODE:
                raise UnexpectedObjectDeclarationError(
                    f"[STRICT MODE] Fatal Error: No equations were declared in model '{self.name}'."
                )
            elif cfg.VERBOSITY_LEVEL >= 1:
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

        var._owner_model_instance = self
        
        self.variables[var.name] = var
        
        setattr(self, name, var)

        if is_exposed:
            self.exposed_vars[exposure_type].append(var)

        setattr(self, name, var)
        return var

    def createParameter(self, name, units, description="", value=0.0, latex_text=""):
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
        if isinstance(expr, np.ndarray) or isinstance(expr, list):
            for i, eq_expr in enumerate(expr):
                # A MAGIA DA ROBUSTEZ: Se o Numpy converteu o tuplo (LHS, RHS)
                # numa linha de matriz 1D, nós convertemos de volta para Tuplo!
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

    def distributeVariable(self, variable, domain):
        """
        Discretiza a variável ao longo do domínio e regista os nós matemáticos
        no motor principal (Model) para que o Transpilador os mapeie.
        """
        # 1. A variável gera os seus 50 nós internamente
        variable.distributeOnDomain(domain)

        # 2. A CORREÇÃO: Registamos cada nó na lista oficial do simulador
        for node in variable.discrete_nodes:
            # Garante que o modelo é reconhecido como o dono do nó
            node._owner_model_instance = self

            # Adiciona ao dicionário principal para o transpiler criar o símbolo CasADi!
            self.variables[node.name] = node

        # 3. (Opcional, mas limpo) Removemos a variável "mãe" do dicionário do solver.
        # Ela continuará acessível via self.T para gerar derivadas,
        # mas o CasADi já não vai tentar criar uma incógnita inútil para ela.
        if variable.name in self.variables:
            del self.variables[variable.name]

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
        Cria um domínio de discretização (espacial, temporal, populacional, etc.).
        Atua como uma variável independente sobre a qual outras variáveis podem ser distribuídas.
        """
        from .domain import Domain1D

        domain_obj = Domain1D(name, length, n_points, unit, description, method, diff_scheme)

        domain_obj._owner_model_instance = self

        setattr(self, name, domain_obj)
        return domain_obj

    # =========================================================================
    # BOUNDARY CONDITIONS
    # =========================================================================

    def setBoundaryCondition(self, variable, domain, boundary_locator, bc_type, value=0.0):
        """
        Automatically generates and registers boundary condition (BC) equations.

        This method is dimension-agnostic and discretization-aware. It adapts the 
        underlying mathematical constraints depending on whether the domain uses 
        the Method of Lines (MOL) or Orthogonal Collocation, and supports N-dimensional 
        boundaries via array slicing.

        :param Variable variable: The distributed state variable to apply the BC on.
        :param Domain1D domain: The spatial domain associated with the boundary.
        :param str/tuple/slice boundary_locator: The physical location of the boundary. 
            Can be a semantic string for 1D ('start', 'inlet', 'left', 'end', 'outlet', 'right') 
            or a standard Python/NumPy slice/tuple for N-dimensional grids (e.g., `(slice(None), 0)`).
        :param str bc_type: The mathematical classification of the boundary. 
            Accepted values: 'dirichlet' (fixed value) or 'neumann' (fixed spatial gradient).
        :param float/EquationNode value: The numerical or symbolic value of the condition. 
            Defaults to 0.0.
        :raises ValueError: If an unsupported boundary locator or type is provided.
        """
        
        # 1. PARSE BOUNDARY LOCATOR (Dimension-Agnostic Support)
        if isinstance(boundary_locator, str):
            pos_lower = boundary_locator.lower()
            if pos_lower in ["start", "inlet", "left", "bottom", "front"]:
                idx = 0
                node_suffix = "start"
            elif pos_lower in ["end", "outlet", "right", "top", "back"]:
                idx = -1
                node_suffix = "end"
            else:
                raise ValueError(
                    f"Invalid boundary string locator '{boundary_locator}'. "
                    f"Use 1D semantics ('start'/'end') or provide an N-D index tuple/slice."
                )
        else:
            # N-Dimensional support (e.g., idx = (slice(None), 0) for a 2D edge)
            idx = boundary_locator
            node_suffix = f"idx_{str(idx).replace(' ', '')}"

        # 2. GENERATE UNIQUE SOLVER IDENTIFIERS
        eq_name = f"bc_{variable.name}_{domain.name}_{node_suffix}"
        description = f"{bc_type.capitalize()} boundary condition at {node_suffix}"

        # 3. APPLY MATHEMATICAL CONSTRAINTS (Discretization-Aware)
        bc_lower = str(bc_type).lower()
        method = getattr(domain, 'method', 'mol').lower()
        
        if bc_lower == "dirichlet":
            if method == "mol":
                # Method of Lines (ODE/DAE): Lock the state by forcing temporal derivative to zero.
                # The actual numerical value must be provided in the initial conditions (ic_dict).
                self.createEquation(
                    name=eq_name, 
                    description=description, 
                    expr=(variable.Diff()[idx] == 0.0)
                )
            else:
                # Collocation / Finite Elements (NLP): Purely algebraic constraint.
                # Enforces state == value directly at the boundary nodes.
                self.createEquation(
                    name=eq_name, 
                    description=description, 
                    expr=(variable()[idx] == value)
                )

        elif bc_lower == "neumann":
            # Neumann applies to the spatial gradient (e.g., Fourier's law flux).
            # Works symmetrically for both MOL and Collocation since spatial derivatives 
            # are algebraic matrix multiplications in both methods.
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

    # =========================================================================
    # DECLARATIVE INTERFACES (To be overridden by the user)
    # =========================================================================

    def DeclareVariables(self):
        pass

    def DeclareParameters(self):
        pass

    def DeclareConstants(self):
        pass

    def DeclareEquations(self):
        pass

    # =========================================================================
    # DIAGNOSTICS & UTILITIES
    # =========================================================================

    def print_dof_report(self):
        """
        Prints a basic Degrees of Freedom (DOF) report for the model.
        Output is suppressed if VERBOSITY_LEVEL is set to 0.
        """
        if cfg.VERBOSITY_LEVEL >= 1:
            n_eq = len(self.equations)
            n_var = len(self.variables)
            n_par = len(self.parameters)

            print(f"\n--- Model Report: {self.name} ---")
            print(f"Variables:  {n_var}")
            print(f"Equations:  {n_eq}")
            print(f"Parameters: {n_par}")
            print(f"Degrees of Freedom (Vars - Eqs): {n_var - n_eq}")
            print("-" * 30)
