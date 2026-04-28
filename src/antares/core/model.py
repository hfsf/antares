# -*- coding: utf-8 -*-

"""
Model Module (V5 Native CasADi Architecture).

Defines the central mathematical container for equations and variables.
Upgraded with dynamic attribute overriding (__setattr__) to automatically 
register submodels and instantly propagate abstract objects upwards, 
ensuring a zero-leak declarative Equation-Oriented (EO) environment while
maintaining strict backward compatibility with positional APIs and legacy
initialization sequences.
"""

import numpy as np

from . import GLOBAL_CFG as cfg
from .constant import Constant
from .equation import Equation
from .error_definitions import UnexpectedObjectDeclarationError
from .parameter import Parameter
from .print_headings import print_heading
from .variable import Variable

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable


class Model:
    """
    Base Model class definition for phenomenological systems.
    Acts as an implicit topological graph node in the flowsheet.
    Users must inherit from this class to declare process models.
    """

    def __init__(self, name, description="", submodels=None):
        """
        Initializes the Model container.

        :param str name: Unique identifier for the model (used for namespace scoping).
        :param str description: Physical description of the modeled system.
        :param list submodels: List of child Model instances to be incorporated.
        """
        print_heading()
        self.name = name
        self.description = description
        self.submodels = submodels if submodels is not None else []
        self.parent = None  # Reference to the flowsheet parent

        self.variables = {}
        self.parameters = {}
        self.constants = {}
        self.equations = {}

        self.exposed_vars = {"input": [], "output": []}

        # Establish parent-child relationship for explicitly passed submodels
        for child in self.submodels:
            child.parent = self

    def __setattr__(self, key, value):
        """
        Overrides default attribute assignment to auto-register submodels.
        This guarantees zero abstraction leaks by instantly linking the hierarchy
        without requiring manual submodel extension. Includes an initialization 
        guard to support pre-super() instantiations.

        :param str key: Attribute name.
        :param object value: Object being assigned.
        """
        super().__setattr__(key, value)
        
        # Intercept submodel assignment to link hierarchy reactively.
        # The hasattr guard ensures we don't intercept assignments made BEFORE 
        # super().__init__ is called in the child class (legacy initialization).
        if isinstance(value, Model) and key not in ["parent", "_owner_model_instance"]:
            if hasattr(self, "variables") and hasattr(self, "submodels"):
                if value not in self.submodels:
                    self.submodels.append(value)
                if getattr(value, "parent", None) is not self:
                    value.parent = self
                    self.incorporateFromExternalModel(value)

    def __call__(self):
        """
        Resolves the configuration of the model and its hierarchical submodels.
        Executes the declarative lifecycle methods.
        """
        self.DeclareConstants()
        self.DeclareParameters()
        self.DeclareVariables()
        self.DeclareEquations()

        for child in self.submodels:
            self.incorporateFromExternalModel(child)

        if len(self.variables) == 0 and getattr(cfg, "STRICT_MODE", False):
            raise UnexpectedObjectDeclarationError(
                f"No variables declared in '{self.name}'."
            )

    def _propagate_object_upward(self, obj, category):
        """
        Recursively registers a mathematical object into the parent hierarchy.
        Ensures dynamic objects (like .fix() equations) reach the root flowsheet.

        :param object obj: The ANTARES mathematical object to propagate.
        :param str category: The target dictionary (e.g., 'variables', 'equations').
        """
        getattr(self, category)[obj.name] = obj
        if self.parent is not None:
            self.parent._propagate_object_upward(obj, category)

    def incorporateFromExternalModel(self, child_model):
        """
        Absorbs all objects from a nested submodel and triggers upward propagation.

        :param Model child_model: The instantiated submodel to absorb.
        """
        child_model.parent = self
        for category in ["variables", "parameters", "constants", "equations"]:
            for obj in getattr(child_model, category).values():
                self._propagate_object_upward(obj, category)

    # =========================================================================
    # FACTORY METHODS (STRICT API SIGNATURES RESTORED)
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
        Factory method to instantiate a Variable and bind it to the model.

        :param str name: Local name of the variable.
        :param Unit units: Dimensional unit object.
        :param str description: Physical description.
        :param bool is_lower_bounded: Flag for active lower bounds.
        :param bool is_upper_bounded: Flag for active upper bounds.
        :param float lower_bound: Numerical minimum limit.
        :param float upper_bound: Numerical maximum limit.
        :param bool is_exposed: True if the variable acts as a flowsheet port.
        :param str exposure_type: DAE classification (e.g., 'differential', 'algebraic').
        :param str latex_text: LaTeX formatting string.
        :param float value: Nominal initial value.
        :return: The generated Variable object.
        :rtype: Variable
        """
        scoped_name = f"{name}_{self.name}"
        latex_text = latex_text if latex_text else name

        var = Variable(
            name=scoped_name,
            units=units,
            description=description,
            is_lower_bounded=is_lower_bounded,
            is_upper_bounded=is_upper_bounded,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            value=value,
            is_exposed=is_exposed,
            exposure_type=exposure_type,
            latex_text=latex_text,
            owner_model_name=self.name,
        )

        var._owner_model_instance = self
        self.variables[scoped_name] = var
        setattr(self, name, var)
        self._propagate_object_upward(var, "variables")

        if is_exposed:
            self.exposed_vars[exposure_type].append(var)

        return var

    def createParameter(self, name, units, description="", value=0.0, latex_text=""):
        """
        Factory method to instantiate a Parameter and bind it to the model.

        :param str name: Local name of the parameter.
        :param Unit units: Dimensional unit object.
        :param str description: Physical description.
        :param float value: Nominal value.
        :param str latex_text: LaTeX formatting string.
        :return: The generated Parameter object.
        :rtype: Parameter
        """
        scoped_name = f"{name}_{self.name}"
        latex_text = latex_text if latex_text else name

        par = Parameter(
            name=scoped_name,
            units=units,
            description=description,
            value=value,
            latex_text=latex_text,
            owner_model_name=self.name,
        )

        par._owner_model_instance = self
        self.parameters[scoped_name] = par
        setattr(self, name, par)
        self._propagate_object_upward(par, "parameters")

        return par

    def createConstant(self, name, units, description="", value=0.0, latex_text=""):
        """
        Factory method to instantiate a Constant and bind it to the model.

        :param str name: Local name of the constant.
        :param Unit units: Dimensional unit object.
        :param str description: Physical description.
        :param float value: Constant numerical value.
        :param str latex_text: LaTeX formatting string.
        :return: The generated Constant object.
        :rtype: Constant
        """
        scoped_name = f"{name}_{self.name}"
        latex_text = latex_text if latex_text else name

        con = Constant(
            name=scoped_name,
            units=units,
            description=description,
            value=value,
            latex_text=latex_text,
            owner_model_name=self.name,
        )

        con._owner_model_instance = self
        self.constants[scoped_name] = con
        setattr(self, name, con)
        self._propagate_object_upward(con, "constants")

        return con

    def createEquation(self, name, description="", expr=None):
        """
        Factory method to instantiate an Equation and bind it to the model.

        :param str name: Local name of the equation.
        :param str description: Physical description.
        :param EquationNode|tuple expr: The residual expression.
        :return: The generated Equation object.
        :rtype: Equation
        """
        scoped_name = f"{name}_{self.name}"

        eq = Equation(
            name=scoped_name,
            description=description,
            fast_expr=expr,
            owner_model_name=self.name,
        )

        self.equations[scoped_name] = eq
        setattr(self, name, eq)
        self._propagate_object_upward(eq, "equations")

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
        Factory method to instantiate a 1D Cartesian spatial domain.

        :param str name: Local name of the domain.
        :param Unit unit: Dimensional length unit.
        :param str description: Physical description.
        :param float length: Total physical length.
        :param int n_points: Number of discretization nodes.
        :param str method: Method of discretization. Defaults to 'mol'.
        :param str diff_scheme: Differentiation scheme. Defaults to 'central'.
        :return: The generated Domain1D object.
        :rtype: Domain1D
        """
        from .domain import Domain1D

        domain_obj = Domain1D(
            name, length, n_points, unit, description, method, diff_scheme
        )
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)

        return domain_obj

    def createRadialDomain(
        self,
        name,
        unit,
        description="",
        radius=1.0,
        n_points=10,
        method="mol",
        diff_scheme="central",
        inner_radius=0.0,
    ):
        """
        Factory method to instantiate a 1D Radial domain (Cylindrical coordinates).

        :param str name: Local name of the domain.
        :param Unit unit: Dimensional length unit.
        :param str description: Physical description.
        :param float radius: Total physical outer radius.
        :param int n_points: Number of discretization nodes.
        :param str method: Method of discretization. Defaults to 'mol'.
        :param str diff_scheme: Differentiation scheme. Defaults to 'central'.
        :param float inner_radius: Starting radius for annular configurations.
        :return: The generated RadialDomain object.
        :rtype: RadialDomain
        """
        from .domain import RadialDomain

        domain_obj = RadialDomain(
            name, radius, n_points, unit, description, method, diff_scheme, inner_radius
        )
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)

        return domain_obj

    def createSphericalDomain(
        self,
        name,
        unit,
        description="",
        radius=1.0,
        n_points=10,
        method="mol",
        diff_scheme="central",
        inner_radius=0.0,
    ):
        """
        Factory method to instantiate a 1D Spherical domain (Spherical coordinates).

        :param str name: Local name of the domain.
        :param Unit unit: Dimensional length unit.
        :param str description: Physical description.
        :param float radius: Total physical outer radius.
        :param int n_points: Number of discretization nodes.
        :param str method: Method of discretization. Defaults to 'mol'.
        :param str diff_scheme: Differentiation scheme. Defaults to 'central'.
        :param float inner_radius: Starting radius for hollow configurations.
        :return: The generated SphericalDomain object.
        :rtype: SphericalDomain
        """
        from .domain import SphericalDomain

        domain_obj = SphericalDomain(
            name, radius, n_points, unit, description, method, diff_scheme, inner_radius
        )
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)

        return domain_obj

    def distributeVariable(self, variable, domain):
        """
        Discretizes the variable along the domain as a Unified Block Tensor.

        :param Variable variable: The variable to be distributed.
        :param Domain domain: The target discretization domain.
        """
        variable.distributeOnDomain(domain)
        variable._owner_model_instance = self
        if variable.name not in self.variables:
            self.variables[variable.name] = variable
        self._propagate_object_upward(variable, "variables")

    # =========================================================================
    # INITIAL & BOUNDARY CONDITIONS
    # =========================================================================

    def fix_variable(self, variable, value):
        """
        Explicitly fixes a variable's value by generating a structural equality constraint.

        :param Variable variable: Target variable to be fixed.
        :param float value: The numerical value to apply.
        """
        variable.fix(value)

    def setInitialCondition(self, variable, value, location=None):
        """
        Sets Initial Conditions safely.

        :param Variable variable: Target variable.
        :param float|ndarray value: Value to apply.
        :param tuple|slice location: Optional specific node location.
        """
        if not getattr(variable, "is_distributed", False):
            if hasattr(variable, "setValue"):
                variable.setValue(value)
            else:
                variable.value = value
        else:
            variable.setVectorialInitialCondition(value, location)

    def setBoundaryCondition(
        self, variable, domain, boundary_locator, bc_type, value=0.0
    ):
        """
        Generates Boundary Conditions safely mapping the flat topological slices
        directly into the Unified Equation Block.

        :param Variable variable: Target distributed variable.
        :param Domain domain: The domain defining the geometry.
        :param str boundary_locator: Semantic location (e.g., 'left', 'top').
        :param str bc_type: 'dirichlet' or 'neumann'.
        :param float value: Boundary fixed value or flux.
        """
        slice_idx, node_suffix = domain.get_boundary(boundary_locator)
        flat_idx = domain.get_mesh_indices()[slice_idx].flatten().tolist()

        if not hasattr(variable, "_assigned_boundary_indices"):
            variable._assigned_boundary_indices = set()

        unique_flat_idx = [
            i for i in flat_idx if i not in variable._assigned_boundary_indices
        ]

        if not unique_flat_idx:
            return

        variable._assigned_boundary_indices.update(unique_flat_idx)
        variable.setNodeType(unique_flat_idx, "algebraic")

        bc_lower = str(bc_type).lower()
        if bc_lower == "dirichlet":
            target_expr = variable()
        elif bc_lower == "neumann":
            target_expr = domain.get_normal_gradient(variable, boundary_locator)
        else:
            raise ValueError(f"Unsupported BC type '{bc_type}'.")

        residual = target_expr - value
        unique_name = f"bc_eq_{variable.name}_{boundary_locator}"

        eq = self.createEquation(
            unique_name, description=f"{bc_type} at {node_suffix}", expr=residual
        )
        eq.flat_indices = unique_flat_idx
        eq.is_distributed = True
        eq.type = "algebraic"

    def addBulkEquation(self, name, expression, domain, description=""):
        """
        Registers a governing PDE directly to the interior bulk using
        topological flat mapping.

        :param str name: Local name of the equation.
        :param EquationNode|tuple expression: The governing PDE expression.
        :param Domain domain: The domain of the variable.
        :param str description: Physical description.
        """
        bulk_slice = (
            domain.get_bulk_slice()
            if hasattr(domain, "get_bulk_slice")
            else slice(1, -1)
        )
        flat_idx = domain.get_mesh_indices()[bulk_slice].flatten().tolist()

        eq = self.createEquation(name, description=description, expr=expression)
        eq.flat_indices = flat_idx
        eq.is_distributed = True
        eq.type = "differential"

    # =========================================================================
    # DECLARATIVE INTERFACES (To be overridden by user)
    # =========================================================================

    def DeclareVariables(self):
        """User implementation block for declaring variables."""
        pass

    def DeclareParameters(self):
        """User implementation block for declaring parameters."""
        pass

    def DeclareConstants(self):
        """User implementation block for declaring constants."""
        pass

    def DeclareEquations(self):
        """User implementation block for declaring equations."""
        pass

    def print_dof_report(self):
        """Outputs an abstract mapping report of the current formulation."""
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"\n--- Model Abstract Report: {self.name} ---")
            print(f"Variables blocks:  {len(self.variables)}")
            print(f"Equations blocks:  {len(self.equations)}")