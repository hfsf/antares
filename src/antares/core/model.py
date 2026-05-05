# -*- coding: utf-8 -*-

r"""
Model Module (V5 Native CasADi Architecture).

Defines the central mathematical container for equations and variables.
Upgraded with dynamic attribute overriding (__setattr__) to automatically 
register submodels and instantly propagate abstract objects upwards, 
ensuring a zero-leak declarative Equation-Oriented (EO) environment while
maintaining strict backward compatibility with positional APIs and legacy
initialization sequences.
"""

import casadi as ca
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
    r"""
    Base Model class definition for phenomenological systems.
    
    Acts as an implicit topological graph node in the flowsheet.
    Users must inherit from this class to declare process models, applying 
    the Declarative Interface methods to instantiate physics and thermodynamics.
    """

    def __init__(self, name, description="", submodels=None):
        """
        Initializes the Model container.

        :param name: Unique identifier for the model (used for namespace scoping).
        :type name: str
        :param description: Physical description of the modeled system. Defaults to "".
        :type description: str, optional
        :param submodels: List of child Model instances to be incorporated.
        :type submodels: list, optional
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

        :param key: Attribute name.
        :type key: str
        :param value: Object being assigned.
        :type value: object
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
        Executes the declarative lifecycle methods automatically.
        
        :raises UnexpectedObjectDeclarationError: If strict mode is enabled and no variables are declared.
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

        :param obj: The ANTARES mathematical object to propagate.
        :type obj: object
        :param category: The target dictionary (e.g., 'variables', 'equations').
        :type category: str
        """
        getattr(self, category)[obj.name] = obj
        if self.parent is not None:
            self.parent._propagate_object_upward(obj, category)

    def incorporateFromExternalModel(self, child_model):
        """
        Absorbs all objects from a nested submodel and triggers upward propagation.

        :param child_model: The instantiated submodel to absorb.
        :type child_model: Model
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

        :param name: Local name of the variable.
        :type name: str
        :param units: Dimensional unit object.
        :type units: Unit
        :param description: Physical description.
        :type description: str, optional
        :param is_lower_bounded: Flag for active lower bounds.
        :type is_lower_bounded: bool, optional
        :param is_upper_bounded: Flag for active upper bounds.
        :type is_upper_bounded: bool, optional
        :param lower_bound: Numerical minimum limit.
        :type lower_bound: float, optional
        :param upper_bound: Numerical maximum limit.
        :type upper_bound: float, optional
        :param is_exposed: True if the variable acts as a flowsheet port.
        :type is_exposed: bool, optional
        :param exposure_type: DAE classification (e.g., 'differential', 'algebraic').
        :type exposure_type: str, optional
        :param latex_text: LaTeX formatting string.
        :type latex_text: str, optional
        :param value: Nominal initial value.
        :type value: float, optional
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

        :param name: Local name of the parameter.
        :type name: str
        :param units: Dimensional unit object.
        :type units: Unit
        :param description: Physical description.
        :type description: str, optional
        :param value: Nominal value.
        :type value: float, optional
        :param latex_text: LaTeX formatting string.
        :type latex_text: str, optional
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

        :param name: Local name of the constant.
        :type name: str
        :param units: Dimensional unit object.
        :type units: Unit
        :param description: Physical description.
        :type description: str, optional
        :param value: Constant numerical value.
        :type value: float, optional
        :param latex_text: LaTeX formatting string.
        :type latex_text: str, optional
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
        r"""
        Factory method to instantiate an Equation and bind it to the model.
        
        Features an autonomous DAE classification engine: it inspects the 
        computational graph (CasADi MX) for the presence of temporal derivatives 
        (e.g., symbols ending in '_dot'). If detected, the equation is 
        automatically flagged as 'differential', bypassing the need for manual 
        specification by the user.

        :param name: Local name of the equation.
        :type name: str
        :param description: Physical description. Defaults to "".
        :type description: str, optional
        :param expr: The residual expression (EquationNode or tuple).
        :type expr: EquationNode or tuple or None
        :return: The generated Equation object.
        :rtype: Equation
        """
        scoped_name = f"{name}_{self.name}"

        try:
            eq = Equation(
                name=scoped_name,
                description=description,
                expression=expr,
                owner_model_name=self.name,
            )
        except TypeError:
            eq = Equation(
                name=scoped_name,
                description=description,
                owner_model_name=self.name,
            )

        # Force AST binding for the Deep Incidence Analyzer to read it later
        eq._ast_reference = expr

        # =====================================================================
        # AUTOMATIC DAE CLASSIFICATION FALLBACK
        # Ensures robust tracking if EquationNode bypasses deep initialization
        # =====================================================================
        if eq.type != "differential" and getattr(eq, "equation_expression", None) is not None:
            mx_obj = getattr(eq.equation_expression, "symbolic_object", eq.equation_expression)
            if isinstance(mx_obj, ca.MX):
                sym_vars = ca.symvar(mx_obj)
                if any(s.name().endswith("_dot") for s in sym_vars):
                    eq.type = "differential"
        # =====================================================================

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
        """
        from .domain import Domain1D
        domain_obj = Domain1D(name, length, n_points, unit, description, method, diff_scheme)
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
        """
        from .domain import RadialDomain
        domain_obj = RadialDomain(name, radius, n_points, unit, description, method, diff_scheme, inner_radius)
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
        """
        from .domain import SphericalDomain
        domain_obj = SphericalDomain(name, radius, n_points, unit, description, method, diff_scheme, inner_radius)
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)
        return domain_obj

    def distributeVariable(self, variable, domain):
        """
        Discretizes the variable along the domain as a Unified Block Tensor.
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
        """
        variable.fix(value)

    def setInitialCondition(self, variable, value, location=None):
        """
        Sets Initial Conditions safely across scalar or distributed vectors.
        """
        variable.is_specified = True
        if not getattr(variable, "is_distributed", False):
            if hasattr(variable, "setValue"):
                variable.setValue(value)
            else:
                variable.value = value
        else:
            variable.setVectorialInitialCondition(value, location)

    def setBoundaryCondition(self, variable, domain, boundary_locator, bc_type, value=0.0):
        """
        Generates Boundary Conditions safely mapping the flat topological slices
        directly into the Unified Equation Block.
        """
        slice_idx, node_suffix = domain.get_boundary(boundary_locator)
        flat_idx = domain.get_mesh_indices()[slice_idx].flatten().tolist()

        if not hasattr(variable, "_assigned_boundary_indices"):
            variable._assigned_boundary_indices = set()

        unique_flat_idx = [i for i in flat_idx if i not in variable._assigned_boundary_indices]
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

        eq = self.createEquation(unique_name, description=f"{bc_type} at {node_suffix}", expr=residual)
        eq.flat_indices = unique_flat_idx
        eq.is_distributed = True
        eq.type = "algebraic"

    def addBulkEquation(self, name, expression, domain, description=""):
        """
        Registers a governing PDE directly to the interior bulk using
        topological flat mapping. 
        
        Relies on the autonomous DAE classification engine to dynamically 
        determine if the PDE is transient (differential) or steady-state 
        (algebraic) based on the presence of temporal derivatives in the graph.

        :param str name: Local name of the equation.
        :param expression: The mathematical residual expression (EquationNode or tuple).
        :param domain: The spatial domain object defining the mesh.
        :param str description: Physical description. Defaults to "".
        """
        bulk_slice = domain.get_bulk_slice() if hasattr(domain, "get_bulk_slice") else slice(1, -1)
        flat_idx = domain.get_mesh_indices()[bulk_slice].flatten().tolist()

        eq = self.createEquation(name, description=description, expr=expression)
        eq.flat_indices = flat_idx
        eq.is_distributed = True
        
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

    # =========================================================================
    # STRUCTURAL TOPOLOGY & INCIDENCE ANALYSIS
    # =========================================================================

    def print_dof_report(self):
        """
        Outputs a comprehensive, hierarchical Degree of Freedom (DOF) report.
        
        Extracts the mathematical contribution (Variables, Equations, Parameters) 
        of the root model and every nested submodel. Automatically detects local 
        structural over-specification. If Verbosity >= 2, extracts the Abstract 
        Syntax Tree (AST) to compute Variable-by-Variable Incidence, pointing out 
        orphan variables and highly coupled components.

        Controlled by `GLOBAL_CFG.VERBOSITY_OF_DOF_ANALYSIS`.
        """
        verbosity = getattr(cfg, "VERBOSITY_OF_DOF_ANALYSIS", 0)
        if verbosity <= 0:
            return

        print("\n" + "=" * 70)
        print(f" ANTARES STRUCTURAL TOPOLOGY & INCIDENCE REPORT: '{self.name}'")
        print("=" * 70)

        topology = {}
        target_dicts = [
            (self.variables, 'Variables'), 
            (self.equations, 'Equations'), 
            (self.parameters, 'Parameters')
        ]
        
        for obj_dict, cat in target_dicts:
            for obj in obj_dict.values():
                owner = getattr(obj, 'owner_model_name', 'Unknown')
                if owner not in topology:
                    topology[owner] = {'Variables': [], 'Equations': [], 'Parameters': []}
                topology[owner][cat].append(obj)

        for owner, contents in topology.items():
            # =================================================================
            # FIX: Mathematical Nodes vs Logical Objects for PDE consistency
            # =================================================================
            n_vars_math = 0
            for v in contents['Variables']:
                if getattr(v, 'is_distributed', False) and hasattr(v, 'n_points'):
                    n_vars_math += v.n_points
                else:
                    n_vars_math += 1

            n_eqs_math = 0
            for e in contents['Equations']:
                if getattr(e, 'is_distributed', False) and hasattr(e, 'flat_indices'):
                    # The length of flat_indices represents exactly how many 
                    # mathematical equations are generated for this logical block.
                    n_eqs_math += len(e.flat_indices)
                else:
                    n_eqs_math += 1
            
            n_vars_logical = len(contents['Variables'])
            n_eqs_logical = len(contents['Equations'])
            n_params = len(contents['Parameters'])
            
            print(f"\n[+] Submodel Node: {owner}")
            
            # Variables
            if n_vars_math == n_vars_logical:
                print(f" |-- Variables : {n_vars_logical}")
            else:
                print(f" |-- Variables : {n_vars_logical} logical objects ({n_vars_math} discrete nodes)")
                
            if verbosity >= 2 and n_vars_logical > 0:
                for v in contents['Variables']: 
                    print(f" |    |-- {v.name}")
                    
            # Equations
            if n_eqs_math == n_eqs_logical:
                print(f" |-- Equations : {n_eqs_logical}")
            else:
                print(f" |-- Equations : {n_eqs_logical} logical objects ({n_eqs_math} discrete nodes)")
                
            if verbosity >= 2 and n_eqs_logical > 0:
                for e in contents['Equations']: 
                    print(f" |    |-- {e.name}")
                    
            # Parameters
            print(f" |-- Parameters: {n_params}")
            if verbosity >= 2 and n_params > 0:
                for p in contents['Parameters']: 
                    print(f" |    |-- {p.name}")
            
            # Local Structural Analysis
            local_dof = n_vars_math - n_eqs_math
            if local_dof < 0:
                if n_vars_math == 0:
                    print(f" |-> [INFO] Routing Node Detected (Local DOF = {local_dof})")
                    print(" |   This block primarily supplies topological connections, which is expected.")
                else:
                    print(f" |-> [WARNING] Local Structural Over-specification! (Local DOF = {local_dof})")
                    print(" |   Check if this block is supplying more residual equations than variables.")

        # Deep AST Incidence Analysis (Variable-by-Variable Mapping)
        if verbosity >= 2:
            print("\n" + "-" * 70)
            print(" VARIABLE-BY-VARIABLE INCIDENCE MATRIX ANALYSIS")
            print("-" * 70)
            
            var_usage = {v_name: [] for v_name in self.variables.keys()}
            
            for e_name, e_obj in self.equations.items():
                eq_node = getattr(e_obj, "equation_expression", None)
                
                if eq_node is not None:
                    mx_obj = getattr(eq_node, "symbolic_object", eq_node)
                    
                    if isinstance(mx_obj, ca.MX):
                        sym_vars = ca.symvar(mx_obj)
                        for sym in sym_vars:
                            s_name = sym.name()
                            
                            # Clean DAE suffixes applied natively by CasADi
                            if s_name.endswith('_dot'):
                                s_name = s_name[:-4]
                            
                            if s_name in var_usage and e_name not in var_usage[s_name]:
                                var_usage[s_name].append(e_name)

            orphans = []
            over_specified = []
            
            for v_name, used_in in var_usage.items():
                count = len(used_in)
                if count == 0:
                    orphans.append(v_name)
                    print(f" [!] ORPHAN: '{v_name}' -> Not present in any equation graph.")
                elif count == 1:
                    print(f" [-] EXACT : '{v_name}' -> Solved exclusively by '{used_in[0]}'")
                else:
                    print(f" [+] COUPLED : '{v_name}' -> Appears in {count} equations.")
                    over_specified.append(v_name)
                    
            if orphans:
                print(f"\n [CRITICAL WARNING] Found {len(orphans)} variables isolated from the math graph.")
                print(" They will trigger singular Jacobians (DOF issues) if not fixed or parametrized.")

        print("=" * 70 + "\n")