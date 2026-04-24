# -*- coding: utf-8 -*-

"""
Error Definitions Module (V5 Native CasADi Architecture).

Defines custom exception classes for the ANTARES framework.
These errors provide precise, domain-specific feedback regarding topological
mismatches, dimensional incoherence, mathematical closure, and compilation failures.
"""


class ExposedVariableError(Exception):
    """
    Raised when attempting to connect models using non-exposed variables.
    """

    def __init__(
        self, model_1_exposed_vars, model_2_exposed_vars, output_var, input_var
    ):
        """
        :param list model_1_exposed_vars: List of exposed variables in the output model.
        :param list model_2_exposed_vars: List of exposed variables in the input model.
        :param Variable output_var: The referenced output variable.
        :param Variable input_var: The referenced input variable.
        """
        self.m1_exposed_names = [var_i.name for var_i in model_1_exposed_vars]
        self.m2_exposed_names = [var_i.name for var_i in model_2_exposed_vars]
        self.output_var_name = output_var.name
        self.input_var_name = input_var.name

    def __str__(self):
        return (
            f"Non-exposed variable declaration in the output model (1):\n{self.m1_exposed_names}\n"
            f"and/or input model (2):\n{self.m2_exposed_names}.\n"
            f"The declared output variable is '{self.output_var_name}', "
            f"and the input variable is '{self.input_var_name}'."
        )


class UnexpectedObjectDeclarationError(Exception):
    """
    Raised when an unregistered Variable, Parameter, or Constant is utilized
    within the current Model scope.
    """

    def __init__(self, objects, declared_objects):
        self.objects = objects
        self.declared_objects = declared_objects

    def __str__(self):
        return (
            f"Unexpected object declaration error.\n"
            f"The following objects were illegally used: {self.objects}\n"
            f"Valid declared objects for the current model are: {self.declared_objects}"
        )


class AbsentRequiredObjectError(Exception):
    """
    Raised when a mathematically required object (e.g., Initial Condition) is missing.
    """

    def __init__(self, expected_type, supplied_object=""):
        self.expected_type = expected_type
        self.supplied_object = supplied_object

    def __str__(self):
        msg = f"Absent required object error. A '{self.expected_type}' was expected, but none was supplied."
        if self.supplied_object:
            msg += f" Supplied object was: {self.supplied_object}."
        return msg


class UnexpectedValueError(Exception):
    """
    Raised when an unsupported type or value is injected into an algebraic operation.
    """

    def __init__(self, expected_type):
        self.expected_type = expected_type

    def __str__(self):
        return f"Unexpected value error. Expected type(s): {self.expected_type}, but a divergent type was supplied."


class UnresolvedPanicError(Exception):
    """
    Critical framework failure. Raised by unhandled or systemic state corruption.
    """

    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return f"Unresolved Panic Error: {self.msg}\nThis should not have occurred. Please debug the framework core."


class NumericalError(Exception):
    """
    Raised by unsolvable numeric conditions during evaluation.
    """

    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return (
            f"NumericalError: {self.msg}\nThis indicates a severe mathematical instability."
            if self.msg
            else "NumericalError: Unknown numerical instability."
        )


class NonDimensionalArgumentError(Exception):
    """
    Raised when a dimensional argument is provided to a function that mathematically
    requires a dimensionless input (e.g., transcendental functions like log, sin, exp).
    """

    def __init__(self, unit):
        self.unit = unit

    def __str__(self):
        return f"A dimensionless argument was expected, but received dimensions:\n{self.unit.dimension}"


class DimensionalCoherenceError(Exception):
    """
    Raised when addition, subtraction, or assignment operations attempt to merge
    variables with conflicting physical dimensions.
    """

    def __init__(self, unit_1, unit_2):
        null_dim = {
            "m": 0.0,
            "kg": 0.0,
            "s": 0.0,
            "A": 0.0,
            "K": 0.0,
            "mol": 0.0,
            "cd": 0.0,
        }
        self.unit_1_dim = unit_1.dimension if unit_1 is not None else null_dim
        self.unit_2_dim = unit_2.dimension if unit_2 is not None else null_dim

    def __str__(self):
        return f"Dimensional coherence violation. Dimensions are incompatible:\n({self.unit_1_dim}\n != \n{self.unit_2_dim})."


class DegreesOfFreedomError(Exception):
    """
    Raised when the mathematical system is not perfectly closed
    (Degrees of Freedom != 0) prior to the DAE numerical integration phase.
    """

    pass


class UnitOperationError(Exception):
    """
    Raised due to structural errors in the definition of Unit Operation flowsheet elements.
    """

    def __init__(self, port, n, elem):
        self.n = n
        self.port = port
        self.elem = elem

    def __str__(self):
        return f"UnitOp object was defined for port '{self.port}' with length {self.n}, but an element of size {len(self.elem)} was supplied."
