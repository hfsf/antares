# -*- coding: utf-8 -*-

"""
Define the Unit class for the ANTARES framework.
Handles dimensional tracking, parsing of string-based unit definitions,
and dimensional coherence analysis across the mathematical equations.
"""

import copy

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import DimensionalCoherenceError, UnexpectedValueError

# Null dimension dictionary
null_dimension = {
    "m": 0.0,
    "kg": 0.0,
    "s": 0.0,
    "A": 0.0,
    "K": 0.0,
    "mol": 0.0,
    "cd": 0.0,
}


def _sanitizeUnitDef(unsanitized_str):
    """
    Sanitizes string-based unit definitions (e.g., 'kg * m / s**2') to ensure
    they contain only allowed mathematical operators and pre-defined base units.
    """
    unsanitized_str = (
        unsanitized_str.replace("+", " + ").replace("^", " ** ").replace("**", " ** ")
    )
    unsanitized_str = (
        unsanitized_str.replace("-", " - ").replace("*", " * ").replace("/", " / ")
    )
    unsanitized_str = unsanitized_str.replace("(", " ( ").replace(")", " ) ")

    unsanitized_list = unsanitized_str.split()
    allowed_strings = list(predef_units_map.keys())
    op_strings = ["+", "-", "*", "/", "(", ")", "^", "**"]
    allowed_strings.extend(op_strings)

    # Check for illegal terms (e.g., malicious code injection or typos)
    for u in unsanitized_list:
        if u not in allowed_strings and not u.replace(".", "", 1).isdigit():
            raise ValueError(f"Illegal term '{u}' detected in unit definition.")

    return "".join(unsanitized_list)


def _processUnitDef(unit_definition):
    """
    Evaluates the sanitized string to generate the resulting compound Unit object.
    """
    sanitized = _sanitizeUnitDef(unit_definition)
    # Safe evaluation using only the predefined units mapping
    return eval(sanitized, {"__builtins__": None}, predef_units_map)


class Unit:
    """
    Unit class definition. Holds capabilities for:
    - Tracking the dimensional index for each SI dimension.
    - Providing overloaded mathematical operators (*, /, **) to create
      derived units dynamically (e.g., m * m = m²).
    """

    def __init__(self, name, dimension_dict, description=""):
        """
        Initializes the Unit object.

        :param str name: Name of the present unit.
        :param dimension_dict: Dictionary containing dimensions (or a string expression).
        :type dimension_dict: dict or str or Unit
        :param str description: Short description of the unit.
        """
        self.name = name
        self.description = description
        self.dimension = {k: 0.0 for k in null_dimension.keys()}
        self._re_eval_dimensions(dimension_dict)

    def _is_dimensionless(self):
        """
        Checks if the current Unit object is dimensionless (all exponents are 0).
        """
        return all(float(d_i) == 0.0 for d_i in self.dimension.values())

    def _re_eval_dimensions(self, dimension_dict):
        """
        Private function to evaluate and assign the dimensions of the unit.
        """
        if isinstance(dimension_dict, Unit):
            dimension_dict = dimension_dict.dimension
        elif isinstance(dimension_dict, str):
            dimension_dict = _processUnitDef(dimension_dict).dimension

        for dim_i, idx_i in dimension_dict.items():
            if dim_i in self.dimension:
                self.dimension[dim_i] = idx_i

    def __str__(self):
        """
        Prints the dimension of the Unit in a convenient algebraic way.
        Example: 'm^1 s^-1'
        """
        output = [f"{dim}^{val}" for dim, val in self.dimension.items() if val != 0.0]
        return " ".join(output) if output else "dimensionless"

    def __add__(self, other_unit):
        """
        Addition of two units. Returns the same unit if coherent.
        """
        if self._check_dimensional_coherence(other_unit):
            return self
        raise DimensionalCoherenceError(self, other_unit)

    def __sub__(self, other_unit):
        """
        Subtraction of two units. Returns the same unit if coherent.
        """
        if self._check_dimensional_coherence(other_unit):
            return self
        raise DimensionalCoherenceError(self, other_unit)

    def __mul__(self, other_unit):
        """
        Multiplication of units. Sums the dimensions to create a derived unit.
        """
        if isinstance(other_unit, self.__class__):
            new_dimension = copy.copy(self.dimension)
            for dim_i, idx_i in other_unit.dimension.items():
                new_dimension[dim_i] += idx_i
            return self.__class__(name="", dimension_dict=new_dimension)

        elif isinstance(other_unit, (float, int)):
            # Local import to avoid circular dependency with quantity.py
            from .quantity import Quantity

            return Quantity("", self.__class__("", self.dimension), value=other_unit)
        else:
            raise UnexpectedValueError("(Unit, float, int)")

    def __truediv__(self, other_unit):
        """
        True division of units. Subtracts the dimensions to create a derived unit.
        """
        if isinstance(other_unit, self.__class__):
            new_dimension = copy.copy(self.dimension)
            for dim_i, idx_i in other_unit.dimension.items():
                new_dimension[dim_i] -= idx_i
            return self.__class__(name="", dimension_dict=new_dimension)

        elif isinstance(other_unit, (float, int)):
            from .quantity import Quantity

            return Quantity(
                "", self.__class__("", self.dimension), value=1.0 / other_unit
            )
        else:
            raise UnexpectedValueError("(Unit, float, int)")

    def __pow__(self, power):
        """
        Exponentiation of a unit. Multiplies the dimensions by the power.
        """
        if isinstance(power, (int, float)):
            new_dimension = {dim: val * power for dim, val in self.dimension.items()}
            return self.__class__(name="", dimension_dict=new_dimension)

        elif isinstance(power, self.__class__) and power._is_dimensionless():
            return self.__class__(name="", dimension_dict=self.dimension)
        else:
            raise UnexpectedValueError("(int, float, dimensionless unit)")

    def _check_dimensional_coherence(self, other_unit):
        """
        Checks if two units share exactly the same dimensions.
        """
        if cfg.DIMENSIONAL_COHERENCE_CHECK:
            return all(
                self.dimension[idx] == other_unit.dimension[idx]
                for idx in self.dimension.keys()
            )
        else:
            # Respect the global verbosity setting to avoid terminal spam
            if cfg.VERBOSITY_LEVEL >= 2:
                print("Warning: Skipping dimensional coherence test.")
            return True


# =============================================================================
# PREDEFINED BASE UNITS (SI SYSTEM)
# =============================================================================
_kg_ = Unit("kilogram", {"kg": 1.0})
_m_ = Unit("meter", {"m": 1.0})
_s_ = Unit("second", {"s": 1.0})
_A_ = Unit("Ampere", {"A": 1.0})
_K_ = Unit("Kelvin", {"K": 1.0})
_mol_ = Unit("mol", {"mol": 1.0})

# Derived common units
_J_ = Unit("Joule", {"kg": 1.0, "m": 2.0, "s": -2.0})
_N_ = Unit("Newton", {"kg": 1.0, "m": 1.0, "s": -2.0})
_Pa_ = Unit("Pascal", {"kg": 1.0, "m": -1.0, "s": -2.0})

predef_units_map = {
    "kg": _kg_,
    "m": _m_,
    "s": _s_,
    "A": _A_,
    "K": _K_,
    "mol": _mol_,
    "J": _J_,
    "N": _N_,
    "Pa": _Pa_,
}
