# -*- coding: utf-8 -*-

"""
Unit Module (V5 Native CasADi Architecture).

Defines the Unit class for the ANTARES framework.
Handles dimensional tracking, parsing of string-based unit definitions,
and dimensional coherence analysis across mathematical equations.
In the V5 architecture, this module acts as the strict physical guardian,
validating dimensional integrity in pure Python before operations are
delegated to the CasADi C++ computational graph.
"""

import copy

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import DimensionalCoherenceError, UnexpectedValueError

# Null dimension dictionary mapping the 7 base SI dimensions
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
    they contain only allowed mathematical operators and predefined base units.

    :param str unsanitized_str: The raw string definition of the unit.
    :return: A safe, formatted string ready for parsing.
    :rtype: str
    :raises ValueError: If illegal characters, terms, or injection attempts are detected.
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
    Evaluates the sanitized string mathematically to generate the resulting compound Unit object.

    :param str unit_definition: The sanitized string representation of the unit algebra.
    :return: The corresponding derived Unit object.
    :rtype: Unit
    """
    sanitized = _sanitizeUnitDef(unit_definition)
    # Safe evaluation using strictly the predefined units mapping
    return eval(sanitized, {"__builtins__": None}, predef_units_map)


class Unit:
    """
    Unit class definition.
    Tracks the dimensional index for each SI dimension and overloads mathematical
    operators (*, /, **) to create derived units dynamically (e.g., m * m = m²).
    """

    def __init__(self, name, dimension_dict, description=""):
        """
        Initializes the Unit object.

        :param str name: Internal name of the present unit.
        :param dimension_dict: Dictionary containing dimensions, a string expression, or another Unit.
        :type dimension_dict: dict, str, or Unit
        :param str description: Short physical description of the unit.
        """
        self.name = name
        self.description = description
        self.dimension = {k: 0.0 for k in null_dimension.keys()}
        self._re_eval_dimensions(dimension_dict)

    def _is_dimensionless(self):
        """
        Checks if the current Unit object is dimensionless (all exponents are zero).

        :return: True if the unit represents a dimensionless quantity.
        :rtype: bool
        """
        return all(float(d_i) == 0.0 for d_i in self.dimension.values())

    def _re_eval_dimensions(self, dimension_dict):
        """
        Internally evaluates and assigns the SI dimensions of the unit.

        :param dimension_dict: The source representation of the unit's dimensions.
        :type dimension_dict: dict, str, or Unit
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
        Formats the dimension of the Unit into a convenient algebraic string (e.g., 'm^1 s^-1').

        :return: Algebraic string representation of the unit.
        :rtype: str
        """
        output = [f"{dim}^{val}" for dim, val in self.dimension.items() if val != 0.0]
        return " ".join(output) if output else "dimensionless"

    def __add__(self, other_unit):
        """
        Overloads the addition operator (+). Validates dimensional coherence before proceeding.

        :param Unit other_unit: The unit being added.
        :return: The same unit, assuming coherence holds.
        :rtype: Unit
        :raises DimensionalCoherenceError: If dimensions mismatch.
        """
        if self._check_dimensional_coherence(other_unit):
            return self
        raise DimensionalCoherenceError(self, other_unit)

    def __sub__(self, other_unit):
        """
        Overloads the subtraction operator (-). Validates dimensional coherence before proceeding.

        :param Unit other_unit: The unit being subtracted.
        :return: The same unit, assuming coherence holds.
        :rtype: Unit
        :raises DimensionalCoherenceError: If dimensions mismatch.
        """
        if self._check_dimensional_coherence(other_unit):
            return self
        raise DimensionalCoherenceError(self, other_unit)

    def __mul__(self, other_unit):
        """
        Overloads the multiplication operator (*). Sums the SI dimensions to create a derived unit.
        Also acts as a factory for Quantities if multiplied by a float/int (e.g., 5.0 * _m_).

        :param other_unit: The multiplier, either another Unit or a scalar.
        :type other_unit: Unit, float, or int
        :return: A new derived Unit or a new Quantity object.
        :rtype: Unit or Quantity
        :raises UnexpectedValueError: If multiplied by an unsupported type.
        """
        if isinstance(other_unit, self.__class__):
            new_dimension = copy.copy(self.dimension)
            for dim_i, idx_i in other_unit.dimension.items():
                new_dimension[dim_i] += idx_i
            return self.__class__(name="", dimension_dict=new_dimension)

        elif isinstance(other_unit, (float, int)):
            # Local import to prevent circular dependency during initialization
            from .quantity import Quantity

            return Quantity("", self.__class__("", self.dimension), value=other_unit)

        raise UnexpectedValueError("(Unit, float, int)")

    def __truediv__(self, other_unit):
        """
        Overloads the division operator (/). Subtracts the SI dimensions to create a derived unit.
        Also acts as a factory for Quantities if divided by a float/int.

        :param other_unit: The divisor, either another Unit or a scalar.
        :type other_unit: Unit, float, or int
        :return: A new derived Unit or a new Quantity object.
        :rtype: Unit or Quantity
        :raises UnexpectedValueError: If divided by an unsupported type.
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

        raise UnexpectedValueError("(Unit, float, int)")

    def __pow__(self, power):
        """
        Overloads the exponentiation operator (**). Multiplies the SI dimensions by the power.

        :param power: The exponent value.
        :type power: int, float, or Unit (dimensionless only)
        :return: A new derived Unit.
        :rtype: Unit
        :raises UnexpectedValueError: If the exponent type is invalid or has dimensions.
        """
        if isinstance(power, (int, float)):
            new_dimension = {dim: val * power for dim, val in self.dimension.items()}
            return self.__class__(name="", dimension_dict=new_dimension)

        elif isinstance(power, self.__class__) and power._is_dimensionless():
            return self.__class__(name="", dimension_dict=self.dimension)

        raise UnexpectedValueError("(int, float, dimensionless unit)")

    def _check_dimensional_coherence(self, other_unit):
        """
        Checks if two units share exactly the same dimensions, governed by the GLOBAL_CFG.

        :param Unit other_unit: The unit to compare against.
        :return: True if dimensions match or if checking is disabled globally.
        :rtype: bool
        """
        if cfg.DIMENSIONAL_COHERENCE_CHECK:
            return all(
                self.dimension[idx] == other_unit.dimension[idx]
                for idx in self.dimension.keys()
            )
        else:
            # Respects the global verbosity setting to avoid terminal spam
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
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
