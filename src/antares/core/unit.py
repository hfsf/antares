# -*- coding: utf-8 -*-

"""
Unit Module (V5 Native CasADi Architecture).

Defines the Unit class for the ANTARES framework.
Handles dimensional tracking, parsing of string-based unit definitions,
and dimensional coherence analysis across mathematical equations.
In the V5 architecture, this module acts as the strict physical guardian,
incorporating scaling factors for automatic SI normalization before
delegating operations to the CasADi C++ computational graph.
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
    Automatically catches empty strings or dimensionless markers ("1", "-")
    returning a pure dimensionless Unit, bypassing the algebraic evaluation.

    :param str unit_definition: The sanitized string representation of the unit algebra.
    :return: The corresponding derived Unit object.
    :rtype: Unit
    """
    # Intercept dimensionsell units
    if unit_definition is None or str(unit_definition).strip() in ["", "1", "-"]:
        # Instancia e retorna uma unidade zerada (adimensional)
        return Unit("dimensionless", {})

    sanitized = _sanitizeUnitDef(unit_definition)
    # Safe evaluation using strictly the predefined units mapping
    return eval(sanitized, {"__builtins__": None}, predef_units_map)


class Unit:
    """
    Unit class definition.
    Tracks the dimensional index for each SI dimension and overloads mathematical
    operators (*, /, **) to create derived units dynamically. It maintains a
    scaling factor to facilitate automatic conversion between derived units
    and their SI counterparts.
    """

    def __init__(self, name, dimension_dict, description="", scaling_factor=1.0):
        """
        Initializes the Unit object.

        :param str name: Internal name of the present unit.
        :param dimension_dict: Dictionary containing dimensions, a string expression, or another Unit.
        :type dimension_dict: dict, str, or Unit
        :param str description: Short physical description of the unit.
        :param float scaling_factor: Multiplier to convert this unit to its SI base equivalent.
        """
        self.name = name
        self.description = description
        self.scaling_factor = float(scaling_factor)
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
        Overloads the multiplication operator (*). Sums the SI dimensions and
        multiplies the scaling factors. Also acts as a factory for Quantities
        if multiplied by a scalar.

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

            return self.__class__(
                name="",
                dimension_dict=new_dimension,
                scaling_factor=self.scaling_factor * other_unit.scaling_factor,
            )

        elif isinstance(other_unit, (float, int)):
            from .quantity import Quantity

            return Quantity("", self, value=float(other_unit))

        raise UnexpectedValueError("(Unit, float, int)")

    def __truediv__(self, other_unit):
        """
        Overloads the division operator (/). Subtracts the SI dimensions and
        divides the scaling factors. Also acts as a factory for Quantities
        if divided by a scalar.

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

            return self.__class__(
                name="",
                dimension_dict=new_dimension,
                scaling_factor=self.scaling_factor / other_unit.scaling_factor,
            )

        elif isinstance(other_unit, (float, int)):
            from .quantity import Quantity

            return Quantity("", self, value=1.0 / float(other_unit))

        raise UnexpectedValueError("(Unit, float, int)")

    def __rtruediv__(self, other):
        """
        Handles Right True Division to support scalar numerators (e.g., '1/s').

        When the Python `eval()` engine encounters a division where the left
        operand is a native scalar (like the integer 1) and the right operand
        is a Unit object, it invokes this magic method. It safely clones the
        current unit and inverts all its dimensional exponents.

        :param int|float other: The scalar numerator (typically 1).
        :return: A new Unit object with inverted dimensions.
        :rtype: Unit
        :raises TypeError: If the numerator is not a supported scalar type.
        """
        if isinstance(other, (int, float)):
            import copy

            # Create a deep clone to avoid mutating the base unit (e.g., 's')
            new_unit = copy.deepcopy(self)

            # Invert the sign of all dimensional exponents
            new_unit.dimension = {
                base: -exponent for base, exponent in self.dimension.items()
            }

            # Update the unit's string representation
            new_unit.name = f"{other}/{self.name}"

            return new_unit

        return NotImplemented

    def __pow__(self, power):
        """
        Overloads the exponentiation operator (**). Multiplies the SI dimensions
        and scaling factor exponent by the power.

        :param power: The exponent value.
        :type power: int, float, or Unit (dimensionless only)
        :return: A new derived Unit.
        :rtype: Unit
        :raises UnexpectedValueError: If the exponent type is invalid or has dimensions.
        """
        if isinstance(power, (int, float)):
            new_dimension = {dim: val * power for dim, val in self.dimension.items()}
            return self.__class__(
                name="",
                dimension_dict=new_dimension,
                scaling_factor=self.scaling_factor ** float(power),
            )

        elif isinstance(power, self.__class__) and power._is_dimensionless():
            return self.__class__(
                name="",
                dimension_dict=self.dimension,
                scaling_factor=self.scaling_factor,
            )

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
            if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
                print("Warning: Skipping dimensional coherence test.")
            return True


# =============================================================================
# PREDEFINED BASE SI UNITS
# =============================================================================
_kg_ = Unit("kilogram", {"kg": 1.0})
_m_ = Unit("meter", {"m": 1.0})
_s_ = Unit("second", {"s": 1.0})
_A_ = Unit("Ampere", {"A": 1.0})
_K_ = Unit("Kelvin", {"K": 1.0})
_mol_ = Unit("mol", {"mol": 1.0})

# =============================================================================
# DERIVED SCIENTIFIC & ENGINEERING UNITS (WITH SCALING FACTORS)
# =============================================================================

# Length
_km_ = Unit("kilometer", _m_, scaling_factor=1000.0)
_cm_ = Unit("centimeter", _m_, scaling_factor=0.01)
_mm_ = Unit("millimeter", _m_, scaling_factor=0.001)
_um_ = Unit("micrometer", _m_, scaling_factor=1e-6)

# Mass
_g_ = Unit("gram", _kg_, scaling_factor=0.001)
_mg_ = Unit("milligram", _kg_, scaling_factor=1e-6)
_ton_ = Unit("tonne", _kg_, scaling_factor=1000.0)

# Time
_min_ = Unit("minute", _s_, scaling_factor=60.0)
_h_ = Unit("hour", _s_, scaling_factor=3600.0)
_day_ = Unit("day", _s_, scaling_factor=86400.0)

# Volume
_L_ = Unit("liter", _m_**3, scaling_factor=0.001)
_mL_ = Unit("milliliter", _m_**3, scaling_factor=1e-6)

# Molar Amount
_mmol_ = Unit("milimol", _mol_, scaling_factor=1e-3)
_kmol_ = Unit("kilomol", _mol_, scaling_factor=1000.0)
_Mmol_ = Unit("megamol", _mol_, scaling_factor=1e6)

# Pressure (Base: Pa = kg/(m*s^2))
_Pa_ = Unit("Pascal", {"kg": 1.0, "m": -1.0, "s": -2.0})
_kPa_ = Unit("kilopascal", _Pa_, scaling_factor=1000.0)
_bar_ = Unit("bar", _Pa_, scaling_factor=100000.0)
_atm_ = Unit("atmosphere", _Pa_, scaling_factor=101325.0)

# Energy & Power
_J_ = Unit("Joule", {"kg": 1.0, "m": 2.0, "s": -2.0})
_kJ_ = Unit("kilojoule", _J_, scaling_factor=1000.0)
_W_ = Unit("Watt", _J_ / _s_)
_kW_ = Unit("kilowatt", _W_, scaling_factor=1000.0)
_MW_ = Unit("megawatt", _W_, scaling_factor=1000000.0)
_GW_ = Unit("gigawatt", _W_, scaling_factor=1000000000.0)

# Force
_N_ = Unit("Newton", {"kg": 1.0, "m": 1.0, "s": -2.0})
_kN_ = Unit("kilonewton", _N_, scaling_factor=1000.0)

# Viscosity (Dynamic: Pa*s = kg/(m*s))
_Pas_ = _Pa_ * _s_
_cP_ = Unit("centipoise", _Pas_, scaling_factor=0.001)

# Mapping dictionary for parser and factories
predef_units_map = {
    # Dimensionless unit definition
    "": _m_ / _m_,
    " ": _m_ / _m_,
    "-": _m_ / _m_,
    " ": _m_ / _m_,
    # SI Bases
    "kg": _kg_,
    "m": _m_,
    "s": _s_,
    "A": _A_,
    "K": _K_,
    "mol": _mol_,
    # Length
    "km": _km_,
    "cm": _cm_,
    "mm": _mm_,
    "um": _um_,
    # Mass
    "g": _g_,
    "mg": _mg_,
    "ton": _ton_,
    # Time
    "min": _min_,
    "h": _h_,
    "day": _day_,
    # Volume
    "L": _L_,
    "mL": _mL_,
    # Molar
    "mmol": _mmol_,
    "kmol": _kmol_,
    "Mmol": _Mmol_,
    # Pressure
    "Pa": _Pa_,
    "kPa": _kPa_,
    "bar": _bar_,
    "atm": _atm_,
    # Energy/Power
    "J": _J_,
    "kJ": _kJ_,
    "W": _W_,
    "kW": _kW_,
    "MW": _MW_,
    "GW": _GW_,
    # Force
    "N": _N_,
    "kN": _kN_,
    # Viscosity
    "cP": _cP_,
}
