# -*- coding: utf-8 -*-

"""
Expression Evaluation Module (V5 Native CasADi Architecture).

Defines the EquationNode class, which acts as a physical and mathematical
wrapper around CasADi MX symbolic graphs. It enforces dimensional coherence
and tracks equation types (differential vs algebraic) during native C++
graph construction, effectively replacing the legacy SymPy AST.
"""

import casadi as ca

from .error_definitions import DimensionalCoherenceError, UnexpectedValueError


class EquationNode:
    """
    Definition of an EquationNode (ENODE).
    Encapsulates a CasADi MX object and its physical unit, intercepting
    arithmetic operations to validate dimensional coherence before building
    the underlying C++ computational graph.
    """

    def __init__(
        self,
        name="",
        symbolic_object=None,
        is_linear=True,
        is_nonlinear=False,
        is_differential=False,
        unit_object=None,
        latex_text="",
    ):
        """
        Instantiates an EquationNode.

        :param str name: String representation of the current mathematical expression.
        :param casadi.MX symbolic_object: The native CasADi symbolic graph or tensor.
        :param bool is_linear: Flag indicating if the expression remains linear.
        :param bool is_nonlinear: Flag indicating if nonlinearities (e.g., multiplications) occurred.
        :param bool is_differential: Flag indicating if a temporal derivative exists in the tree.
        :param Unit unit_object: The physical unit corresponding to this exact node.
        :param str latex_text: LaTeX representation for rendering and reporting.
        """
        self.name = name
        self.symbolic_object = symbolic_object

        self.equation_type = {
            "is_linear": is_linear,
            "is_nonlinear": is_nonlinear,
            "is_differential": is_differential,
        }

        self.unit_object = unit_object
        self.latex_text = latex_text

    def _checkEquationTypePrecedence(self, eq_type_1, eq_type_2):
        """
        Evaluates the resulting equation type when two nodes interact.
        Differentials take absolute precedence, followed by non-linearities.

        :param dict eq_type_1: Classification dictionary of the LHS node.
        :param dict eq_type_2: Classification dictionary of the RHS node.
        :return: A new dictionary mapping the resulting node classification.
        :rtype: dict
        """
        res = {"is_linear": False, "is_nonlinear": False, "is_differential": False}

        if eq_type_1.get("is_differential") or eq_type_2.get("is_differential"):
            res["is_differential"] = True
        elif eq_type_1.get("is_nonlinear") or eq_type_2.get("is_nonlinear"):
            res["is_nonlinear"] = True
        else:
            res["is_linear"] = True

        return res

    def __str__(self):
        """Returns the string representation of the underlying CasADi graph."""
        return str(self.symbolic_object)

    def __repr__(self):
        """Returns the string representation of the underlying CasADi graph."""
        return str(self.symbolic_object)

    def __eq__(self, other_obj):
        """
        Overloads the equality operator (==) to build the mathematical residual.
        Returns the equation in the residual form (LHS - RHS = 0).
        """
        if isinstance(other_obj, (self.__class__, int, float)):
            return self - other_obj
        return NotImplemented

    # =========================================================================
    # ARITHMETICAL OPERATORS OVERLOADING (Native CasADi + Unit Checks)
    # =========================================================================

    def __neg__(self):
        """Supports the unary negation operator: -x"""
        return self * (-1.0)

    def __pos__(self):
        """Supports the unary positive operator: +x"""
        return self

    def __add__(self, other_obj):
        """Overloads the addition operator (+)."""
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}+{other_obj.name})",
                symbolic_object=self.symbolic_object + other_obj.symbolic_object,
                unit_object=self.unit_object + other_obj.unit_object,
                latex_text=f"({self.latex_text}+{other_obj.latex_text})",
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}+{other_obj})",
                symbolic_object=self.symbolic_object + other_obj,
                unit_object=self.unit_object,
                latex_text=f"({self.latex_text}+{other_obj})",
            )
            enode_.equation_type = {**self.equation_type}
            return enode_

        return NotImplemented

    def __radd__(self, other_obj):
        """Overloads the right-side addition operator."""
        return self.__add__(other_obj)

    def __sub__(self, other_obj):
        """Overloads the subtraction operator (-)."""
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}-{other_obj.name})",
                symbolic_object=self.symbolic_object - other_obj.symbolic_object,
                unit_object=self.unit_object - other_obj.unit_object,
                latex_text=f"({self.latex_text}-{other_obj.latex_text})",
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}-{other_obj})",
                symbolic_object=self.symbolic_object - other_obj,
                unit_object=self.unit_object,
                latex_text=f"({self.latex_text}-{other_obj})",
            )
            enode_.equation_type = {**self.equation_type}
            return enode_

        return NotImplemented

    def __rsub__(self, other_obj):
        """Overloads the right-side subtraction operator."""
        return (self * -1.0) + other_obj

    def __mul__(self, other_obj):
        """Overloads the multiplication operator (*)."""
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}*{other_obj.name})",
                symbolic_object=self.symbolic_object * other_obj.symbolic_object,
                unit_object=self.unit_object * other_obj.unit_object,
                latex_text=f"({self.latex_text}*{other_obj.latex_text})",
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            enode_.equation_type["is_nonlinear"] = True
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}*{other_obj})",
                symbolic_object=self.symbolic_object * other_obj,
                unit_object=self.unit_object,
                latex_text=f"({self.latex_text}*{other_obj})",
            )
            enode_.equation_type = {**self.equation_type}
            return enode_

        return NotImplemented

    def __rmul__(self, other_obj):
        """Overloads the right-side multiplication operator."""
        return self.__mul__(other_obj)

    def __truediv__(self, other_obj):
        """Overloads the true division operator (/)."""
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}/{other_obj.name})",
                symbolic_object=self.symbolic_object / other_obj.symbolic_object,
                unit_object=self.unit_object / other_obj.unit_object,
                latex_text=f"\\frac{{{self.latex_text}}}{{{other_obj.latex_text}}}",
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            enode_.equation_type["is_nonlinear"] = True
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}/{other_obj})",
                symbolic_object=self.symbolic_object / other_obj,
                unit_object=self.unit_object,
                latex_text=f"\\frac{{{self.latex_text}}}{{{other_obj}}}",
            )
            enode_.equation_type = {**self.equation_type}
            return enode_

        return NotImplemented

    def __rtruediv__(self, other_obj):
        """Overloads the right-side true division operator."""
        return (self ** (-1.0)) * other_obj

    def __pow__(self, other_obj):
        """Overloads the exponentiation operator (**)."""
        if isinstance(other_obj, self.__class__):
            if (
                hasattr(other_obj.unit_object, "_is_dimensionless")
                and other_obj.unit_object._is_dimensionless()
            ):
                enode_ = self.__class__(
                    name=f"({self.name}^{other_obj.name})",
                    symbolic_object=self.symbolic_object**other_obj.symbolic_object,
                    unit_object=self.unit_object**other_obj.unit_object,
                    latex_text=f"({self.latex_text}^{{{other_obj.latex_text}}})",
                )
                enode_.equation_type = self._checkEquationTypePrecedence(
                    self.equation_type,
                    {
                        "is_linear": False,
                        "is_nonlinear": True,
                        "is_differential": False,
                    },
                )
                return enode_
            raise DimensionalCoherenceError(other_obj.unit_object, None)

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}^{other_obj})",
                symbolic_object=self.symbolic_object**other_obj,
                unit_object=self.unit_object**other_obj,
                latex_text=f"({self.latex_text}^{{{other_obj}}})",
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type,
                {"is_linear": False, "is_nonlinear": True, "is_differential": False},
            )
            return enode_

        raise UnexpectedValueError(
            "Expected an int, float, or dimensionless EquationNode for exponentiation."
        )
