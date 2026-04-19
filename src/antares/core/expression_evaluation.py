# -*- coding: utf-8 -*-

"""
Define the EquationNode class, which represents the nodes of the
Abstract Syntax Tree (AST) for symbolic mathematical expressions.
"""

from .error_definitions import DimensionalCoherenceError, UnexpectedValueError


class EquationNode:
    """
    Definition of an EquationNode (ENODE).
    Represents an object upon which arithmetical operations can be performed
    to dynamically build symbolic equations.
    """

    def __init__(
        self,
        name="",
        symbolic_object=None,
        symbolic_map=None,
        variable_map=None,
        is_linear=True,
        is_nonlinear=False,
        is_differential=False,
        unit_object=None,
        args=None,
        latex_text="",
        repr_symbolic=None,
    ):
        """
        Instantiates an EquationNode.
        """
        self.name = name
        self.symbolic_object = symbolic_object
        self.symbolic_map = symbolic_map if symbolic_map is not None else {}
        self.variable_map = variable_map if variable_map is not None else {}

        self.equation_type = {
            "is_linear": is_linear,
            "is_nonlinear": is_nonlinear,
            "is_differential": is_differential,
        }

        self.unit_object = unit_object
        self.args = args if args is not None else []
        self.latex_text = latex_text
        self.repr_symbolic = repr_symbolic

    def _checkEquationTypePrecedence(self, eq_type_1, eq_type_2):
        """
        Evaluates the resulting equation type when two nodes interact.
        Differentials take absolute precedence, followed by non-linearities.
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
        return str(self.symbolic_object)

    def __repr__(self):
        return str(self.repr_symbolic)

    def __eq__(self, other_obj):
        """
        Deprecated in 0.2:
        Overloads the equality operator (==).
        Instead of evaluating a boolean, it returns a  (LHS, RHS) representing
        the elementary form of the equation. The Equation class later converts this
        into the residual form (LHS - RHS = 0).

        Returns the equation in the residual form (LHS - RHS = 0)
        """
        if isinstance(other_obj, (self.__class__, int, float)):
            return self - other_obj
        else:
            # raise UnexpectedValueError("Expected EquationNode, int, or float.")
            return NotImplemented

    # =========================================================================
    # ARITHMETICAL OPERATORS OVERLOADING
    # =========================================================================
    def __neg__(self):
        """Suporta o operador unário de negação: -x"""
        # Apenas delega para a multiplicação por um float negativo!
        return self * (-1.0)

    def __pos__(self):
        """Suporta o operador unário positivo: +x (apenas por segurança)"""
        return self

    def __add__(self, other_obj):
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}+{other_obj.name})",
                symbolic_object=self.symbolic_object + other_obj.symbolic_object,
                symbolic_map={**self.symbolic_map, **other_obj.symbolic_map},
                variable_map={**self.variable_map, **other_obj.variable_map},
                unit_object=self.unit_object + other_obj.unit_object,
                latex_text=f"({self.latex_text}+{other_obj.latex_text})",
                repr_symbolic=self.repr_symbolic + other_obj.repr_symbolic,
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}+{other_obj})",
                symbolic_object=self.symbolic_object + other_obj,
                symbolic_map={**self.symbolic_map},
                variable_map={**self.variable_map},
                unit_object=self.unit_object,
                latex_text=f"({self.latex_text}+{other_obj})",
                repr_symbolic=self.repr_symbolic + other_obj,
            )
            enode_.equation_type = {**self.equation_type}
            return enode_
        else:
            # raise UnexpectedValueError("(int, float, EquationNode)")
            return NotImplemented

    def __radd__(self, other_obj):
        return self.__add__(other_obj)

    def __sub__(self, other_obj):
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}-{other_obj.name})",
                symbolic_object=self.symbolic_object - other_obj.symbolic_object,
                symbolic_map={**self.symbolic_map, **other_obj.symbolic_map},
                variable_map={**self.variable_map, **other_obj.variable_map},
                unit_object=self.unit_object - other_obj.unit_object,
                latex_text=f"({self.latex_text}-{other_obj.latex_text})",
                repr_symbolic=self.repr_symbolic - other_obj.repr_symbolic,
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}-{other_obj})",
                symbolic_object=self.symbolic_object - other_obj,
                symbolic_map={**self.symbolic_map},
                variable_map={**self.variable_map},
                unit_object=self.unit_object,
                latex_text=f"({self.latex_text}-{other_obj})",
                repr_symbolic=self.repr_symbolic - other_obj,
            )
            enode_.equation_type = {**self.equation_type}
            return enode_
        else:
            # raise UnexpectedValueError("(int, float, EquationNode)")
            return NotImplemented

    def __rsub__(self, other_obj):
        return (self * -1) + other_obj

    def __mul__(self, other_obj):
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}*{other_obj.name})",
                symbolic_object=self.symbolic_object * other_obj.symbolic_object,
                symbolic_map={**self.symbolic_map, **other_obj.symbolic_map},
                variable_map={**self.variable_map, **other_obj.variable_map},
                unit_object=self.unit_object * other_obj.unit_object,
                latex_text=f"({self.latex_text}*{other_obj.latex_text})",
                repr_symbolic=self.repr_symbolic * other_obj.repr_symbolic,
            )
            # Nova lógica: Preserva o fato de ser diferencial e verifica não-linearidade
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            if self.variable_map and other_obj.variable_map:
                enode_.equation_type["is_nonlinear"] = True
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}*{other_obj})",
                symbolic_object=self.symbolic_object * other_obj,
                symbolic_map={**self.symbolic_map},
                variable_map={**self.variable_map},
                unit_object=self.unit_object,
                latex_text=f"({self.latex_text}*{other_obj})",
                repr_symbolic=self.repr_symbolic * other_obj,
            )
            enode_.equation_type = {**self.equation_type}
            return enode_
        else:
            # raise UnexpectedValueError("(int, float, EquationNode)")
            return NotImplemented

    def __rmul__(self, other_obj):
        return self.__mul__(other_obj)

    def __truediv__(self, other_obj):
        if isinstance(other_obj, self.__class__):
            enode_ = self.__class__(
                name=f"({self.name}/{other_obj.name})",
                symbolic_object=self.symbolic_object / other_obj.symbolic_object,
                symbolic_map={**self.symbolic_map, **other_obj.symbolic_map},
                variable_map={**self.variable_map, **other_obj.variable_map},
                unit_object=self.unit_object / other_obj.unit_object,
                latex_text=f"\\frac{{{self.latex_text}}}{{{other_obj.latex_text}}}",
                repr_symbolic=self.repr_symbolic / other_obj.repr_symbolic,
            )
            # Nova lógica: Preserva o fato de ser diferencial e verifica não-linearidade
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type, other_obj.equation_type
            )
            if self.variable_map and other_obj.variable_map:
                enode_.equation_type["is_nonlinear"] = True
            return enode_

        elif isinstance(other_obj, (int, float)):
            enode_ = self.__class__(
                name=f"({self.name}/{other_obj})",
                symbolic_object=self.symbolic_object / other_obj,
                symbolic_map={**self.symbolic_map},
                variable_map={**self.variable_map},
                unit_object=self.unit_object,
                latex_text=f"\\frac{{{self.latex_text}}}{{{other_obj}}}",
                repr_symbolic=self.repr_symbolic / other_obj,
            )
            enode_.equation_type = {**self.equation_type}
            return enode_
        else:
            # raise UnexpectedValueError("(int, float, EquationNode)")
            return NotImplemented

    def __rtruediv__(self, other_obj):
        return (self ** (-1)) * other_obj

    def __pow__(self, other_obj):
        if isinstance(other_obj, self.__class__):
            if other_obj.unit_object._is_dimensionless():
                enode_ = self.__class__(
                    name=f"({self.name}^{other_obj.name})",
                    symbolic_object=self.symbolic_object**other_obj.symbolic_object,
                    symbolic_map={**self.symbolic_map, **other_obj.symbolic_map},
                    variable_map={**self.variable_map, **other_obj.variable_map},
                    unit_object=self.unit_object**other_obj.unit_object,
                    latex_text=f"({self.latex_text}^{other_obj.latex_text})",
                    repr_symbolic=self.repr_symbolic**other_obj.repr_symbolic,
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
                symbolic_map={**self.symbolic_map},
                variable_map={**self.variable_map},
                unit_object=self.unit_object**other_obj,
                latex_text=f"({self.latex_text}^{other_obj})",
                repr_symbolic=self.repr_symbolic**other_obj,
            )
            enode_.equation_type = self._checkEquationTypePrecedence(
                self.equation_type,
                {"is_linear": False, "is_nonlinear": True, "is_differential": False},
            )
            return enode_

        raise UnexpectedValueError("(int, float, EquationNode)")
