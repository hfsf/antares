# -*- coding: utf-8 -*-

"""
Define the generic Connection class for the ANTARES framework.
This class acts as a topological link between two models or ports,
generating equality constraints to represent physical or logical streams.
"""

import src.antares.core.GLOBAL_CFG as cfg

from .equation import Equation
from .error_definitions import AbsentRequiredObjectError, UnexpectedValueError
from .variable import Variable


class Connection:
    """
    Generic Connection class.
    It can dynamically inspect ports for auto-connection or explicitly connect
    user-defined variables between two models.
    """

    def __init__(
        self, name, source_port, sink_port, source_var_name=None, sink_var_name=None
    ):
        """
        Connects two ports/models.

        :param str name: Name of the connection.
        :param object source_port: The origin object (source model/port).
        :param object sink_port: The destination object (sink model/port).
        :param str source_var_name: (Optional) Specific variable name in the source port.
        :param str sink_var_name: (Optional) Specific variable name in the sink port.
        """
        self.name = name
        self.source_port = source_port
        self.sink_port = sink_port
        self.equations = []

        # Decision logic: Explicit connection vs. Auto-discovery
        if source_var_name and sink_var_name:
            self._generate_specific_equation(source_var_name, sink_var_name)
        else:
            self._generate_auto_connection_equations()

    def _generate_specific_equation(self, source_var_name, sink_var_name):
        """
        Generates a single equality equation for specifically named variables.
        """
        # Verify if variables exist in their respective ports
        if not hasattr(self.source_port, source_var_name):
            raise AbsentRequiredObjectError(
                f"Variable '{source_var_name}' not found in {self.source_port.name}"
            )

        if not hasattr(self.sink_port, sink_var_name):
            raise AbsentRequiredObjectError(
                f"Variable '{sink_var_name}' not found in {self.sink_port.name}"
            )

        var_source = getattr(self.source_port, source_var_name)
        var_sink = getattr(self.sink_port, sink_var_name)

        # Ensure both attributes are valid Variable instances
        if not (isinstance(var_source, Variable) and isinstance(var_sink, Variable)):
            raise UnexpectedValueError(
                "Both specified attributes for a connection must be instances of Variable."
            )

        # Create the constraint equation in residual format (Source - Sink = 0)
        # The overloaded __sub__ handles the EquationNode generation automatically
        expr = var_source() - var_sink()

        new_eq = Equation(
            name=f"{self.name}_{source_var_name}_to_{sink_var_name}",
            description=f"Explicit link: {self.source_port.name}.{source_var_name} -> {self.sink_port.name}.{sink_var_name}",
            fast_expr=expr,
        )
        self.equations.append(new_eq)

        if cfg.VERBOSITY_LEVEL >= 2:
            print(f"[DEBUG] Explicit connection generated: {new_eq.name}")

    def _generate_auto_connection_equations(self):
        """
        Dynamically inspects the source_port to find Variable attributes
        and matches them with the sink_port by exact name.
        """
        source_attributes = dir(self.source_port)
        connected_count = 0

        for attr_name in source_attributes:
            # Skip private attributes and methods to optimize lookup
            if attr_name.startswith("_"):
                continue

            var_source = getattr(self.source_port, attr_name)

            if isinstance(var_source, Variable):
                if hasattr(self.sink_port, attr_name):
                    var_sink = getattr(self.sink_port, attr_name)

                    if isinstance(var_sink, Variable):
                        # Create the constraint equation (Source - Sink = 0)
                        expr = var_source() - var_sink()

                        new_eq = Equation(
                            name=f"{self.name}_{attr_name}_equality",
                            description=f"Auto topological link for {attr_name}: {self.source_port.name} -> {self.sink_port.name}",
                            fast_expr=expr,
                        )
                        self.equations.append(new_eq)
                        connected_count += 1

                        if cfg.VERBOSITY_LEVEL >= 2:
                            print(f"[DEBUG] Auto-connection generated: {new_eq.name}")

        # Warning if an auto-connection finds zero matches (helps catch typos in large plants)
        if connected_count == 0 and cfg.VERBOSITY_LEVEL >= 1:
            print(
                f"Warning: Connection '{self.name}' found no matching variables to auto-connect between '{self.source_port.name}' and '{self.sink_port.name}'."
            )

    def apply_to(self, target_model):
        """
        Injects the generated equality equations into a target model
        (usually the Master Flowsheet).
        """
        for eq in self.equations:
            target_model.equations[eq.name] = eq
            setattr(target_model, eq.name, eq)

        if cfg.VERBOSITY_LEVEL >= 2:
            print(
                f"[DEBUG] Connection '{self.name}' successfully applied {len(self.equations)} topological constraints to '{target_model.name}'."
            )
