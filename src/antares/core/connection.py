# -*- coding: utf-8 -*-

"""
Connection Module.

Defines the generic Connection class for the ANTARES framework.
This class acts as a topological link between two models or ports,
generating equality constraints to represent physical or logical streams.
It leverages the master Model's creation methods to ensure full tensor-safety
across 1D, 2D, and 3D domain couplings.
"""

from . import GLOBAL_CFG as cfg
from .error_definitions import AbsentRequiredObjectError, UnexpectedValueError
from .variable import Variable


class Connection:
    """
    Generic Connection class.
    
    Dynamically inspects ports for auto-connection or explicitly connects
    user-defined variables between two phenomenological models.
    """

    def __init__(
        self, name, source_port, sink_port, source_var_name=None, sink_var_name=None
    ):
        """
        Initializes the topological connection and maps the equality expressions.

        :param str name: Unique name of the connection.
        :param Model source_port: The origin object (source model/port).
        :param Model sink_port: The destination object (sink model/port).
        :param str source_var_name: Specific variable name in the source port. Defaults to None.
        :param str sink_var_name: Specific variable name in the sink port. Defaults to None.
        """
        self.name = name
        self.source_port = source_port
        self.sink_port = sink_port
        
        # Stores the generated mathematical constraints before applying them
        # Format: list of tuples (eq_name, description, symbolic_expression)
        self._pending_links = []

        # Decision logic: Explicit connection vs. Auto-discovery
        if source_var_name and sink_var_name:
            self._generate_specific_equation(source_var_name, sink_var_name)
        else:
            self._generate_auto_connection_equations()

    def _generate_specific_equation(self, source_var_name, sink_var_name):
        """
        Generates a single equality expression for specifically named variables.
        
        :raises AbsentRequiredObjectError: If the specified variables do not exist in the ports.
        :raises UnexpectedValueError: If the attributes are not valid Variable instances.
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

        # Create the constraint expression in residual format (Source - Sink = 0)
        # This natively handles N-Dimensional array subtraction thanks to NumPy
        expr = var_source() - var_sink()

        eq_name = f"{self.name}_{source_var_name}_to_{sink_var_name}"
        desc = f"Explicit link: {self.source_port.name}.{source_var_name} -> {self.sink_port.name}.{sink_var_name}"
        
        self._pending_links.append((eq_name, desc, expr))

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
            print(f"[DEBUG] Explicit connection generated: {eq_name}")

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
                        # Create the constraint expression (Source - Sink = 0)
                        expr = var_source() - var_sink()

                        eq_name = f"{self.name}_{attr_name}_equality"
                        desc = f"Auto topological link for {attr_name}: {self.source_port.name} -> {self.sink_port.name}"
                        
                        self._pending_links.append((eq_name, desc, expr))
                        connected_count += 1

                        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
                            print(f"[DEBUG] Auto-connection generated: {eq_name}")

        # Warning if an auto-connection finds zero matches
        if connected_count == 0 and getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(
                f"Warning: Connection '{self.name}' found no matching variables "
                f"to auto-connect between '{self.source_port.name}' and '{self.sink_port.name}'."
            )

    def apply_to(self, target_model):
        """
        Injects the generated equality constraints into a target model 
        (usually the Master Flowsheet). Delegates the instantiation to the target 
        model to guarantee tensor-flattening safety.

        :param Model target_model: The flowsheet receiving the topological constraints.
        """
        for eq_name, desc, expr in self._pending_links:
            # Delegating to createEquation guarantees N-Dimensional safety
            target_model.createEquation(name=eq_name, description=desc, expr=expr)

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
            print(
                f"[DEBUG] Connection '{self.name}' successfully applied "
                f"{len(self._pending_links)} topological constraints to '{target_model.name}'."
            )