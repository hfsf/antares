# -*- coding: utf-8 -*-

"""
Global Configuration File for the ANTARES framework.
Contains environment variables, default behaviors, and strictness flags.
Users can modify these values to change the framework's global behavior
without altering the core architectural files.
"""

# =============================================================================
# 1. PHYSICAL & MATHEMATICAL CHECKS
# =============================================================================

# If True, checks for dimensional consistency in all equations.
# Can be set to False for pure mathematical problems (dimensionless)
# or to slightly improve the transpilation setup speed.
DIMENSIONAL_COHERENCE_CHECK = True

# Default unit for the independent variable (usually time) in dynamic simulations.
DEFAULT_TIME_UNIT = "s"

# If True, validates the mathematical closure of the system (Degrees of Freedom == 0)
# before numerical integration. Highly recommended to catch modeling errors early.
PERFORM_DOF_CHECK = True

# =============================================================================
# 2. SIMULATOR & SOLVER DEFAULTS
# =============================================================================

# Default CasADi integrator for dynamic simulations.
# 'idas' is highly recommended for stiff DAE systems. 'cvodes' for simple ODEs.
DEFAULT_INTEGRATOR = "idas"

# Default numerical tolerances for the CasADi integrators and rootfinders.
# Tighten these (e.g., 1e-8 / 1e-10) for highly non-linear or sensitive systems.
DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_ABSOLUTE_TOLERANCE = 1e-8

# =============================================================================
# 3. VERBOSITY & DEBUGGING
# =============================================================================

# Controls the amount of console output during model transpilation and simulation.
# 0 = Silent (Ideal for optimization loops)
# 1 = Standard Progress (Prints setup phases and success messages)
# 2 = Detailed Debug (Prints CasADi matrix sizes and deep transpiler steps)
VERBOSITY_LEVEL = 1

# If True, the framework treats non-fatal warnings as fatal errors.
# Excellent for strict validation in commercial models (e.g., forcing the user
# to provide explicit initial guesses even for algebraic variables).
STRICT_MODE = False

# =============================================================================
# 4. PLOTTING & AESTHETICS
# =============================================================================

# Defines if the framework should use Seaborn to style the plots natively.
USE_SEABORN_STYLE = True

# Seaborn visual themes. Options: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
SEABORN_THEME = "whitegrid"

# Color palettes. Options: 'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind'
SEABORN_PALETTE = "colorblind"

# Scale of elements (fonts, lines). Options: 'paper', 'notebook', 'talk', 'poster'
SEABORN_CONTEXT = "notebook"

# Default properties for exported and rendered figures.
PLOT_FIGSIZE = (10, 6)
PLOT_DPI = 300
PLOT_LINEWIDTH = 2.5

# Specific visual markers for spatial profiles and discrete data points.
PLOT_PRIMARY_COLOR = "#d55e00"  # Vermillion (Colorblind safe default)
PLOT_MARKER = "o"               # Standard circle marker
PLOT_MARKERSIZE = 4             # Standard size of the markers