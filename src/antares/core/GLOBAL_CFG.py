# -*- coding: utf-8 -*-

"""
Global Configuration File for the ANTARES framework.
Contains environment variables, default behaviors, and strictness flags.
Users can modify these values to change the framework's global behavior
without altering the core architectural files.

V4 UPDATE: Removed legacy SymPy chunking configs and added Linear Solver delegation.
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

# Defines the default linear solver for the Newton-Raphson iterations inside
# the integrators (IDAS) and rootfinders (KINSOL).
# Options:
# - "direct": Uses the CasADi native sparse direct solver (csparse).
#             Extremely fast for 1D/2D meshes and moderately sized 3D meshes (N < 50k).
# - "iterative": Uses Krylov subspace methods (GMRES via SUNDIALS) without exact Jacobian.
#                Mandatory for massive 3D meshes (> 100k nodes) to prevent RAM exhaustion.
DEFAULT_LINEAR_SOLVER = "direct"

# If True, CasADi will translate the mathematical graph into pure C code,
# compile it in the background using GCC/Clang, and inject the shared library
# into the solver. This yields maximum execution speed but requires a C compiler
# to be installed and available in the system PATH.
USE_C_CODE_COMPILATION = False

# Define C compilation optimization. Set to "aggressive" for maximum optimization
# and slow compilation time, or "basic" for default optimization for fast compilation
# time, using RAM to store temporary files during compilation
C_COMPILATION_OPTIMIZATION_LEVEL = "basic"  # aggressive

# Keep the files related to C compilation after simulation. It is useful
# for generated code depuration
KEEP_TEMPORARY_COMPILATION_FILES = False

# Default numerical tolerances for the CasADi integrators and rootfinders.
# For lumped (0D) systems, 1e-6/1e-8 is standard.
# For heavily distributed 3D PDEs, relaxing to 1e-4/1e-5 is recommended to drastically
# reduce simulation time without meaningful loss of physical accuracy.
DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_ABSOLUTE_TOLERANCE = 1e-8

# Debug flag for extra verbosity for the rootfinder solver
ROOTFINDER_SOLVER_DEBUG_LEVEL = 0

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

# Controls the exhibition of loading bars for generation of the model equation and
# their transpilation, in order to provide some visual output for the user. It is
# particularly useful for large simulations.
SHOW_LOADING_BARS = True

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
PLOT_MARKER = "o"  # Standard circle marker
PLOT_MARKERSIZE = 8  # Standard size of the markers

# Configurations for heatmaps (2D) and 3D slicing (3D)
PLOT_COLORMAP_HEAT = "inferno"
PLOT_COLORMAP_MASS = "viridis"
PLOT_CONTOUR_LEVELS = 100
