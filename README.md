# ANTARES - v.0.1.5a

###### (Derived from SLOTH project)

![antares_logo](https://github.com/hfsf/antares/blob/master/docs/antares_logo.png?raw=true)

![version](https://img.shields.io/badge/version-0.0a-orange?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.8--3.7-blue?style=for-the-badge)
[![License: GPL](https://img.shields.io/badge/License-GPL-blue.svg?style=for-the-badge)](https://opensource.org/licenses/GPL)
[![Build Status](https://travis-ci.com/hfsf/antares.svg?style=for-the-badge&branch=master)](https://travis-ci.com/hfsf/antares)
[![Documentation Status](https://readthedocs.org/projects/antares/badge/?style=for-the-badge&version=latest)](https://antares.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/00000.svg)](https://zenodo.org/badge/latestdoi/00000)

## ANTARES aims to:

- **Bridge the gap between usability and performance:** Combine the intuitive, object-oriented modeling syntax of **Python/SymPy** (for equation declaration) with the numerical robustness and speed of **CasADi** (for graph-based optimization and JIT compilation).
- **Enable rapid prototyping of Digital Twins:** Allow researchers to move from mathematical formulation to **Dynamic Real-Time Optimization (DRTO)** and **Nonlinear Model Predictive Control (NMPC)** in a few lines of code, bypassing the complexity of industrial suites.
- **Provide a specialized framework for Bioprocesses:** Focus on the specific challenges of biochemical engineering—such as hybrid modeling (AI + First Principles), nonlinear kinetics, and oscillatory regimes—where traditional steady-state simulators fall short.
- **Delegate, don't reinvent:** Focus on solving dynamic Differential-Algebraic Equations (DAE) efficiently, while seamlessly delegating rigorous property calculations to specialized external libraries (e.g., **Cantera**, **Thermo**) when necessary.

### Key Features

- **Equation-Oriented Modeling:** Write mass and energy balances naturally using Python objects.
- **Automatic Differentiation (AD):** Leverages CasADi's algorithmic differentiation to provide exact Jacobians and Hessians, ensuring convergence in stiff and unstable systems (e.g., continuous fermentation with bifurcations).
- **High-Performance Backend:** Automatically transpiles high-level Python models into C-code via CasADi for blazing-fast execution.
- **Optimization-Ready:** Native support for parameter estimation, optimal control, and sensitivity analysis without redefining the model.

## Warning!

This software is not mature yeat to receive contributions. Please, wait until all the core functionalities are up and running

## Contact

For further questions, suggestions and collaborations contact me through: <hanniel.freitas@ifrn.edu.br>.

<!---
.. |cantera| image:: https://cantera.org/assets/img/cantera-logo.png
    :target: https://cantera.org
    :alt: cantera logo
    :width: 675px
    :align: middle

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.170284.svg
   :target: https://doi.org/10.5281/zenodo.1174508

.. |codecov| image:: https://img.shields.io/codecov/c/github/Cantera/cantera/master.svg
   :target: https://codecov.io/gh/Cantera/cantera?branch=master

.. |release| image:: https://img.shields.io/github/release/cantera/cantera.svg
   :target: https://github.com/Cantera/cantera/releases
   :alt: GitHub release
--->
