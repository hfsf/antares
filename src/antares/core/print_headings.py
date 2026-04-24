# -*- coding: utf-8 -*-

"""
Print Headings Module (V5 Native CasADi Architecture).

Handles the execution banner and framework initialization messages.
"""

import datetime

import antares.core.GLOBAL_CFG as cfg

# V5 Architecture Release
version = "v.0.1.5a"

_BANNER_PRINTED = False


def print_heading():
    """
    Outputs the ANTARES ASCII art banner and build metadata to the terminal.
    Execution is guarded by the global verbosity configuration and runs exactly once.
    """
    global _BANNER_PRINTED

    if not _BANNER_PRINTED and getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
        # Raw string ('r') is mandatory to prevent escape character parsing in ASCII art
        print(
            r"""
:::'###::::'##::: ##:'########::::'###::::'########::'########::'######::
::'## ##::: ###:: ##:... ##..::::'## ##::: ##.... ##: ##.....::'##... ##:
:'##:. ##:: ####: ##:::: ##:::::'##:. ##:: ##:::: ##: ##::::::: ##:::..::
'##:::. ##: ## ## ##:::: ##::::'##:::. ##: ########:: ######:::. ######::
 #########: ##. ####:::: ##:::: #########: ##.. ##::: ##...:::::..... ##:
 ##.... ##: ##:. ###:::: ##:::: ##.... ##: ##::. ##:: ##:::::::'##::: ##:
 ##:::: ##: ##::. ##:::: ##:::: ##:::: ##: ##:::. ##: ########:. ######::
..:::::..::..::::..:::::..:::::..:::::..::..:::::..::........:::......:::

    Version: {}
    Date/Time: {}
        """.format(version, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        # Locks the banner so it does not repeat during model composition
        _BANNER_PRINTED = True
