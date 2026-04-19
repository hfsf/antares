# -*- coding: utf-8 -*-

import datetime

import antares.core.GLOBAL_CFG as cfg

version = "0.1"

_BANNER_PRINTED = False


def print_heading():
    global _BANNER_PRINTED

    # Só imprime se ainda não tiver sido impresso E se a verbosidade não for silenciosa (0)
    if not _BANNER_PRINTED and getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
        # O prefixo 'r' (raw string) é obrigatório para arte ASCII não quebrar com caracteres de escape
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

        # Trava o banner para não aparecer mais durante esta execução do Python
        _BANNER_PRINTED = True
