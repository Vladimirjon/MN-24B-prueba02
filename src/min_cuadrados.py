# -*- coding: utf-8 -*-

"""
Python 3
17 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

from typing import Callable

from src import eliminacion_gaussiana

# ####################################################################
def ajustar_min_cuadrados(
    xs: list,
    ys: list,
    gradiente: list[Callable[[list[float], list[float]], tuple]],
) -> list[float]:
    """Resuelve el sistema de ecuaciones para encontrar los parámetros del método de mínimos cuadrados.
    Plantea el sistema de ecuaciones lineales al reemplazar los valores de ``xs`` y ``ys`` en las derivadas parciales.

    ## Parameters

    ``xs``: lista con los valores de x.
    ``ys``: lista con los valores de y.
    ``gradiente``: lista de funciones que representan las derivadas parciales del modelo.

    ## Return

    ``params``: lista con los parámetros ajustados del modelo.
    """
    n = len(gradiente)
    Ab = np.zeros((n, n + 1))

    for i, der_parcial in enumerate(gradiente):
        assert callable(der_parcial), "Cada derivada parcial debe ser una función."
        Ab[i, :] = der_parcial(xs, ys)

    return list(eliminacion_gaussiana(Ab))