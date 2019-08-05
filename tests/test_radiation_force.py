"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import wind, constants

radiation = wind.Qwind(
    M=2e8,
    mdot=0.5,
    spin=0.,
    eta=0.06,
    r_in=200,
    r_out=1600,
    r_min=6.,
    r_max=1400.,
    T=2e6,
    mu=1,
    modes=[],
    rho_shielding=2e8,
    intsteps=1,
    nr=20,
    save_dir="results",
    radiation_mode="Qwind",
    n_cpus=1,
).radiation
