"""
This module is used to store all relevant physical constants.
"""

import astropy.constants as astroconst
from astropy import units as u
import scipy.constants as const
import numpy as np

# basic physical constants #
G = astroconst.G.cgs.value
M_SUN = astroconst.M_sun.cgs.value
C = astroconst.c.cgs.value
H_PLANCK = astroconst.h.cgs.value
M_P = astroconst.m_p.cgs.value
K_B = astroconst.k_B.cgs.value
RYD = u.astrophys.Ry.cgs.scale
SIGMA_SB = astroconst.sigma_sb.cgs.value
SIGMA_T = const.physical_constants['Thomson cross section'][0] * 1e4
YEAR_TO_SEC = u.yr.cgs.scale
PI = np.pi

# useful normalization factors #
IONIZATION_PARAMETER_CRITICAL = 1e5  # / 8.2125
EMISSIVITY_CONSTANT = 4 * PI * M_P * C**3 / SIGMA_T


def convert_units(value, current_unit, new_unit):
    """
    Convenient function to convert units using astropy.

    Args:
        value: numerical value
        current_unit: unit of the current value
        new_unit: target unit

    Returns:
        value in new_unit.
    """
    try:  # make sure value is unitless
        value = value.value
    except:
        value = value

    current = value * current_unit
    new = current.to(new_unit)
    return new.value
