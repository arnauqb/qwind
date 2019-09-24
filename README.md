Qwind: A non-hydrodynamical model of AGN line-driven winds.
###########################################################

Qwind is a code that aims to simulate the launching and acceleration phase of line-driven winds in the context of AGN accretion discs. To do that, we model the wind as a set of streamlines originating on the surface of the AGN accretion disc, and we evolve them following their equation of motion, given by the balance between radiative and gravitational force.

Code summary
============

Please refer to (paper) for a detailed physical explanation of the model. The code is divided into three main classes: <em>wind</em>, <em>radiation</em>, and <em>streamline</em>. The <em>wind</em> class is the main class and the one that is used to initialise the model. It contains the global information of the accretion disc such as accretion rate, density profile, etc. The radiation processes are handled in the <em>radiation</em> class. There are multiple implementations of the radiation class available, for different treatments of the radiation field. A model can be initialised with a particular radiation model by changing the ``radiation_mode`` argument when initialising the wind class. Lastly, the <em>streamline</em> class represents a single streamline, and contains the Euler iterator that solves its equation of motion. It also stores all the physical information of the streamline  at every point it travels, so that it is easy to analyse and debug.

Getting started
===============

Prerequisites
-------------

The required Python modules are

``
scipy
numba
``

Installing
----------

The code can be installed with pip

```
pip install qwind
```

Running the tests
=================

The tests can be easily run by installing pytest and running

```
cd tests
pytest
```

License
=======

The project is licensed under the GPL3 license. See LICENSE.md file for details.
Please refer to (link to Notebook) for a quick start on how to use the code.
























