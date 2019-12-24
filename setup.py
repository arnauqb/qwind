# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from Cython.Build import cythonize


ext_modules = cythonize("qwind/streamline/ida.pyx")

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='qwind',
    version='0.1.0',
    description='Qwind: A Python code to compute AGN UV line-driven winds. ',
    long_description=readme,
    author='Arnau Quera-Bofarull',
    author_email='arnau.quera-bofarull@durham.ac.uk',
    url='https://github.com/arnauq/qwind',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    setup_requires=['pbr'],
    pbr=True,
    ext_modules=ext_modules,
)
