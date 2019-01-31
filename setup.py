# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Qwind',
    version='0.1.0',
    description='Qwind: A Python code to compute AGN UV line-driven winds.',
    long_description=readme,
    author='Arnau Quera-Bofarull',
    author_email='arnau.quera-bofarull@durham.ac.uk',
    url='https://github.com/arnauq/qwind',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

