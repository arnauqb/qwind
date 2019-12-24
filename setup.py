# -*- coding: utf-8 -*-
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
import subprocess
import os

#class CustomInstall(install):
#    def run(self):
#        #command = "git clone 'https://github.com/arnauq/qwind"
#        #process = subprocess.Popen(command, shell=True, cwd="qwind")
#        #process.wait()
#        process= subprocess.Popen("make", shell=True, cwd="qwind") 
#        process.wait()
#        install.run(self)

#module = Extension("qwind",
#                   sources = ['qwind/integration/integrand.c'],
#                   extra_compile_args=['-fPIC', '-shared'],
#                   libraries=['gsl', 'gslcblas'])
#

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
#    ext_modules=[module],
    package_data = {'qwind' : ['qwind/integration/qwind_library.so']},
    setup_requires=['pbr'],
    pbr=True,
    #cmdclass={'install' : CustomInstall},
)
