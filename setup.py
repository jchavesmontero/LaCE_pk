#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Lya power spectrum routines"
version="1.0"

setup(name="lace_pk",
    version=version,
    description=description,
    url="https://github.com/jchavesmontero/LaCE_pk",
    author="Jonas Chaves-Montero, Andreu Font-Ribera et al.",
    author_email="jchaves@ifae.es",
    packages=find_packages(),
    )
