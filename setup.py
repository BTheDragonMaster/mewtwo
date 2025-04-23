from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'MEWTWO: mRNA Expression Wizard 2'
LONG_DESCRIPTION = "3'UTR analysis software"

setup(
    name="mewtwo",
    version=VERSION,
    author="Barbara Terlouw",
    author_email="barbara.terlouw@wur.nl",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude="build"))
