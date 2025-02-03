from distutils.core import setup
from setuptools import find_packages

requirements = [
    "numpy",
    "networkx",
    "pandas",
    "torch",
    "shap",
    "matplotlib",
]

setup(
    author="Erik Hartman",
    author_email="erik.hartman@hotmail.com",
    name="binn",
    version="0.1.0",
    packages=find_packages(),
    license="MIT",
    long_description=open("README.md").read(),
    install_requires=requirements,
    url="https://github.com/InfectionMedicineProteomics/BINN",
    include_package_data=True,
)
