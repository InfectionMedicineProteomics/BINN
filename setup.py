from distutils.core import setup

requirements = [
    "numpy",
    "networkx",
    "pandas",
    "torch",
    "lightning",
    "shap",
    "matplotlib",
    "plotly",
    "nbformat>=4.2.0",
    "kaleido",
]

setup(
    author="Erik Hartman",
    author_email="erik.hartman@hotmail.com",
    name="binn",
    version="0.0.2",
    packages=["binn"],
    license="MIT",
    long_description=open("README.md").read(),
    install_requires=requirements,
    url="https://github.com/InfectionMedicineProteomics/BINN",
)
