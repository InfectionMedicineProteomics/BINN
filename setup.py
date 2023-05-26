from distutils.core import setup

requirements = [
    "numpy",
    "networkx",
    "pandas<=1.5.3",
    "torch<=1.13",
    "pytorch-lightning<=1.9.5",
    "shap",
    "matplotlib",
    "plotly",
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
