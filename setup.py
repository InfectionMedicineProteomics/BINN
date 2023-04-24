from distutils.core import setup

requirements = [
    "numpy",
    "networkx",
    "pandas",
    "torch",
    "pytorch-lightning",
    "shap",
    "matplotlib",
    "plotly"
]

setup(
    author="Erik Hartman",
    author_email="erik.hartman@hotmail.com",
    name='binn',
    version='0.0.2',
    packages=['binn'],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=requirements,
    url="https://github.com/InfectionMedicineProteomics/BINN"
)
