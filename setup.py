from distutils.core import setup

requirements = [
    "numpy",
    "networkx",
    "pandas",
    "torch",
    "pytorch-lightning",
    "shap",
]

setup(
    author="Erik Hartman",
    author_email="erik.hartman@hotmail.com",
    name='BINN',
    version='0.1',
    packages=['binn'],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=requirements,
    url="https://github.com/InfectionMedicineProteomics/BINN"
)
