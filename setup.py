from setuptools import setup, find_packages

setup(
    name="pyse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "librosa"],
    author="Tanay Poddar",
    description="A brief description of pyse",
    python_requires=">=3.7",
)