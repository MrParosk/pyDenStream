from setuptools import setup

with open("requirements.txt") as file:
    required = file.read().splitlines()


setup(
    name="denstream",
    version="0.1",
    description="Implementation of the DenStream algorithm",
    author="MrParosk",
    author_email="TBC",
    packages=["denstream"],
    install_requires=required,
    setup_requires=["wheel"],
    python_requires=">=3.8, <3.12",
)
