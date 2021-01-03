from setuptools import setup


with open('requirements.txt') as file:
    required = file.read().splitlines()


setup(name='pyDenStream',
      version='0.1',
      description='Implementation of the DenStream algorithm',
      author='MrParosk',
      author_email='TBC',
      packages=['pyDenStream'],
      install_requires=required,
      )
