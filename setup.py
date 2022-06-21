from setuptools import setup, find_packages

setup(name='GSFS',
      version='1.0.0',
      description='Graph Search Feature Selection module',
      url='https://github.com/pwolosz/MCTS',
      author='pwolosz',
      author_email='wolosz.patryk2@gmail.com',
      license='GNU General Public License v3.0',
      packages=find_packages(),
      install_requires=[
	'numpy==1.22.0',
	'pandas==0.23.4',
	'scikit-learn==0.20.2',
	'graphviz==0.8.4',
	])