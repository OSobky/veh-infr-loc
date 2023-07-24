""" https://roboticsbackend.com/ros-import-python-module-from-another-package/ """

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[],
    package_dir={'': 'scripts'}
)
setup(**d)
