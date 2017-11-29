## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
# This setup.py is only for use with catkin. Remember not to invoke it yourself.
#
#
# Put that script in the top directory of your package, and add this
# to your CMakeLists.txt:
#
#   catkin_python_setup()
#
#
## About Modules
# Standard ROS practice is to place Python modules under
# the src/your_package subdirectory, making the top-level
# module name the same as your package. Python requires 
# that directory to have an __init__.py file, too.
#
#
## About "scripts" argument
# ROS Users should generally not use the scripts argument, as in ROS,
# executables should be executed using rosrun rather than being installed
# to the global bin folder. One way of installing such python scripts is 
# to add the following to the CMakeLists.txt:
#
#    install(PROGRAMS scripts/myscript
#          DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#    )
#
# see also: http://docs.ros.org/indigo/api/catkin/html/howto/format2/installing_python.html
#
# kindly, sysadmin ;)
#
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['simple_biped'],
    scripts=[''],
    package_dir={'': 'src'}
)

setup(**setup_args)
