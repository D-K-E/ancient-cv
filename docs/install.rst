#############
Installation
#############

:code:`conda` is the main package manager used in the project.

To install the dependencies related to project, launch the following from the
top level directory of the project:

:code:`conda create --name ancient-cv --file objects-ancient-spec-file.txt`

This will create a :code:`conda` environment including all the dependencies
related to project.

Also make sure that you also include the submodules with:

:code:`git submodule update --init --recursive`

Submodules usage might change in the future.
