Installation
============

We use [CMake](https://cmake.org/) to build LDA++. We have tested it on Ubuntu
14.04 and 16.04 as well as Cent OS 6.5 . LDA++ when built creates the following
targets.

- *ldaplusplus.(so|a|dll|lib)* depending on build options
- *slda*, *lda* and *fslda* that build the console applications
- *check* that builds and runs the tests
- *bench* that builds a couple of benchmark applications

The first 4 are built by default. The build system checks the dependencies and
enables the above targets depending on the availability of the dependencies.
*ldaplusplus.(so|a|dll|lib)* is the minimum that will ever be built.

Library dependencies
--------------------

LDA++ depends strongly on
**[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)** for all
matrix and vector mathematics. It depends on
**[Docopt](https://github.com/docopt/docopt.cpp)** for parsing command line
arguments in all console applications. And finally it depends on
**[GTest](https://github.com/google/googletest)** for compiling and running the
tests. **C++11** is also required as well as some kind of threads supported by
the standard library.

If **Eigen** is missing nothing will be built. If **Docopt** is missing the
library will be built but the console applications will not and if **GTest** is
missing the tests will not be built.

Building with CMake
-------------------

One can probably run CMake using a GUI or a different shell but the following
has been tested in bash.

```bash
# Create a build directory
mkdir build && cd build
# this will build with optimizations on ready for use
cmake -DCMAKE_BUILD_TYPE=Release ..
# this is what we use for debugging with gdb but one could even try 'Debug'
cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
# this can be used to install the library in a specific directory
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release ..
# Build the library and the console applications
make
# Install the library, the console applications and the header files
# (might require sudo or root access)
make install
# Build and run the tests
make check
# Build the benchmarks (they must be run manually)
make bench
```
