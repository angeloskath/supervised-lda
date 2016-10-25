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

Bash completion
---------------

The [console applications](/console-applications/) do provide an extensive help
but to make things even easier we provide a bash completion script. One can
find this script at the project root under the name `bash_completion.sh` and
install it manually but for standard platforms we also provide the custom make
target `autocomplete`.

### Automatic installation

The following code assumes that CMake was able to find the `bash_completion.d`
folder and will attempt to install the autocomplete script automatically via
`make autocomplete`.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
sudo make install
sudo make autocomplete
# Reopen the shell or source /etc/bash_completion.d/ldaplusplus
lda [Tab]
lda tra
lda tra[Tab][Tab]
train    transform
lda train --[Tab][Tab]
--compute_likelihood  --e_step_tolerance    --initialize_seeded   --random_state        --workers
--continue            --help                --iterations          --snapshot_every      
--e_step_iterations   --initialize_random   --quiet               --topics
```

### Manual installation

The following commands can be used to install manually the autocomplete script
in any UNIX like environment (Mac OSX for instance). We assume that the current
directory is the project root.

```bash
mkdir -p ~/bin
cp bash_completion.sh ~/bin/ldaplusplus_completion.sh
echo >>~/.bash_profile
echo "# LDA++ autocomplete" >>~/.bash_profile
echo "source \${HOME}/bin/ldaplusplus_completion.sh" >>~/.bash_profile
source ~/bin/ldaplusplus_completion.sh
# Now we are all set
lda [Tab]
lda tra
lda tra[Tab][Tab]
train    transform
lda train --[Tab][Tab]
--compute_likelihood  --e_step_tolerance    --initialize_seeded   --random_state        --workers
--continue            --help                --iterations          --snapshot_every      
--e_step_iterations   --initialize_random   --quiet               --topics
```
