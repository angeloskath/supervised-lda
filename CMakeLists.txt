cmake_minimum_required(VERSION 2.8.9)
project(LDA++)

# This CMakeLists file is structured as follows
#
# 1. Build the ldaplusplus library (as a shared library by default).
#    The sources are explicitly defined in the ${SOURCES} variable and the
#    target is ldaplusplus.
# 2. Build the slda executable linking it with the ldaplusplus library and the
#    rest of the required libraries
# 3. Set up the install target so that the library can be installed and used by
#    others
# 4. Build the tests in a single executable `test_all`. The test sources are
#    explicitly defined in the ${TEST_FILES} variable. A target `check` is
#    also created tha allows building and running all the tests in a single
#    step
# 5. Build the benchmarks.
#
# Example usage of this file
#
#     $ mkdir build
#     $ cd build
#     $ cmake -DCMAKE_BUILD_TYPE=Release ..
#     $ make -j4
#     $ make check
#     $ sudo make install

# Set the compiler flags
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

# Set options for the compilation
option(BUILD_SHARED_LIBS "Build library as a shared object" ON)

# Bring the headers into the project
include_directories(include)

# And add the sources explicitly, see 
# http://stackoverflow.com/questions/1027247/best-way-to-specify-sourcefiles-in-cmake
# for an explanation (yes stackoverflow the very credible software authority)
set(SOURCES
    src/ldaplusplus/Document.cpp
    src/ldaplusplus/em/AbstractEStep.cpp
    src/ldaplusplus/em/CorrespondenceSupervisedEStep.cpp
    src/ldaplusplus/em/CorrespondenceSupervisedMStep.cpp
    src/ldaplusplus/em/FastSupervisedEStep.cpp
    src/ldaplusplus/em/MultinomialSupervisedEStep.cpp
    src/ldaplusplus/em/MultinomialSupervisedMStep.cpp
    src/ldaplusplus/em/OnlineSupervisedMStep.cpp
    src/ldaplusplus/em/SecondOrderSupervisedMStep.cpp
    src/ldaplusplus/em/SemiSupervisedEStep.cpp
    src/ldaplusplus/em/SemiSupervisedMStep.cpp
    src/ldaplusplus/em/SupervisedEStep.cpp
    src/ldaplusplus/em/SupervisedMStep.cpp
    src/ldaplusplus/em/UnsupervisedEStep.cpp
    src/ldaplusplus/em/UnsupervisedMStep.cpp
    src/ldaplusplus/e_step_utils.cpp
    src/ldaplusplus/events/Events.cpp
    src/ldaplusplus/LDABuilder.cpp
    src/ldaplusplus/LDA.cpp
    src/ldaplusplus/optimization/MultinomialLogisticRegression.cpp
    src/ldaplusplus/optimization/SecondOrderLogisticRegressionApproximation.cpp
)

# Generate a shared and static library from the sources
add_library(ldaplusplus ${SOURCES})
# Actually removing the SHARED/STATIC part allows for choosing using the
# standard BUILD_SHARED_LIBS=ON
#
# add_library(ldaplusplus SHARED ${SOURCES})
# add_library(ldaplusplus_static STATIC ${SOURCES})
#
# set the name of the static library to equal the shared one this will pose
# problems (so I've heard :-) not sure) for windows builds
# set_target_properties(ldaplusplus_static PROPERTIES OUTPUT_NAME ldaplusplus)

# Create the slda executable
add_executable(slda slda.cpp)
target_link_libraries(slda m)
target_link_libraries(slda pthread)
target_link_libraries(slda docopt)
target_link_libraries(slda ldaplusplus)

# Set the installation rules
install(TARGETS slda ldaplusplus
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/ldaplusplus
        PATTERN "test" EXCLUDE)

# Build the tests
# We need to explicitly define a test the same way we explicitly define a
# source file
set(TEST_FILES
    test/test_approximate_supervised_expectation_step.cpp
    test/test_corpus.cpp
    test/test_correspondence_supervised_expectation_step.cpp
    test/test_correspondence_supervised_maximization_step.cpp
    test/test_expectation_step.cpp
    test/test_fit.cpp
    test/test_maximization_step.cpp
    test/test_mlr.cpp
    test/test_multinomial_supervised_expectation_step.cpp
    test/test_multinomial_supervised_maximization_step.cpp
    test/test_numpy_data.cpp
    test/test_online_maximization_step.cpp
    test/test_second_order_mlr_approximation.cpp
)
# We exclude the test_all target from all so it is only built when requested
add_executable(test_all EXCLUDE_FROM_ALL ${TEST_FILES})
target_link_libraries(test_all gtest)
target_link_libraries(test_all gtest_main)
target_link_libraries(test_all m)
target_link_libraries(test_all pthread)
target_link_libraries(test_all ldaplusplus)
# We will also add a custom target check that runs the tests
# The || : below is so that make won't complain in case of failed tests. We
# have gtest complaining in color and style no need for another whiner.
add_custom_target(check ./test_all || : DEPENDS test_all)

# Build the benchmarks
set(BENCH_FILES
    bench/bench_compute_approximate_phi.cpp
    bench/bench_compute_h.cpp
    bench/bench_compute_unsupervised_phi.cpp
    bench/bench_compute_supervised_phi_gamma.cpp
)
foreach(BENCH_FILE ${BENCH_FILES})
    get_filename_component(BENCH_TARGET ${BENCH_FILE} NAME_WE)
    set(BENCH_TARGETS ${BENCH_TARGETS} ${BENCH_TARGET})
    add_executable(${BENCH_TARGET} EXCLUDE_FROM_ALL ${BENCH_FILE})
    target_link_libraries(${BENCH_TARGET} ldaplusplus)
    target_link_libraries(${BENCH_TARGET} pthread)
endforeach(BENCH_FILE)
add_custom_target(bench : DEPENDS ${BENCH_TARGETS})
