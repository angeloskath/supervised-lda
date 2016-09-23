cmake_minimum_required(VERSION 3.0.0)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

# Set options for the compilation
option(BUILD_SHARED_LIBS "Build library as a shared object" ON)

# Search for the following dependencies
# - Eigen
# - Threads
# - Docopt (if not available slda won't be built)
# - GTest (if not available tests won't be compiled)
#
# The following variables will be defined and used where needed
# - EIGEN_INCLUDE_DIRS
# - CMAKE_THREAD_LIBS_INIT
# - DOCOPT_INCLUDE_DIRS
# - DOCOPT_LIBRARIES
# - GTEST_INCLUDE_DIRS
# - GTEST_BOTH_LIBRARIES
#
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(Eigen REQUIRED)
find_package(Threads REQUIRED)
find_package(Docopt)
find_package(GTest)

# Bring the headers into the project
include_directories(
    include
    ${EIGEN_INCLUDE_DIRS}
    ${DOCOPT_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

# And add the sources explicitly, see 
# http://stackoverflow.com/questions/1027247/best-way-to-specify-sourcefiles-in-cmake
# for an explanation (yes stackoverflow the very credible software authority)
set(SOURCES
    src/ldaplusplus/Document.cpp
    src/ldaplusplus/em/AbstractEStep.cpp
    src/ldaplusplus/em/CorrespondenceSupervisedEStep.cpp
    src/ldaplusplus/em/CorrespondenceSupervisedMStep.cpp
    src/ldaplusplus/em/FastOnlineSupervisedMStep.cpp
    src/ldaplusplus/em/FastSupervisedEStep.cpp
    src/ldaplusplus/em/FastSupervisedMStep.cpp
    src/ldaplusplus/em/MultinomialSupervisedEStep.cpp
    src/ldaplusplus/em/MultinomialSupervisedMStep.cpp
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
set(TARGETS_FOR_INSTALL ${TARGETS_FOR_INSTALL} ldaplusplus)
add_library(ldaplusplus ${SOURCES})
target_link_libraries(ldaplusplus m)
target_link_libraries(ldaplusplus ${CMAKE_THREAD_LIBS_INIT})
# Actually removing the SHARED/STATIC part allows for choosing using the
# standard BUILD_SHARED_LIBS=ON
#
# add_library(ldaplusplus SHARED ${SOURCES})
# add_library(ldaplusplus_static STATIC ${SOURCES})
#
# set the name of the static library to equal the shared one this will pose
# problems (so I've heard :-) not sure) for windows builds
# set_target_properties(ldaplusplus_static PROPERTIES OUTPUT_NAME ldaplusplus)

# Create command application for unsupervised_lda executable
if (DOCOPT_FOUND)
    set(APPLICATION_SOURCES
        src/applications/lda_io.cpp
        src/applications/EpochProgress.cpp
        src/applications/ExpectationProgress.cpp
        src/applications/MaximizationProgress.cpp
        src/applications/SnapshotEvery.cpp
        src/applications/utils.cpp
    )
    add_library(applications OBJECT ${APPLICATION_SOURCES})

    set(TARGETS_FOR_INSTALL ${TARGETS_FOR_INSTALL} lda)
    add_executable(lda src/applications/lda.cpp $<TARGET_OBJECTS:applications>)
    target_link_libraries(lda ${DOCOPT_LIBRARIES})
    target_link_libraries(lda ldaplusplus)

    set(TARGETS_FOR_INSTALL ${TARGETS_FOR_INSTALL} slda)
    add_executable(slda src/applications/slda.cpp $<TARGET_OBJECTS:applications>)
    target_link_libraries(slda ${DOCOPT_LIBRARIES})
    target_link_libraries(slda ldaplusplus)

    set(TARGETS_FOR_INSTALL ${TARGETS_FOR_INSTALL} fslda)
    add_executable(fslda src/applications/fslda.cpp $<TARGET_OBJECTS:applications>)
    target_link_libraries(fslda ${DOCOPT_LIBRARIES})
    target_link_libraries(fslda ldaplusplus)
endif()

# Set the installation rules
install(TARGETS ${TARGETS_FOR_INSTALL}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        PATTERN "test" EXCLUDE)

# Build the tests
# We need to explicitly define a test the same way we explicitly define a
# source file
if (GTEST_FOUND)
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
    target_link_libraries(test_all ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(test_all ldaplusplus)
    # We will also add a custom target check that runs the tests
    # The || : below is so that make won't complain in case of failed tests. We
    # have gtest complaining in color and style no need for another whiner.
    add_custom_target(check ./test_all || : DEPENDS test_all)
else()
    message(WARNING "GTest was not found so tests will not be built")
endif()

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
endforeach(BENCH_FILE)
add_custom_target(bench : DEPENDS ${BENCH_TARGETS})
