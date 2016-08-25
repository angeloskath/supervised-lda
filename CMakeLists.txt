cmake_minimum_required(VERSION 2.8.9)
project(LDA++)

# Set the compiler flags
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

# Set other flags and variables
set(BUILD_SHARED_LIBS ON)

# Bring the headers into the project
include_directories(include)

# And add the sources explicitly, see 
# http://stackoverflow.com/questions/1027247/best-way-to-specify-sourcefiles-in-cmake
# for an explanation (yes stackoverflow the very credible software authority)
set(SOURCES
    src/ApproximatedSupervisedEStep.cpp
    src/CorrespondenceSupervisedEStep.cpp
    src/CorrespondenceSupervisedMStep.cpp
    src/Document.cpp
    src/e_step_utils.cpp
    src/Events.cpp
    src/FastSupervisedEStep.cpp
    src/FastUnsupervisedEStep.cpp
    src/initialize.cpp
    src/LDABuilder.cpp
    src/LDA.cpp
    src/MultinomialLogisticRegression.cpp
    src/MultinomialSupervisedEStep.cpp
    src/MultinomialSupervisedMStep.cpp
    src/OnlineSupervisedMStep.cpp
    src/SecondOrderLogisticRegressionApproximation.cpp
    src/SecondOrderSupervisedMStep.cpp
    src/SemiSupervisedEStep.cpp
    src/SemiSupervisedMStep.cpp
    src/SupervisedEStep.cpp
    src/SupervisedMStep.cpp
    src/UnsupervisedEStep.cpp
    src/UnsupervisedMStep.cpp
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
        LIBRARY DESTINATION lib)
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
