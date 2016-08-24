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
