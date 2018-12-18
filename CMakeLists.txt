# General parameters
cmake_minimum_required (VERSION 2.8.12)
project (qpmad_sqp CXX)

# --------------
include(ExternalProject)
include(CMakeParseArguments)

set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
include(ariles_add_external_git_project)
# --------------


# --------------
set (CMAKE_VERBOSE_MAKEFILE     ON)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
endif (NOT CMAKE_BUILD_TYPE)

set (CMAKE_CXX_FLAGS            "${CMAKE_CXX_FLAGS} -std=c++11")
# --------------


# --------------
set(REPOS_INSTALL_PREFIX   "${PROJECT_BINARY_DIR}/extra/")
set(REPOS_LIBDIR       "${REPOS_INSTALL_PREFIX}/lib")
set(REPOS_INCLUDE_DIR  "${REPOS_INSTALL_PREFIX}/include")

set(REPOS_ARGS
    --no-warn-unused-cli
    -DCMAKE_INSTALL_PREFIX=${REPOS_INSTALL_PREFIX}
    -DCMAKE_INSTALL_LIBDIR=${REPOS_LIBDIR}
    -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}
    -DCMAKE_BUILD_TYPE=Release)

include_directories (BEFORE SYSTEM  "${REPOS_INCLUDE_DIR}")
# --------------


# --------------
set(GOOGLETEST_DIR  "${CMAKE_CURRENT_LIST_DIR}/googletest")

set(GOOGLETEST_ARGS
    ${REPOS_ARGS}
    -DBUILD_GMOCK=OFF
    -DBUILD_GTEST=ON
    -Dgtest_disable_pthreads=ON
    "-DCMAKE_CXX_FLAGS=-fPIC")

ariles_add_external_git_project(TARGET_NAME "TGT_googletest"
                                PROJECT_DIR "${GOOGLETEST_DIR}"
                                CMAKE_ARGS   ${GOOGLETEST_ARGS})

set(GOOGLETEST_LIBS "${REPOS_LIBDIR}/${CMAKE_SHARED_MODULE_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}")
# --------------


# --------------
set(QPMAD_DIR  "${CMAKE_CURRENT_LIST_DIR}/qpmad")

set(QPMAD_ARGS
    ${REPOS_ARGS}
    -DQPMAD_BUILD_TESTS=OFF
    -DQPMAD_ENABLE_TRACING=OFF
    -DQPMAD_USE_HOUSEHOLDER=OFF
    "-DCMAKE_CXX_FLAGS=-fPIC")

ariles_add_external_git_project(TARGET_NAME "TGT_qpmad"
                                PROJECT_DIR "${QPMAD_DIR}"
                                CMAKE_ARGS   ${QPMAD_ARGS})
# --------------


# --------------
set(CPPAD_DIR  "${CMAKE_CURRENT_LIST_DIR}/CppAD")

set(CPPAD_ARGS
    ${REPOS_ARGS}
    "-Dcppad_prefix=${REPOS_INSTALL_PREFIX}"
    "-DCMAKE_CXX_FLAGS=-fPIC")

ariles_add_external_git_project(TARGET_NAME "TGT_cppad"
                                PROJECT_DIR "${CPPAD_DIR}"
                                CMAKE_ARGS   ${CPPAD_ARGS})
# --------------


find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})


include_directories ("${PROJECT_SOURCE_DIR}/include")

enable_testing()
add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
