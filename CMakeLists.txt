cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(npc-dialect LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
endif()

#-------------------------------------------------------------------------------
# MSVC defaults
#-------------------------------------------------------------------------------

if(MSVC)
  add_compile_options(
    $<$<CONFIG:>:/MD>
    $<$<CONFIG:Debug>:/MD>
    $<$<CONFIG:Release>:/MD>
  )
endif()

#-------------------------------------------------------------------------------
# Options of Runtime
#-------------------------------------------------------------------------------

#option(NPC_ENABLE_REFJIT "Enables the reference JIT backend." ON)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# To load binding header files at under lib directory.
include_directories(${LLVM_INCLUDE_DIRS}/../../mlir/lib)
message(STATUS "LLVM lib directory path: ${LLVM_INCLUDE_DIRS}/../../mlir/lib")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

set(NPC_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(NPC_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Building numpy-compiler project at ${NPC_MLIR_SOURCE_DIR} (into ${NPC_MLIR_BINARY_DIR})")

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
set(NPC_TABLEGEN_ARGS "")

add_subdirectory(include)
add_subdirectory(lib)
#add_subdirectory(test)
add_subdirectory(tools)
#
#add_custom_target(check-il-all)
#add_dependencies(check-il-all check-il)

#-------------------------------------------------------------------------------
# Optional feature selection
#-------------------------------------------------------------------------------

#if(IL_ENABLE_REFJIT)
#  add_compile_definitions(IL_ENABLE_REFJIT)
#  message(STATUS "Reference JIT backend enabled")
#endif()

#-------------------------------------------------------------------------------
# Python Configuration
#-------------------------------------------------------------------------------

#find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)
#message(STATUS "Found pybind11 include dirs: ${pybind11_INCLUDE_DIR}")
#
#include(MLIRDetectPythonEnv)
#mlir_configure_python_dev_packages()
#
#if(NOT IL_PYTHON_DIR)
#  set(IL_PYTHON__DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
#endif()
#
#if(NOT IL_PYTHON_PACKAGES_DIR)
#  set(IL_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
#endif()
#add_dependencies(check-il-all check-il-python)

#add_subdirectory(python)
