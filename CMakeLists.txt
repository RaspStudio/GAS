cmake_minimum_required(VERSION 3.16)

project(gaslib LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(Boost QUIET)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_IPO "Enable interprocedural optimization (LTO) in optimized builds" ON)
option(DISABLE_RTTI "Build without RTTI" ON)
option(ENABLE_SANITIZERS "Enable ASan/UBSan in Debug" OFF)
option(ENABLE_NATIVE_OPTIMIZATIONS "Enable native architecture optimizations" ON)

add_executable(bench      test/benchmark.cpp)
add_executable(gengt      test/groundtruth.cpp)

set(PROJECT_TARGETS bench gengt)

message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")

option(DEFINE_RUN_SCRIPT_BUILD "Define macro RUN_SCRIPT_BUILD" OFF)
if(DEFINE_RUN_SCRIPT_BUILD)
  target_compile_definitions(bench PRIVATE RUN_SCRIPT_BUILD)
endif()

foreach(t ${PROJECT_TARGETS})
  target_include_directories(${t}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/third-party
  )
  target_compile_features(${t} PRIVATE cxx_std_20)
endforeach()

target_link_libraries(gengt PRIVATE OpenMP::OpenMP_CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  foreach(t ${PROJECT_TARGETS})
    target_compile_options(${t} PRIVATE -Wall)

    if (NOT CMAKE_CROSSCOMPILING AND ENABLE_NATIVE_OPTIMIZATIONS)
      target_compile_options(${t} PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
    target_compile_options(${t} PRIVATE
      $<$<CONFIG:Release>:-ffast-math -funroll-loops -finline-functions -ftree-vectorize>
    )

    if (DISABLE_RTTI)
      target_compile_options(${t} PRIVATE -fno-rtti)
    endif()

    target_compile_options(${t} PRIVATE
      $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
      $<$<CONFIG:Release>:-Ofast>
      $<$<CONFIG:RelWithDebInfo>:-O3 -g -fno-omit-frame-pointer>
    )
  endforeach()

  if (ENABLE_IPO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
  endif()

  if (ENABLE_SANITIZERS)
    foreach(t ${PROJECT_TARGETS})
      target_compile_options(${t} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
      target_link_options(${t}    PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    endforeach()
  endif()

else()
  message(STATUS "Only Supported on Clang and GCC compilers")
endif()

message(STATUS "Generator: ${CMAKE_GENERATOR}")
message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
