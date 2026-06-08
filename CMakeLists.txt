cmake_minimum_required(VERSION 3.16)

project(gas_library LANGUAGES CXX)

find_package(OpenMP REQUIRED)

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
option(DEFINE_RUN_SCRIPT_BUILD "Define macro RUN_SCRIPT_BUILD" OFF)

# ---- Targets ----
add_executable(bench      test/benchmark.cpp)
add_executable(bench_wide test/benchmark_widesheet.cpp)
add_executable(gengt      test/groundtruth.cpp)
add_executable(gengt_wide test/groundtruth_wide.cpp)

set(PROJECT_TARGETS bench bench_wide gengt gengt_wide)

if(DEFINE_RUN_SCRIPT_BUILD)
  target_compile_definitions(bench PRIVATE RUN_SCRIPT_BUILD)
  target_compile_definitions(bench_wide PRIVATE RUN_SCRIPT_BUILD)
endif()

foreach(t ${PROJECT_TARGETS})
  target_include_directories(${t} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party)
  target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
endforeach()

# ---- Compiler flags ----
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  foreach(t ${PROJECT_TARGETS})
    target_compile_options(${t} PRIVATE -Wall -Wno-terminate)
    if(NOT CMAKE_CROSSCOMPILING AND ENABLE_NATIVE_OPTIMIZATIONS)
      target_compile_options(${t} PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
    target_compile_options(${t} PRIVATE
      $<$<CONFIG:Release>:-ffast-math -funroll-loops -finline-functions -ftree-vectorize>)
    if(DISABLE_RTTI)
      target_compile_options(${t} PRIVATE -fno-rtti)
    endif()
    target_compile_options(${t} PRIVATE
      $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
      $<$<CONFIG:Release>:-Ofast>
      $<$<CONFIG:RelWithDebInfo>:-O3 -g -fno-omit-frame-pointer>)
  endforeach()
  if(ENABLE_IPO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
  endif()
  if(ENABLE_SANITIZERS)
    foreach(t ${PROJECT_TARGETS})
      target_compile_options(${t} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
      target_link_options(${t}    PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    endforeach()
  endif()
elseif(MSVC)
  foreach(t ${PROJECT_TARGETS})
    target_compile_options(${t} PRIVATE /W4 /permissive-)
    if(DISABLE_RTTI)
      target_compile_options(${t} PRIVATE /GR-)
    endif()
    target_compile_options(${t} PRIVATE
      $<$<CONFIG:Debug>:/Od /Z7 /Ob0>
      $<$<CONFIG:Release>:/O2 /Oi /Ot /GL>
      $<$<CONFIG:RelWithDebInfo>:/O2 /Z7>)
    if(ENABLE_IPO)
      target_link_options(${t} PRIVATE $<$<CONFIG:Release>:/LTCG>)
    endif()
  endforeach()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
