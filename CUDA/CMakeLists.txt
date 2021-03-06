cmake_minimum_required(VERSION 2.8.11)
project(cdn)

# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX)
#     include(CheckCXXCompilerFlag)
#     check_cxx_compiler_flag(--std=c++11 SUPPORTS_STD_CXX11)
#     if(SUPPORTS_STD_CXX11)
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
#         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --std=c++11")
#     else()
#         message(ERROR "Compiler does not support --std=c++11")
#     endif()
# endif()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET (CMAKE_AR      "/usr/bin/llvm-ar")
    SET (CMAKE_LINKER  "/usr/bin/llvm-ld")
    SET (CMAKE_NM      "/usr/bin/llvm-nm")
    SET (CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
    SET (CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
endif()

find_package(CUDA REQUIRED)

# Add the include directories
INCLUDE_DIRECTORIES(win_support Global InputTests NeuralNetwork MLP CUDAFiles TinyXML ${CUDA_INCLUDE_DIRS})

# Use the compile definitions
add_definitions()

# Add compiler flags for bulding executables
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

FILE(GLOB_RECURSE cdn_SRCS CUDAFiles/*.cpp Global/*.cpp InputTests/*.cpp MLP/*.cpp NeuralNetwork/*.cpp TinyXML/*.cpp */*.cu)
#set(cdn_SRCS Global/CUDA.cpp Global/)
message("sources: ${cdn_SRCS}")

CUDA_ADD_EXECUTABLE(cdn ${cdn_SRCS})
target_link_libraries(cdn ${CUDA_LIBRARIES})

#install(TARGETS cdn RUNTIME DESTINATION bin)
