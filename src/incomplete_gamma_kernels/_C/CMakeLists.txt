cmake_minimum_required(VERSION 3.27)

project(igk LANGUAGES CXX CUDA)


if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
    message(FATAL_ERROR "CUDA 12.4+ required but only ${CMAKE_CUDA_COMPILER_VERSION} detected.")
endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND HOST_DEVICE_FLAGS "-Wall")
    list(APPEND HOST_DEVICE_FLAGS "-Wextra")
    list(APPEND HOST_DEVICE_FLAGS "-Wshadow")
    list(APPEND HOST_DEVICE_FLAGS "-Wsign-compare")
    list(APPEND HOST_DEVICE_FLAGS "-Wconversion")
    list(APPEND HOST_DEVICE_FLAGS "-Wfloat-equal")
    list(APPEND HOST_DEVICE_FLAGS "-Wundef")
    list(APPEND HOST_DEVICE_FLAGS "-Wdouble-promotion")
endif()


include(FetchContent)

if(NOT TARGET glm::glm)
    FetchContent_Declare(
        glm
        PREFIX glm
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
        URL_HASH SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
        DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/external/glm"
        SYSTEM
    )

    FetchContent_MakeAvailable(glm)
endif()


find_package(CUDAToolkit REQUIRED)


find_package(charonload)

if(charonload_FOUND)
    charonload_add_torch_library(igk STATIC)

    target_sources(igk PRIVATE src/sparse/linalg/cg_cuda.cpp
                               src/sparse/linalg/cg.cpp
                               src/sparse/gram_cuda.cu
                               src/sparse/gram.cpp
                               src/density_weights_cuda.cu
                               src/density_weights.cpp
                               src/lop_cuda.cu
                               src/lop.cpp
                               src/mean_shift_cuda.cu
                               src/mean_shift.cpp
                               src/regularity_cuda.cu
                               src/regularity.cpp)

    target_include_directories(igk PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_compile_definitions(igk PRIVATE "__CUDA_NO_HALF_OPERATORS__")
    target_compile_features(igk PUBLIC cxx_std_17)
    target_compile_options(igk PRIVATE ${HOST_DEVICE_FLAGS})
    target_link_libraries(igk PUBLIC glm::glm PRIVATE CUDA::cublas CUDA::cusparse)


    charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)

    target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/bindings.cpp)

    target_compile_options(${TORCH_EXTENSION_NAME} PRIVATE ${HOST_DEVICE_FLAGS})
    target_link_libraries(${TORCH_EXTENSION_NAME} PRIVATE igk)
endif()
