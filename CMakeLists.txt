cmake_minimum_required(VERSION 3.18)
project(TransformerCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(include)

set(CORE_SOURCES
    src/model.cu
    src/optimizer.cu
    src/dataloader.cpp
    src/kernels/embedding.cu
    src/kernels/layernorm.cu
    src/kernels/matmul.cu
    src/kernels/attention.cu
    src/kernels/ffn.cu
    src/kernels/residual.cu
    src/kernels/loss.cu
)

# Executables
add_executable(train src/main.cu ${CORE_SOURCES})
target_link_libraries(train PRIVATE CUDA::cublas CUDA::cudart)

add_executable(benchmark src/benchmark.cu ${CORE_SOURCES})
target_link_libraries(benchmark PRIVATE CUDA::cublas CUDA::cudart)

add_executable(parity_audit src/parity_audit.cu ${CORE_SOURCES})
target_link_libraries(parity_audit PRIVATE CUDA::cublas CUDA::cudart)

add_executable(test_kernels tests/test_kernels.cu ${CORE_SOURCES})
target_link_libraries(test_kernels PRIVATE CUDA::cublas CUDA::cudart)

set_target_properties(train benchmark parity_audit test_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
