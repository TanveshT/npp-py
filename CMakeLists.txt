cmake_minimum_required(VERSION 3.22)
project(npp_wrapper LANGUAGES CXX CUDA)

# Python + pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# CUDA / NPP
find_package(CUDAToolkit REQUIRED COMPONENTS cudart nppig nppisu nppc)

# Set your GPU archs (edit as needed)
# 86 = Ampere desktop, 87 = Jetson Orin, 89 = Ada
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86;87;89)
endif()

# Build the Python extension in one go
pybind11_add_module(npp_wrapper MODULE src/npp_wrapper.cpp)
target_compile_features(npp_wrapper PRIVATE cxx_std_17)
set_target_properties(npp_wrapper PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(npp_wrapper PROPERTIES
  BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>;$<TARGET_FILE_DIR:CUDA::nppig>;$<TARGET_FILE_DIR:CUDA::nppisu>;$<TARGET_FILE_DIR:CUDA::nppc>"
  INSTALL_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>;$<TARGET_FILE_DIR:CUDA::nppig>;$<TARGET_FILE_DIR:CUDA::nppisu>;$<TARGET_FILE_DIR:CUDA::nppc>"
)

target_link_libraries(npp_wrapper
  PRIVATE
    CUDA::cudart
    CUDA::nppig   # nppi geometry (resize)
    CUDA::nppisu  # nppi support
    CUDA::nppc    # npp core
)

install(TARGETS npp_wrapper DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/npp_wrapper.pyi DESTINATION .)
