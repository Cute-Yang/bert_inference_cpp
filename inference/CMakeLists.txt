set(target_name "bert_inference")
add_library(
    ${target_name} STATIC
    bert_classification.cpp
    bert_classification.h
)

# set the cuda home
set(CUDA_TOOLKIT_ROOT_DIR "/dockerdata/sunyyao/lazydog_build/cuda-11.0")

# set the tensorrt dir
set(tensorrt_home "/dockerdata/sunyyao/SDK/TensorRT-8.2.1.8")
message(STATUS "tensorrt -> ${tensorrt_home}")

find_library(
    nvinfer_lib
    nvinfer
    ${tensorrt_home}/lib
)

message(STATUS "nvinfer -> ${nvinfer_lib}")


find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Find CUDA,the include dir is ${CUDA_INCLUDE_DIRS},cudart -> ${CUDA_CUDART_LIBRARY}")
else(CUDA_FOUND)
    message(FATAL_ERROR "not found cuda...")
endif(CUDA_FOUND)


target_include_directories(
    ${target_name}
    PUBLIC ${tensorrt_home}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CUDA_INCLUDE_DIRS}
)

target_link_directories(
    ${target_name}
    PRIVATE ${tensorrt_home}/lib
)

target_link_libraries(
    ${target_name}
    PRIVATE ${CUDA_CUDART_LIBRARY}
    PRIVATE ${nvinfer_lib}
    PUBLIC bert_tokenizer
)

