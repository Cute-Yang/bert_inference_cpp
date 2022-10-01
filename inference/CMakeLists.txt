add_library(
    bert_classification STATIC
    bert_classification.h
    bert_classification.cpp
)
# here to set the prefix path of tensorrt!
set(trt_dir /home/layzdog/Seattle/tensorrt)

if(${CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH} ${trt_dir})
else()
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${trt_dir})
endif()
find_library(
    nvinfer_lib
    name nvinfer
)

if(${nvinfer_lib} STREQUAL "nvinfer_lib-NOTFOUND")
    message(FATAL_ERROR "can not find nvinfer lib!")
else()
    message(STATUS "find nvinfer at -> ${nvinfer_lib}")
endif()
    

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    target_include_directories(
        bert_classification
        PRIVATE ${CUDA_INCLUDE_DIRS}
    )
    message(STATUS "cuda_include_dir -> ${CUDA_INCLUDE_DIRS},cudart -> ${CUDA_CUDART_LIBRARY}!")
else()
    message(FATAL_ERROR "can not find cuda...")
endif()

target_include_directories(
    bert_classificaiton
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    bert_classification
    PRIVATE ${CUDA_CUDART_LIBRARY}
)

target_link_libraries(
    bert_classification
    PUBLIC tokenizer
)