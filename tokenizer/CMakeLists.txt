cmake_minimum_required(VERSION 3.15)
project(
    tokenizer_test
    VERSION 1.0.0
    LANGUAGES CXX
)

# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} source_files)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(
    bert_tokenizer SHARED
    bert_tokenizer.cpp
    bert_tokenizer.h
)

add_executable(
    tokenizer_test
    bert_tokenizer_test.cpp
)

target_include_directories(
    bert_tokenizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    tokenizer_test
    bert_tokenizer
)