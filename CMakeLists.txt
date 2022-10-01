cmake_minimum_required(VERSION 3.15)

project(
    lazydog_text_inference
    VERSION 1.0.0
    LANGUAGES CXX
)

add_subdirectory("tokenizer")
add_subdirectory("inference")
add_subdirectory("service")

add_executable(
    main 
    main.cpp
)

target_link_libraries(
    main
    bert_service
)
