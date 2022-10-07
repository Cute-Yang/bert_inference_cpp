cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_COMPILER "/dockerdata/sunyyao/bin/g++")
set(CMAKE_C_COMPILER "/dockerdata/sunyyao/bin/gcc")
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_BUILD_TYPE Release)

project(
    lazydog_text_inference
    VERSION 1.0.0
    LANGUAGES C CXX
)

add_compile_options("-march=native")
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
