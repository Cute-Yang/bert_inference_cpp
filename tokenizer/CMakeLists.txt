set(target_name "bert_tokenizer")

add_library(
    ${target_name} STATIC
    bert_tokenizer.h
    data_value.h
    bert_tokenizer.cpp
)

target_include_directories(
    ${target_name}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
