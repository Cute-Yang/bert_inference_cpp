add_library(
    bert_service STATIC
    bert_service.h
    bert_service.cpp
)

set(model_dir "../inference")

target_include_directories(
    bert_service 
    PRIVATE ${model_dir}
)

find_package(RapidJSON)
if(RapidJSON_FOUND)
    target_include_directories(
        bert_service
        PRIVATE ${RapidJSON_INCLUDE_DIR}
    )
else()
    message(FATAL_ERROR "can not found rapidJson!")
endif()

find_package(workflow)
if(WORKFLOW_FOUND)
    message(STATUS "found workflow at -> ${WORKFLOW_INCLUDE_DIR},with version -> ${WORKFLOW_VERSION")
    target_include_directories(
        bert_service
        PRIVATE ${WORKFLOW_INCLUDE_DIR}
    )

    target_link_directories(
        bert_service
        PRIVATE ${WORKFLOW_LIB_DIR}
    )
else()
    message(FATAL_ERROR "can not find workflow...")
endif()

target_link_libraries(
    bert_service
    PUBLIC workflow
    PUBLIC bert_classifier
)
