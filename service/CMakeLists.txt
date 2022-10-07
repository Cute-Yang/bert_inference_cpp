set(target_name "bert_service")

add_library(
    ${target_name} STATIC
    bert_service.cpp
    bert_service.h
)
set(workflow_home "/dockerdata/sunyyao/lazydog_build/workflow")
set(rapidjson_home "/dockerdata/sunyyao/lazydog_build/rapidjson")
if(CMAKE_PREFIX_PATH STREQUAL "")
    set(CMAKE_PREFIX_PATH ${workflow_home} ${rapidjson_home})
else()
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${workflow_home} ${rapidjson_home})
endif()

message(STATUS "the find prefix is -> ${CMAKE_PREFIX_PATH}")

find_package(WORKFLOW REQUIRED)
if(WORKFLOW_FOUND)
    message(STATUS "Found workflow...")
    message(STATUS "The include_dir for workflow is ${WORKFLOW_INCLUDE_DIR},the lib dir is ${WORKFLOW_LIB_DIR},version -> ${WORKFLOW_VERSION}")
else(WORKFLOW_FOUND)
    message(FATAL_ERROR "workflow not found!")
endif()


find_package(RapidJSON)
if(RapidJSON_FOUND)
    message(STATUS "Found RapidJSON,include dir is ${RapidJSON_INCLUDE_DIRS}")
    target_include_directories(
        ${target_name}
        PUBLIC ${RapidJSON_INCLUDE_DIRS}
    )
else(RapidJSON_FOUND)
    message(FATAL_ERROR "Can not found RapidJSON")
endif()


target_include_directories(
    ${target_name}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${WORKFLOW_INCLUDE_DIR}
)

find_library(
    workflow_lib
    workflow
    ${WORKFLOW_LIB_DIR}
)

target_link_libraries(
    ${target_name}
    PRIVATE ${workflow_lib}
    PUBLIC bert_inference
)
