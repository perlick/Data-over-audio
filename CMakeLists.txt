cmake_minimum_required(VERSION 3.10)
project(DOA)
set(C_STANDARD 11)
set(C_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.c
    src/shared_mem.c
    src/filter.c
)

add_executable(DOA  ${SOURCES})

target_include_directories(DOA
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)
