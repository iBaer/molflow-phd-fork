cmake_minimum_required(VERSION 3.12.2 FATAL_ERROR)

project(algobench)

link_directories(${LINK_DIR_1} ${LINK_DIR_2})

set(EXTRA_SRC
        ../src/IO/CSVExporter.cpp
        ../src_shared/File.cpp
        ../src_shared/FlowMPI.h #contains templates
        ../src_shared/FlowMPI.cpp
        )

add_executable(algobench benchmark_suite.cpp ${EXTRA_SRC})
target_link_libraries(algobench PRIVATE molflowSim)

target_include_directories(algobench PRIVATE ../include ../src_shared ../src)
target_link_directories(algobench PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

target_compile_features(algobench PRIVATE cxx_std_17)
#add_test(NAME example_test COMMAND testsuite)
#[[
file(COPY ./simulation.cfg
        ./TestCases
        DESTINATION ${CMAKE_EXECUTABLE_OUTPUT_DIRECTORY})]]
