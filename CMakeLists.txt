cmake_minimum_required(VERSION 3.12.2 FATAL_ERROR)

project(molflow_proj)

message("-- CMAKE_SYSTEM_INFO_FILE: ${CMAKE_SYSTEM_INFO_FILE}")
message("-- CMAKE_SYSTEM_NAME:      ${CMAKE_SYSTEM_NAME}")
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message("-- CMAKE_SYSTEM:           ${CMAKE_SYSTEM}")

string (REGEX MATCH "\\.el[1-9]" os_version_suffix ${CMAKE_SYSTEM})
message("-- os_version_suffix:      ${os_version_suffix}")

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#Use new BVH by default
#priority: USE_OLD_BVH > USE_KDTREE
option(USE_OLD_BVH "simulation will use old bvh structure by default" OFF)
option(USE_KDTREE "simulation will use kd structure instead of bvh" OFF)

option(USE_CLANG "build application with clang" OFF) # OFF is the default
option(USE_PROFILING "disable optimisation for profiling the application" OFF) # OFF is the default

option(NO_INTERFACE "only build CLI binary and dependencies" OFF) # OFF is the default
if(NO_INTERFACE)
    message(STATUS "Will not build GUI related libraries and executables.")
endif()

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    MESSAGE("Adding Debug flag...")
    ADD_DEFINITIONS(-DDEBUG)
    ADD_DEFINITIONS(-D_DEBUG)
endif()
message(STATUS "Build type used: " ${CMAKE_BUILD_TYPE})
set(CMAKE_VERBOSE_MAKEFILE OFF)

################# Flags ################
# Defines Flags for Windows and Linux. #
########################################

if(USE_OLD_BVH)
    MESSAGE("Simulation with old BVH...")
    ADD_DEFINITIONS(-DUSE_OLD_BVH)
elseif(USE_KDTREE)
    MESSAGE("Simulation with KDTREE...")
    ADD_DEFINITIONS(-DUSE_KDTREE)
else()
    MESSAGE("Simulation with new BVH...")
    #ADD_DEFINITIONS(-DUSE_KDTREE)
endif()

# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options-listed-alphabetically?view=vs-2019
if(MSVC)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")

    #set(CMAKE_GENERATOR_PLATFORM x64)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /mwindows")

    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mwindows")
    # main vs WinMain
    # https://stackoverflow.com/questions/2752792/whats-the-equivalent-of-gccs-mwindows-option-in-cmake
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /subsystem:windows /ENTRY:mainCRTStartup")
    #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W3 /EHsc")
    #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W3 /EHsc")
    #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W4 /MDd /Od /EHsc")
    #set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /W4 /GL /O2 /Ob2 /EHsc")
    #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W4 /GL /O2 /EHsc")
endif(MSVC)
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
	#Flag to set debug symbols
	#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
	#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
	
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
endif(NOT MSVC)

if(USE_PROFILING)
    MESSAGE("Adding Profiling flag...")
    ADD_DEFINITIONS(-DPROFILING)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif(USE_PROFILING)

set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/")
include(CMake/Molflow.cmake)

add_subdirectory(src_shared EXCLUDE_FROM_ALL)

#IF(NOT WIN32)
add_subdirectory(CMake/molflow_cli)
add_subdirectory(CMake/molflow_sim)
add_subdirectory(algobench)
add_subdirectory(src/TimeDependent)
#ENDIF()

if(NOT NO_INTERFACE)
    add_subdirectory(CMake/molflow_win)
    add_dependencies(molflow ziplib)
    IF (WIN32)
        add_subdirectory(CMake/png_zlib_win_library)
    ENDIF(WIN32)
    add_dependencies(molflow shared_simulator shared_interface compress)
endif()

add_dependencies(molflowSim ziplib)


#IF(NOT WIN32)
add_dependencies(molflowCLI molflowSim)
#ENDIF()

option(USE_TESTS "Build the tests" OFF)
if(USE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
    add_dependencies(testsuite molflowSim)
    add_subdirectory(src/TimeDependent)
endif()

# Allow to install molflowCLI into Unix OS folders (e.g. /usr/local/bin)
# molflow GUI not supported right now, as extra folders are needed

#SET(MOLFLOW_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
SET(MOLFLOW_INSTALL_DIR molflow)

include(GNUInstallDirs)

# if no path is provided, use user home path
# e.g. /home/user/molflow
# cmake --install . --prefix /usr/local/bin/
# will install to /usr/local/bin/molflow
# cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/pascal/molflow_test/ .
#
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}" CACHE PATH "..." FORCE)
endif()

install(CODE "MESSAGE(\"Installing in home directory ${CMAKE_INSTALL_PREFIX}\")")
install(DIRECTORY copy_to_build/images
        copy_to_build/desorption_yields
        copy_to_build/parameter_catalog
        DESTINATION ${MOLFLOW_INSTALL_DIR}
        #USE_SOURCE_PERMISSIONS
        PATTERN "images/*"
        PATTERN "desorption_yields/*"
        PATTERN "parameter_catalog/*"
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ GROUP_WRITE
        )

if(NOT NO_INTERFACE)
    install(TARGETS molflow
            CONFIGURATIONS Release
            RUNTIME
            DESTINATION ${MOLFLOW_INSTALL_DIR}
    )
endif()
install(TARGETS molflowCLI
        CONFIGURATIONS Release
        RUNTIME
        DESTINATION ${MOLFLOW_INSTALL_DIR}
        )
#[[install(DIRECTORY molflowCLI
        CONFIGURATIONS Release
        RUNTIME
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        )]]
