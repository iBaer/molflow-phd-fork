cmake_minimum_required(VERSION 3.12.2 FATAL_ERROR)

################### Variables. ####################
# Change if you want modify path or other values. #
###################################################


set(PROJECT_NAME flowGPU)
message(${PROJECT_NAME} " generating in " ${CMAKE_CURRENT_SOURCE_DIR})

include(nvcuda_compile_ptx.cmake)

# Folders files
set(HEADER_DIR_1 ${PROJECT_SOURCE_DIR}/src)
set(HEADER_DIR_2 ${PROJECT_SOURCE_DIR}/src_shared)
set(HEADER_DIR_3 ${PROJECT_SOURCE_DIR}/include)
set(HEADER_DIR_4 ${PROJECT_SOURCE_DIR}/include/windows_only)
set(OPTIX_DIR ${PROJECT_SOURCE_DIR}/include/optix)

set(GPU_DIR_1 ${PROJECT_SOURCE_DIR}/src/GPUSim)
set(CUDA_DIR_1 ${PROJECT_SOURCE_DIR}/src/GPUSim/CUDA)

IF (WIN32)
    # set stuff for windows
    set(LINK_DIR_1 ../../lib/win/${MY_BUILD_TYPE})
    #set(LINK_DIR_2 ../../lib_external/win/${MY_BUILD_TYPE})
ELSE()
    # set stuff for other systems
ENDIF()



############## CMake Project ################
#        The main options of project        #
#############################################

project(${PROJECT_NAME} CUDA CXX)

# Find at least a 5.0 version of CUDA.
#set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
#set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")

#find_package(CUDA 10.2 REQUIRED)
message("CUDA found as " ${CUDA_TOOLKIT_ROOT_DIR})

# Definition of Macros
# ... via MolflowGPU.cmake

################# Flags ################
# Defines Flags for Windows and Linux. #
########################################

if(MSVC)

endif(MSVC)
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
endif(NOT MSVC)

################ Files ################
#   --   Add files to project.   --   #
#######################################

#find_library(PROTOBUF_LIBRARY bzip2 HINTS ${LINK_DIR_2})

get_directory_property( DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )

foreach( d ${DirDefs} )
message( STATUS "Found Define: " ${d} )
endforeach()

#message(${EXECUTABLE_OUTPUT_PATH})

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
#[[set(CMAKE_MODULE_PATH
        "${CMAKE_SOURCE_DIR}"
        ${CMAKE_MODULE_PATH}
        )
message("Module Path is " ${CMAKE_MODULE_PATH})]]

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

set(SRC_FILES ${SRC_FILES}
        ${GPU_DIR_1}/Poly2TriConverter.cpp
        ${GPU_DIR_1}/ModelReader.cpp
        ${GPU_DIR_1}/SimulationControllerGPU.cpp
        ${GPU_DIR_1}/SimulationGPU.cpp

        ${GPU_DIR_1}/SimulationOptiX.cpp
        ${GPU_DIR_1}/Model.cpp
        #${GPU_DIR_1}/RNGKernel.cu
        #${GPU_DIR_1}/RandomWrapper.cpp
        ${HEADER_DIR_2}/NeighborScan.cpp

        )


include_directories(${GPU_DIR_1})
include_directories(${CUDA_DIR_1})

set(common_cuda_dir ${PROJECT_SOURCE_DIR}/../../include/common_cuda/)
include_directories(SYSTEM ${common_cuda_dir})
include_directories(SYSTEM ${HEADER_DIR_3})
if(MSVC)
    include_directories(SYSTEM ${HEADER_DIR_4})
endif()

include(${PROJECT_SOURCE_DIR}/configure_optix.cmake)

set(optix_LIBRARY "")
message("Optix include " ${OptiX_INCLUDE})
include_directories(SYSTEM ${OptiX_INCLUDE})
message("Optix local include " ${OPTIX_DIR})
include_directories(SYSTEM ${OPTIX_DIR})

# Prefix the shaders with the full path name to allow stepping through errors with F8.
set( SHADERS
        # Core shaders.
        ${CUDA_DIR_1}/Exception.cu
        ${CUDA_DIR_1}/Geometry.cu
        ${CUDA_DIR_1}/TraceProcessing.cu
        ${CUDA_DIR_1}/TraceProcessing_polygon.cu
        ${CUDA_DIR_1}/RayGeneration.cu
        )

set( SHADERS_HEADERS
        ${GPU_DIR_1}/GPUDefines.h
        ${GPU_DIR_1}/LaunchParams.h
        ${GPU_DIR_1}/PerRayData.h
        ${CUDA_DIR_1}/CommonFunctions.cuh
        )

#cuda_compile_and_embed(geometry_ptx_code ${CUDA_DIR_1}/Geometry.cu)
#cuda_compile_and_embed(trace_ptx_code ${CUDA_DIR_1}/TraceProcessing.cu)
#cuda_compile_and_embed(ray_ptx_code ${CUDA_DIR_1}/RayGeneration.cu)

if (WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
    # Set the base folder where the per-project ptx folders get created.
    set (PTX_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)")
else()
    # DAR This should be independent of ${CMAKE_BUILD_TYPE} because that single-configuration generator will not create subfolders, will it?
    # Otherwise add something with if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set (PTX_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

message("GPU include " ${GPU_DIR_1})
message("Compiling PTX to: ${PTX_TARGET_DIR}")

NVCUDA_COMPILE_PTX( SOURCES ${SHADERS}
        DEPENDENCIES ${SHADERS_HEADERS}
        TARGET_PATH "${PTX_TARGET_DIR}/flowgpu_ptx"
        GENERATED_FILES PTX_SOURCES
        NVCC_OPTIONS "${cuda_flags}" "-I${OptiX_INCLUDE}" "-I${common_cuda_dir}" "-I${HEADER_DIR_2}" "-I${GPU_DIR_1}" "-I${GPU_DIR_1}/CUDA" "${nvcc_flags}"
        )
#"--gpu-architecture=compute_75" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info"
add_executable(dummy_ptx ${SHADERS} ${SHADERS_HEADERS}) #for indexing to work properly
message("Compiled PTX: ${PTX_SOURCES}")


source_group("shaders" FILES ${SHADERS})
source_group("shader_headers" FILES ${SHADERS_HEADERS})

#cuda_compile_and_embed(rng2_ptx_code ${GPU_DIR_1}/cudaRandom.cu)

#set(SRC_FILES ${geometry_ptx_code} ${SRC_FILES})
#set(SRC_FILES ${trace_ptx_code} ${SRC_FILES})
#set(SRC_FILES ${ray_ptx_code} ${SRC_FILES})
if(USE_CGAL)
    find_package(CGAL REQUIRED)
    message("CGAL Dir: ${CGAL_INCLUDE_DIRS}")
    ADD_DEFINITIONS(-DUSE_CGAL)
endif()


add_library(${PROJECT_NAME} ${SRC_FILES} ${PTX_SOURCES})

#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${GPU_DIR_1}
        ${CUDA_DIR_1}
        ${common_cuda_dir}
        ${HEADER_DIR_2} # shared buffer for model
        ${HEADER_DIR_3}
        ${OPTIX_DIR}
        )

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

#set(CMAKE_CUDA_FLAGS "-arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 --default-stream per-thread")
#target_link_options(${PROJECT_NAME} PUBLIC -arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)

# https://docs.nvidia.com/cuda/pascal-compatibility-guide/index.html
target_compile_options(${PROJECT_NAME} PUBLIC
        #$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_50>
        # Maxwell Gen1
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_50,code=sm_50>
        # Maxwell Gen2
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52>
        # Pascal P100
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>
        # Pascal
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>
        # Volta
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=sm_70>
        # Turing
        #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=compute_75>
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
        )


target_link_libraries(${PROJECT_NAME} PUBLIC
        cudaKernel
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        cereal
        fmt
        )

if(USE_CGAL)
    target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL)
endif()


if(MSVC)
    set(LINK_DIR_1 "${CMAKE_SOURCE_DIR}/lib_external/win/${MY_BUILD_TYPE}")
    message(${LINK_DIR_1})

    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
            #"../../include/windows_only"
            "../../../include/windows_only"
            )
    target_link_libraries(${PROJECT_NAME} PUBLIC
            ${LINK_DIR_1}/libgsl.lib
            ${LINK_DIR_1}/libgslcblas.lib
            )
endif(MSVC)