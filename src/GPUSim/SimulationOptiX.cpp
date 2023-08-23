// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,  t      //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <curand_kernel.h>
#include <algorithm>
#include <iostream>
#include "helper_output.h"
#include "SimulationOptiX.h"
#include "CUDA/cudaRandom.cuh"
#include "GPUDefines.h"
#include "LaunchParams.h"

extern void initializeRand(unsigned int kernelSize, curandState_t *states, float *randomNumbers);

extern void generateRand(unsigned int kernelSize, curandState_t *states, float *randomNumbers);

extern void destroyRand(curandState_t *states, float *randomNumbers);

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <fstream>
#include <ctime>

// Helper functions
inline void ProcessSleep(const unsigned int milliseconds) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    Sleep(milliseconds);
#else
    struct timespec tv{};
    tv.tv_sec = milliseconds / 1000;
    tv.tv_nsec = (milliseconds % 1000) * 1000000;
    nanosleep(&tv, nullptr);
#endif
}

/*! \namespace flowgpu - Molflow GPU code */
namespace flowgpu {

    std::string readPTX(std::string const &filename) {
        std::ifstream inputPtx(filename);

        if (!inputPtx) {
            std::cerr << "ERROR: readPTX() Failed to open file " << filename << '\n';
            return std::string();
        }

        std::stringstream ptx;

        ptx << inputPtx.rdbuf();

        if (inputPtx.fail()) {
            std::cerr << "ERROR: readPTX() Failed to read file " << filename << '\n';
            return std::string();
        }

        return ptx.str();
    }

    //extern "C" char Geometry_ptx[];
    //extern "C" char trace_ptx_code[];
    //extern "C" char ray_ptx_code[];

    /*! SBT record for a raygen program *//*
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        //void *data;
        PolygonRayGenData data_poly;
    };

    *//*! SBT record for a miss program *//*
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        //void *data;
        //void *data_poly;
        PolygonMeshSBTData data_poly;
    };

    *//*! SBT record for a hitgroup program *//*
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        PolygonMeshSBTData data_poly;
    };

    *//*! SBT record for a raygen program *//*
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecordTri
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        //void *data;
        TriangleRayGenData data_tri;
    };

    *//*! SBT record for a miss program *//*
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecordTri
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        //void *data;
        //void *data_poly;
        TriangleMeshSBTData data_tri;
    };

    *//*! SBT record for a hitgroup program *//*
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecordTri
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        TriangleMeshSBTData data_tri;
    };*/

    /*! SBT record template, use void* and init with nullptr for dummy value */
    template<typename T>
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) Record {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT)

        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    typedef Record<PolygonRayGenData> RaygenRecord;          /*! SBT record for a raygen program */
    typedef Record<PolygonMeshSBTData> MissRecord;           /*! SBT record for a miss program */
    typedef Record<PolygonMeshSBTData> HitgroupRecord;       /*! SBT record for a hitgroup program */
#if defined(DEBUG)
typedef Record<int> ExceptionRecord;                     /*! SBT record for a hitgroup program */
#endif
typedef Record<TriangleRayGenData> RaygenRecordTri;
    typedef Record<TriangleMeshSBTData> MissRecordTri;
    typedef Record<TriangleMeshSBTData> HitgroupRecordTri;

    static void
    polygon_bound(std::vector<uint32_t> polyIndicies, uint32_t indexOffset, std::vector<float3> polyVertices,
                  uint32_t nbVert, float *result) {

        float3 m_max{static_cast<float>(-1e100), static_cast<float>(-1e100), static_cast<float>(-1e100)};
        float3 m_min{static_cast<float>(1e100), static_cast<float>(1e100), static_cast<float>(1e100)};

        for (uint32_t ind = indexOffset; ind < indexOffset + nbVert; ind++) {
            auto polyIndex = polyIndicies[ind];
            const auto &vert = polyVertices[polyIndex];
            //auto newVert = vert / dot(vert,vert); // scale back
            m_min.x = std::min(vert.x, m_min.x);
            m_min.y = std::min(vert.y, m_min.y);
            m_min.z = std::min(vert.z, m_min.z);
            m_max.x = std::max(vert.x, m_max.x);
            m_max.y = std::max(vert.y, m_max.y);
            m_max.z = std::max(vert.z, m_max.z);
        }

        OptixAabb *aabb = reinterpret_cast<OptixAabb *>(result);

        *aabb = {
                m_min.x, m_min.y, m_min.z,
                m_max.x, m_max.y, m_max.z
        };
    }

    SimulationOptiX::~SimulationOptiX() {
        cleanup();
    }

    /*! constructor - performs all setup, including initializing
      optix, creates module, pipeline, programs, SBT, etc. */
    SimulationOptiX::SimulationOptiX(const std::shared_ptr<Model> &model,
                                     const std::shared_ptr<flowgpu::MolflowGPUSettings> &settings)
            : model(model), settings(settings) {
        uint2 launchSize = make_uint2(settings->kernelDimensions[0], settings->kernelDimensions[1]);
        Log::console_msg(4, "#flowgpu: initializing launch parameters ...\n");
        try{
        initLaunchParams(launchSize);

        Log::console_msg(4, "#flowgpu: initializing optix ...\n");
        initOptix();

        Log::console_msg(4, "#flowgpu: creating optix context ...\n");
        createContext();

        Log::console_msg(4, "#flowgpu: setting up module ...\n");
        createModule();

        std::vector<OptixProgramGroup> programGroups;
        Log::console_msg(4, "#flowgpu: creating raygen programs ...\n");
        createRaygenPrograms(programGroups);
        Log::console_msg(4, "#flowgpu: creating miss programs ...\n");
        createMissPrograms(programGroups);
        Log::console_msg(4, "#flowgpu: creating hitgroup programs ...\n");
        createHitgroupPrograms(programGroups);
#if defined(DEBUG)
        Log::console_msg(4, "#flowgpu: creating exception programs ...\n");
        createExceptionPrograms(programGroups);
#endif
        Log::console_msg(4, "#flowgpu: building acceleration structure ...\n");
#ifdef WITHTRIANGLES
        state.launchParams.traversable = state.asHandle = buildAccelTriangle();
#else
        state.launchParams.traversable = state.asHandle = buildAccelPolygon();
#endif
        Log::console_msg(4, "#flowgpu: setting up optix pipeline ...\n");
        createPipeline(programGroups);

        Log::console_msg(4, "#flowgpu: building SBT ...\n");
#ifdef WITHTRIANGLES
        buildSBTTriangle();
#else
        buildSBTPolygon();
#endif
    }
    catch (std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        ProcessSleep(10000);
    }
        Log::console_msg(4, "#flowgpu: context, module, pipeline, etc, all set up ...\n");

        Log::console_msg(4, MF_TERMINAL_GREEN);
        Log::console_msg(4, "#flowgpu: Optix 7 Sample fully set up\n");
        Log::console_msg(4, MF_TERMINAL_DEFAULT);
    }

    OptixTraversableHandle SimulationOptiX::buildAccelPolygon() {
        PING;
        PRINT(model->poly_meshes.size());

        // All buffers that should be uploaded to device memory
        poly_memory.aabbBuffer.resize(model->poly_meshes.size());
        poly_memory.vertexBuffer.resize(model->poly_meshes.size());
        poly_memory.vertex2Buffer.resize(model->poly_meshes.size());
        poly_memory.vertex2x64Buffer.resize(model->poly_meshes.size());
        poly_memory.indexBuffer.resize(model->poly_meshes.size());
        poly_memory.polyBuffer.resize(model->poly_meshes.size());
        poly_memory.facprobBuffer.resize(model->poly_meshes.size());
        poly_memory.cdfBuffer.resize(model->poly_meshes.size());

        OptixTraversableHandle asHandle{0};

        // ==================================================================
        // triangle inputs
        // ==================================================================

        //std::vector<OptixBuildInput> triangleInput(model->poly_meshes.size());

        std::vector<OptixBuildInput> polygonInput(model->poly_meshes.size());
        std::vector<CUdeviceptr> d_aabb(model->poly_meshes.size());
        /*std::vector<CUdeviceptr> d_vertices(model->poly_meshes.size());
        std::vector<CUdeviceptr> d_vertices2(model->poly_meshes.size());
        std::vector<CUdeviceptr> d_indices(model->poly_meshes.size());
        std::vector<CUdeviceptr> d_polygons(model->poly_meshes.size());*/

        for (int meshID = 0; meshID < model->poly_meshes.size(); meshID++) {

            // upload the model to the device: the builder
            PolygonMesh &mesh = *model->poly_meshes[meshID];

            std::vector<OptixAabb> aabb(mesh.poly.size());
            int bbCount = 0;
            for (flowgpu::Polygon &poly : mesh.poly) {
                polygon_bound(mesh.indices, poly.indexOffset, mesh.vertices3d, poly.nbVertices,
                              reinterpret_cast<float *>(&aabb[bbCount]));
                //Log::console_msg(4, bbCount<<"# poly box: " << "("<<aabb[bbCount].minX <<","<<aabb[bbCount].minY <<","<<aabb[bbCount].minZ <<")-("<<aabb[bbCount].maxX <<","<<aabb[bbCount].maxY <<","<<aabb[bbCount].maxZ <<")"<<std::endl;
                //Log::console_msg(4, bbCount<<"# poly uv: " << "("<<poly.U <<","<<poly.V <<")"<<std::endl;

                bbCount++;
            }

            poly_memory.aabbBuffer[meshID].alloc_and_upload(aabb);

            polygonInput[meshID] = {};
            polygonInput[meshID].type
                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            uint32_t aabb_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE
                                            | OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
            // create local variables, because we need a *pointer* to the
            // device pointers
            d_aabb[meshID] = poly_memory.aabbBuffer[meshID].d_pointer();

            // in this example we have one SBT entry, and no per-primitive
            // materials:
            polygonInput[meshID].customPrimitiveArray.aabbBuffers = &(d_aabb[meshID]);
            polygonInput[meshID].customPrimitiveArray.flags = aabb_input_flags;
            polygonInput[meshID].customPrimitiveArray.numSbtRecords = 1;
            polygonInput[meshID].customPrimitiveArray.numPrimitives = mesh.poly.size();
            polygonInput[meshID].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
            polygonInput[meshID].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
            polygonInput[meshID].customPrimitiveArray.primitiveIndexOffset = 0;
        }
        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
                                  | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                                  | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
                            (state.context,
                             &accelOptions,
                             polygonInput.data(),
                             (int) model->poly_meshes.size(),  // num_build_inputs
                             &blasBufferSizes
                            ));

        // ==================================================================
        // prepare compaction
        // ==================================================================

        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.d_pointer();

        // ==================================================================
        // execute build (main stage)
        // ==================================================================

        CUDABuffer tempBuffer;
        tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(state.context,
                                    nullptr,
                                    &accelOptions,
                                    polygonInput.data(),
                                    (int) model->poly_meshes.size(),
                                    tempBuffer.d_pointer(),
                                    tempBuffer.sizeInBytes,

                                    outputBuffer.d_pointer(),
                                    outputBuffer.sizeInBytes,

                                    &asHandle,

                                    &emitDesc, 1
        ));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // perform compaction
        // ==================================================================
        uint64_t compactedSize;
        compactedSizeBuffer.download(&compactedSize, 1);

        state.asBuffer.alloc(compactedSize);
        OPTIX_CHECK(optixAccelCompact(state.context,
                                      nullptr,
                                      asHandle,
                                      state.asBuffer.d_pointer(),
                                      state.asBuffer.sizeInBytes,
                                      &asHandle));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // aaaaaand .... clean up
        // ==================================================================
        outputBuffer.free(); // << the UNcompacted, temporary output buffer
        tempBuffer.free();
        compactedSizeBuffer.free();

        return asHandle;
    }

    OptixTraversableHandle SimulationOptiX::buildAccelTriangle() {
        PING;
        PRINT(model->triangle_meshes.size());

        // All buffers that should be uploaded to device memory
        //memory.aabbBuffer.resize(model->triangle_meshes.size());
        tri_memory.vertexBuffer.resize(model->triangle_meshes.size());
        tri_memory.texcoordBuffer.resize(model->triangle_meshes.size());

        //memory.vertex2Buffer.resize(model->triangle_meshes.size());
        tri_memory.indexBuffer.resize(model->triangle_meshes.size());
        tri_memory.sbtIndexBuffer.resize(model->triangle_meshes.size());
        tri_memory.polyBuffer.resize(model->triangle_meshes.size());
        tri_memory.facprobBuffer.resize(model->triangle_meshes.size());

        OptixTraversableHandle asHandle{0};

        // ==================================================================
        // triangle inputs
        // ==================================================================
        std::vector<OptixBuildInput> triangleInput(model->triangle_meshes.size());
        std::vector<CUdeviceptr> d_vertices(model->triangle_meshes.size());
        std::vector<CUdeviceptr> d_indices(model->triangle_meshes.size());
        std::vector<CUdeviceptr> d_sbt_indices(model->triangle_meshes.size());

        //std::vector<uint32_t> triangleInputFlags(model->triangle_meshes.size());

        //std::vector<OptixBuildInput> polygonInput(model->triangle_meshes.size());
        //std::vector<CUdeviceptr> d_aabb(model->triangle_meshes.size());
        /*std::vector<CUdeviceptr> d_vertices(model->triangle_meshes.size());
        std::vector<CUdeviceptr> d_vertices2(model->triangle_meshes.size());
        std::vector<CUdeviceptr> d_indices(model->triangle_meshes.size());
        std::vector<CUdeviceptr> d_polygons(model->triangle_meshes.size());*/

        for (int meshID = 0; meshID < model->triangle_meshes.size(); meshID++) {

            // upload the model to the device: the builder
            TriangleMesh &mesh = *model->triangle_meshes[meshID];

            /*for(int3& vert : mesh.indices){

                Log::console_msg(4, vert << std::endl;
                //Log::console_msg(4, bbCount<<"# poly uv: " << "("<<poly.U <<","<<poly.V <<")"<<std::endl;
            }
            //exit(0);*/

            tri_memory.vertexBuffer[meshID].alloc_and_upload(mesh.vertices3d);
            tri_memory.indexBuffer[meshID].alloc_and_upload(mesh.indices);
            tri_memory.sbtIndexBuffer[meshID].alloc_and_upload(mesh.sbtIndices);

            triangleInput[meshID] = {};
            triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // create local variables, because we need a *pointer* to the
            // device pointers
            d_vertices[meshID] = tri_memory.vertexBuffer[meshID].d_pointer();
            d_indices[meshID] = tri_memory.indexBuffer[meshID].d_pointer();
            d_sbt_indices[meshID] = tri_memory.sbtIndexBuffer[meshID].d_pointer();

            triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
            triangleInput[meshID].triangleArray.numVertices = (int) mesh.vertices3d.size();
            triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

            triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(int3);
            triangleInput[meshID].triangleArray.numIndexTriplets = (int) mesh.indices.size();
            triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];
            uint32_t triangleInputFlags[FacetType::FACET_TYPE_COUNT] = {
                    // one for every FacetType SBT
                    OPTIX_GEOMETRY_FLAG_NONE | OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
#ifdef WITH_TRANS
                    , OPTIX_GEOMETRY_FLAG_NONE | OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
#endif// WITH_TRANS

            };

            // in this example we have one SBT entry, and no per-primitive
            // materials:
            triangleInput[meshID].triangleArray.flags = triangleInputFlags;//&triangleInputFlags[meshID];
            triangleInput[meshID].triangleArray.numSbtRecords = flowgpu::FacetType::FACET_TYPE_COUNT;
#ifdef WITH_TRANS
            triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = d_sbt_indices[meshID];
            triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(flowgpu::FacetType);
            triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(flowgpu::FacetType);
#else
            triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
#endif
        }
        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
                                  | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                                  | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
                            (state.context,
                             &accelOptions,
                             triangleInput.data(),
                             (int) model->triangle_meshes.size(),  // num_build_inputs
                             &blasBufferSizes
                            ));

        // ==================================================================
        // prepare compaction
        // ==================================================================

        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.d_pointer();

        // ==================================================================
        // execute build (main stage)
        // ==================================================================

        CUDABuffer tempBuffer;
        tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(state.context,
                                    nullptr,
                                    &accelOptions,
                                    triangleInput.data(),
                                    (int) model->triangle_meshes.size(),
                                    tempBuffer.d_pointer(),
                                    tempBuffer.sizeInBytes,

                                    outputBuffer.d_pointer(),
                                    outputBuffer.sizeInBytes,

                                    &asHandle,

                                    &emitDesc, 1
        ));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // perform compaction
        // ==================================================================
        uint64_t compactedSize;
        compactedSizeBuffer.download(&compactedSize, 1);

        state.asBuffer.alloc(compactedSize);
        OPTIX_CHECK(optixAccelCompact(state.context,
                                      nullptr,
                                      asHandle,
                                      state.asBuffer.d_pointer(),
                                      state.asBuffer.sizeInBytes,
                                      &asHandle));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // aaaaaand .... clean up
        // ==================================================================
        outputBuffer.free(); // << the UNcompacted, temporary output buffer
        tempBuffer.free();
        compactedSizeBuffer.free();

        return asHandle;
    }

    /*! helper function that initializes optix and checks for errors */
    void SimulationOptiX::initOptix() {
        Log::console_msg(4, "#flowgpu: initializing optix...\n");

        // -------------------------------------------------------
        // check for available optix7 capable devices
        // -------------------------------------------------------
        cudaFree(nullptr);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
            throw std::runtime_error("#flowgpu: no CUDA capable devices found!");
        Log::console_msg(4, "#flowgpu: found {} CUDA devices\n", numDevices);

        // -------------------------------------------------------
        // initialize optix
        // -------------------------------------------------------
        OPTIX_CHECK(optixInit());
        Log::console_msg(4, "{}#flowgpu: successfully initialized optix... yay!{}\n",
                         MF_TERMINAL_GREEN, MF_TERMINAL_DEFAULT);
    }

    static void context_log_cb(unsigned int level,
                               const char *tag,
                               const char *message,
                               void *) {
        fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
    }

    void printDevProp(cudaDeviceProp devProp)
    {   printf("%s\n", devProp.name);
        printf("Major revision number:         %d\n", devProp.major);
        printf("Minor revision number:         %d\n", devProp.minor);
        printf("Total global memory:           %zu", devProp.totalGlobalMem);
        printf(" bytes\n");
        printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
        printf("Total amount of shared memory per block: %zu\n",devProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n", devProp.regsPerBlock);
        printf("Warp size:                     %d\n", devProp.warpSize);
        printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
        printf("Total amount of constant memory:         %zu\n",   devProp.totalConstMem);
    }

    // from https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
    int getSPcores(cudaDeviceProp devProp)
    {
        int cores = 0;
        int mp = devProp.multiProcessorCount;
        switch (devProp.major){
            case 2: // Fermi
                if (devProp.minor == 1) cores = mp * 48;
                else cores = mp * 32;
                break;
            case 3: // Kepler
                cores = mp * 192;
                break;
            case 5: // Maxwell
                cores = mp * 128;
                break;
            case 6: // Pascal
                if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
                else if (devProp.minor == 0) cores = mp * 64;
                else printf("Unknown device type\n");
                break;
            case 7: // Volta and Turing
                if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
                else printf("Unknown device type\n");
                break;
            case 8: // Ampere
                if (devProp.minor == 0) cores = mp * 64;
                else if (devProp.minor == 6) cores = mp * 128;
                else printf("Unknown device type\n");
                break;
            default:
                printf("Unknown device type\n");
                break;
        }
        return cores;
    }

    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void SimulationOptiX::createContext() {
        // for this sample, do everything on one device
        const int deviceID = 0;
        CUDA_CHECK(cudaSetDevice(deviceID));
        CUDA_CHECK(cudaStreamCreate(&state.stream));

#ifdef MULTI_STREAMS
        CUDA_CHECK(cudaStreamCreate(&state.stream2));
        state.cuStreams.resize(8);
        for(auto& stream : state.cuStreams){
            CUDA_CHECK(cudaStreamCreate(&stream));
        }
#endif
        cudaGetDeviceProperties(&state.deviceProps, deviceID);
        Log::console_msg(4, "#flowgpu: running on device: {}",state.deviceProps.name);
        Log::console_msg(4, " with {} cores\n", getSPcores(state.deviceProps));

        CUresult cuRes = cuCtxGetCurrent(&state.cudaContext);
        if (cuRes != CUDA_SUCCESS)
            Log::console_error("Error querying current context: error code {}\n", (size_t)cuRes);

        OPTIX_CHECK(optixDeviceContextCreate(state.cudaContext, nullptr, &state.context));
        OPTIX_CHECK(optixDeviceContextSetLogCallback
                            (state.context, context_log_cb, nullptr, 4));

    }


    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void SimulationOptiX::createModule() {
        state.moduleCompileOptions = {};
        state.moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT; // Do not limit the amount of registers.
#ifdef DEBUG
        state.moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        state.moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#if (OPTIX_VERSION >= 70400)
     state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
        state.moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        state.pipelineCompileOptions = {};
        state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        state.pipelineCompileOptions.usesMotionBlur = 0;
#if defined(PAYLOAD_DIRECT)
        state.pipelineCompileOptions.numPayloadValues = 8; // values that get send as PerRayData
#else
        state.pipelineCompileOptions.numPayloadValues = 2; // just a packed pointer, send as PerRayData
#endif
#if defined(WITHTRIANGLES)
        state.pipelineCompileOptions.numAttributeValues = 2; // default, don't have custom routines
        state.pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
#else
        state.pipelineCompileOptions.numAttributeValues = 5; // ret values e.g. by optixReportIntersection: n(x,y,z),u,v
        state.pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
#endif
#if !defined(DEBUG)
        state.pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
#else // DEBUG
        state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                                      OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                                      OPTIX_EXCEPTION_FLAG_USER |
                                                      OPTIX_EXCEPTION_FLAG_DEBUG;
#endif
        state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        state.pipelineLinkOptions = {};


        //state.pipelineLinkOptions.overrideUsesMotionBlur = false; // Removed with Optix7.1
        state.pipelineLinkOptions.maxTraceDepth = (state.launchParams.simConstants.maxDepth == 0) ? 2 : state.launchParams.simConstants.maxDepth+1;
#if defined(NDEBUG)
        // Keep generated line info for Nsight Compute profiling. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
#if (OPTIX_VERSION >= 70400)
        state.pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        state.pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
#else // DEBUG
        state.pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

        char log[2048];
        size_t sizeof_log = sizeof(log);

#if !defined(WITHTRIANGLES)
        {
            const std::string ptxCode = readPTX("./flowgpu_ptx/Geometry.ptx");
            //const std::string ptxCode = geometry_ptx_code;
            OPTIX_CHECK(optixModuleCreateFromPTX(state.context,
                                                 &state.moduleCompileOptions,
                                                 &state.pipelineCompileOptions,
                                                 ptxCode.c_str(),
                                                 ptxCode.size(),
                                                 log, &sizeof_log,
                                                 &state.modules.geometryModule
            ));
            //if (sizeof_log > 1) PRINT(log);
        }
#endif
        {
            const std::string ptxCode = readPTX("./flowgpu_ptx/RayGeneration.ptx");
            OPTIX_CHECK(optixModuleCreateFromPTX(state.context,
                                                 &state.moduleCompileOptions,
                                                 &state.pipelineCompileOptions,
                                                 ptxCode.c_str(),
                                                 ptxCode.size(),
                                                 log, &sizeof_log,
                                                 &state.modules.rayModule
            ));
            //if (sizeof_log > 1) PRINT(log);
        }

        {
#ifdef WITHTRIANGLES
            std::string ptxFile = "./flowgpu_ptx/TraceProcessing.ptx";
#else
            std::string ptxFile = "./flowgpu_ptx/TraceProcessing_polygon.ptx";
#endif
            const std::string ptxCode = readPTX(ptxFile);

            OPTIX_CHECK(optixModuleCreateFromPTX(state.context,
                                                 &state.moduleCompileOptions,
                                                 &state.pipelineCompileOptions,
                                                 ptxCode.c_str(),
                                                 ptxCode.size(),
                                                 log, &sizeof_log,
                                                 &state.modules.traceModule
            ));
            //if (sizeof_log > 1) PRINT(log);
        }

#if defined(DEBUG)
        {
            const std::string ptxCode = readPTX("./flowgpu_ptx/Exception.ptx");
            OPTIX_CHECK(optixModuleCreateFromPTX(state.context,
                                                 &state.moduleCompileOptions,
                                                 &state.pipelineCompileOptions,
                                                 ptxCode.c_str(),
                                                 ptxCode.size(),
                                                 log, &sizeof_log,
                                                 &state.modules.exceptionModule
            ));
            //if (sizeof_log > 1) PRINT(log);
        }
#endif
    }


    /*! does all setup for the raygen program(s) we are going to use */
    void SimulationOptiX::createRaygenPrograms(std::vector<OptixProgramGroup> &programGroups) {
        // we do a single ray gen program in this example:
        //raygenPGs.resize(1);
        OptixProgramGroup pgRayGen;
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = state.modules.rayModule;
        pgDesc.raygen.entryFunctionName = "__raygen__startFromSource";

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeof_log,
                                            &pgRayGen
        ));
        //if (sizeof_log > 1) PRINT(log);

        programGroups.push_back(pgRayGen);
        state.raygenPG = pgRayGen;
    }

    /*! does all setup for the miss program(s) we are going to use */
    void SimulationOptiX::createMissPrograms(std::vector<OptixProgramGroup> &programGroups) {
        // we do a single ray gen program in this example:
        //missPGs.resize(1);
        OptixProgramGroup pgMiss;
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = state.modules.traceModule;
        pgDesc.miss.entryFunctionName = "__miss__molecule";

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeof_log,
                                            &pgMiss
        ));
        //if (sizeof_log >log 1) PRINT(log);

        programGroups.push_back(pgMiss);
        state.missPG = pgMiss;
    }

    /*! does all setup for the hitgroup program(s) we are going to use */
    void SimulationOptiX::createHitgroupPrograms(std::vector<OptixProgramGroup> &programGroups) {
        char log[2048];
        size_t sizeof_log = sizeof(log);
        // We create one hitgroup program per ray type and facet combo
        std::vector<OptixProgramGroup> pgHitgroup;
        pgHitgroup.resize(RayType::RAY_TYPE_COUNT * FacetType::FACET_TYPE_COUNT);
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
#ifdef WITHTRIANGLES
        // use inbuilt IS routine
        pgDesc.hitgroup.moduleCH            = state.modules.traceModule;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__molecule_triangle";
#else
        pgDesc.hitgroup.moduleIS = state.modules.geometryModule;
        pgDesc.hitgroup.entryFunctionNameIS = "__intersection__polygon";
        pgDesc.hitgroup.moduleCH = state.modules.traceModule;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__molecule_polygon";
#endif

        pgDesc.hitgroup.moduleAH = state.modules.traceModule;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__molecule";

        try {
            OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                                &pgDesc,
                                                1,
                                                &pgOptions,
                                                log, &sizeof_log,
                                                &pgHitgroup[FacetType::FACET_TYPE_SOLID]
            ));
        }
        catch (std::exception &ex) {
            std::cerr << ex.what() << std::endl;
            ProcessSleep(10000);
        }
        //if (sizeof_log > 1) PRINT(log);
        programGroups.push_back(pgHitgroup[FacetType::FACET_TYPE_SOLID]);

#ifdef WITH_TRANS
#ifdef WITHTRIANGLES
        // use inbuilt IS routine
        pgDesc.hitgroup.moduleCH            = state.modules.traceModule;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__transparent_triangle";
#else
        pgDesc.hitgroup.moduleCH = state.modules.traceModule;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__transparent";
        std::cerr << "Transparent polygons (nbVert > 3) not yet supported!\n");
#endif

        OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeof_log,
                                            &pgHitgroup[FacetType::FACET_TYPE_TRANS]
        ));
        //if (sizeof_log > 1) PRINT(log);
        programGroups.push_back(pgHitgroup[FacetType::FACET_TYPE_TRANS]);
#endif// WITH_TRANS

        state.hitgroupPG.insert(state.hitgroupPG.end(), pgHitgroup.begin(), pgHitgroup.end());
    }

#if defined(DEBUG)
/*! does all setup for the hitgroup program(s) we are going to use */
    void SimulationOptiX::createExceptionPrograms(std::vector<OptixProgramGroup> &programGroups) {
        char log[2048];
        size_t sizeof_log = sizeof(log);
        // We create one hitgroup program per ray type and facet combo
        OptixProgramGroup pgExgroup;
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};

        pgDesc.kind  = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        pgDesc.exception.module            = state.modules.exceptionModule;
        pgDesc.exception.entryFunctionName = "__exception__all";

        try {
            OPTIX_CHECK(optixProgramGroupCreate(state.context,
                                                &pgDesc,
                                                1,
                                                &pgOptions,
                                                log, &sizeof_log,
                                                &pgExgroup
            ));
        }
        catch (std::exception &ex) {
            std::cerr << ex.what() << std::endl;
            ProcessSleep(10000);
        }
        //if (sizeof_log > 1) PRINT(log);
        programGroups.push_back(pgExgroup);
        state.exceptionPG = pgExgroup;
    }
#endif

    /*! assembles the full pipeline of all programs */
    void SimulationOptiX::createPipeline(std::vector<OptixProgramGroup> &programGroups) {
        /*std::vector<OptixProgramGroup> programGroups;
        for (auto pg : raygenPGs)
            programGroups.push_back(pg);
        for (auto pg : missPGs)
            programGroups.push_back(pg);
        for (auto pg : hitgroupPGs)
            programGroups.push_back(pg);*/

        char log[2048];
        size_t sizeof_log = sizeof(log);
        try {
            OPTIX_CHECK(optixPipelineCreate(state.context,
                                            &state.pipelineCompileOptions,
                                            &state.pipelineLinkOptions,
                                            programGroups.data(),
                                            static_cast<unsigned int>( programGroups.size()),
                                            log, &sizeof_log,
                                            &state.pipeline
            ));
        }
        catch (std::exception &e) {
            std::cerr << "[optixPipelineCreate] " << e.what() << std::endl;
            ProcessSleep(1000 * 100);
        }
        //if (sizeof_log > 1) PRINT(log);

        try {
            OPTIX_CHECK(optixPipelineSetStackSize
                                (/* [in] The pipeline to configure the stack size for */
                                        state.pipeline,
                                        /* [in] The direct stack size requirement for direct
                                           callables invoked from IS or AH. */
                                        2 * 1024,
                                        /* [in] The direct stack size requirement for direct
                                           callables invoked from RG, MS, or CH.  */
                                        2 * 1024,
                                        /* [in] The continuation stack requirement. */
                                        2 * 1024,
                                        /* [in] The maximum depth of a traversable graph
                                           passed to trace. */
                                        1));
        }
        catch (std::exception &e) {
            std::cerr << "[optixPipelineSetStackSize] " << e.what() << std::endl;
            ProcessSleep(1000 * 100);
        }
        //if (sizeof_log > 1) PRINT(log);
    }


    /*! constructs the shader binding table */
    void SimulationOptiX::buildSBTPolygon() {
        // first allocate device memory and upload data

        sim_memory.moleculeBuffer.initDeviceData(
                state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y * sizeof(MolPRD));

        for (int meshID = 0; meshID < (int) model->poly_meshes.size(); meshID++) {
            PolygonMesh &mesh = *model->poly_meshes[meshID];
            poly_memory.vertexBuffer[meshID].alloc_and_upload(mesh.vertices3d);
            poly_memory.vertex2Buffer[meshID].alloc_and_upload(mesh.vertices2d);
            poly_memory.vertex2x64Buffer[meshID].alloc_and_upload(mesh.vertices2d64);
            poly_memory.indexBuffer[meshID].alloc_and_upload(mesh.indices);
            poly_memory.polyBuffer[meshID].alloc_and_upload(mesh.poly);
            //poly_memory.cdfBuffer[meshID].alloc_and_upload(mesh.cdfs_1);
            poly_memory.facprobBuffer[meshID].alloc_and_upload(mesh.facetProbabilities);
        }

        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        //std::vector<RaygenRecord> raygenRecords;
        {
            RaygenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenPG, &rec));
            rec.data.vertex = (float3 *) poly_memory.vertexBuffer[0].d_pointer();
            rec.data.vertex2 = (float2 *) poly_memory.vertex2Buffer[0].d_pointer();
            rec.data.vertex2x64 = (double2 *) poly_memory.vertex2x64Buffer[0].d_pointer();
            rec.data.index = (uint32_t *) poly_memory.indexBuffer[0].d_pointer();
            rec.data.poly = (flowgpu::Polygon *) poly_memory.polyBuffer[0].d_pointer();
            rec.data.cdfs = (float *) poly_memory.cdfBuffer[0].d_pointer();
            rec.data.facetProbabilities = (float2 *) poly_memory.facprobBuffer[0].d_pointer();
            //raygenRecords.push_back(rec);
            sbt_memory.raygenRecordsBuffer.alloc(sizeof(rec));
            sbt_memory.raygenRecordsBuffer.upload(&rec, 1);
        }
        state.sbt.raygenRecord = sbt_memory.raygenRecordsBuffer.d_pointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        int numObjects = (int) model->poly_meshes.size();
        //std::vector<MissRecord> missRecords;
        {
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.missPG, &rec));
            //rec.data = nullptr; /* for now ... */
            //rec.data_poly = nullptr; /* for now ... */
            rec.data.vertex = (float3 *) poly_memory.vertexBuffer[0].d_pointer();
            rec.data.vertex2 = (float2 *) poly_memory.vertex2Buffer[0].d_pointer();
            rec.data.vertex2x64 = (double2 *) poly_memory.vertex2x64Buffer[0].d_pointer();
            rec.data.index = (uint32_t *) poly_memory.indexBuffer[0].d_pointer();
            rec.data.poly = (flowgpu::Polygon *) poly_memory.polyBuffer[0].d_pointer();
            //missRecords.push_back(rec);
            sbt_memory.missRecordsBuffer.alloc(sizeof(rec));
            sbt_memory.missRecordsBuffer.upload(&rec, 1);
        }
        //sbt_memory.missRecordsBuffer.alloc_and_upload(missRecords);
        state.sbt.missRecordBase = sbt_memory.missRecordsBuffer.d_pointer();
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof(MissRecord));
        state.sbt.missRecordCount = RayType::RAY_TYPE_COUNT;

        // ------------------------------------------------------------------
        // build hitgroup records
        // ------------------------------------------------------------------
        std::vector<HitgroupRecord> hitgroupRecords;
        for (int meshID = 0; meshID < numObjects; meshID++) {
            HitgroupRecord rec;
            // all meshes use the same code, so all same hit group
            OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPG[FacetType::FACET_TYPE_SOLID], &rec));
            //rec.data.color  = gdt::float3(0.5,1.0,0.5);
            //rec.data.vertex = (float3*)vertexBuffer[meshID].d_pointer();
            rec.data.vertex = (float3 *) poly_memory.vertexBuffer[meshID].d_pointer();
            rec.data.vertex2 = (float2 *) poly_memory.vertex2Buffer[meshID].d_pointer();
            rec.data.vertex2x64 = (double2 *) poly_memory.vertex2x64Buffer[meshID].d_pointer();
            rec.data.index = (uint32_t *) poly_memory.indexBuffer[meshID].d_pointer();
            rec.data.poly = (flowgpu::Polygon *) poly_memory.polyBuffer[meshID].d_pointer();
            hitgroupRecords.push_back(rec);
#ifdef WITH_TRANS
            OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPG[FacetType::FACET_TYPE_TRANS], &rec));
            hitgroupRecords.push_back(rec); // TODO: for now use the same data for all facet types
#endif

            //sbt_memory.hitgroupRecordsBuffer.alloc(sizeof(rec));
            //sbt_memory.hitgroupRecordsBuffer.upload(&rec,1);
        }

        sbt_memory.hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        state.sbt.hitgroupRecordBase = sbt_memory.hitgroupRecordsBuffer.d_pointer();
        state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( sizeof(HitgroupRecord));
        state.sbt.hitgroupRecordCount = RayType::RAY_TYPE_COUNT * FacetType::FACET_TYPE_COUNT;

        // ------------------------------------------------------------------
        // build exception records
        // ------------------------------------------------------------------
#if defined(DEBUG)
        std::vector<ExceptionRecord> exceptionRecords;
        for (int meshID = 0; meshID < numObjects; meshID++) {
            ExceptionRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.exceptionPG, &rec));

            rec.data = 0;
            // all meshes use the same code, so all same hit group
            sbt_memory.exceptionRecordsBuffer.alloc(sizeof(rec));
            sbt_memory.exceptionRecordsBuffer.upload(&rec, 1);
        }
        state.sbt.exceptionRecord = sbt_memory.exceptionRecordsBuffer.d_pointer();
#endif
    }

    /*! constructs the shader binding table */
    void SimulationOptiX::buildSBTTriangle() {
        // first allocate device memory and upload data
        sim_memory.moleculeBuffer.initDeviceData(
                state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y * sizeof(MolPRD));

        for (int meshID = 0; meshID < (int) model->triangle_meshes.size(); meshID++) {
            TriangleMesh &mesh = *model->triangle_meshes[meshID];

            tri_memory.texcoordBuffer[meshID].alloc_and_upload(mesh.texCoords);
            tri_memory.polyBuffer[meshID].alloc_and_upload(mesh.poly);
            tri_memory.facprobBuffer[meshID].alloc_and_upload(mesh.facetProbabilities);
        }

        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<RaygenRecordTri> raygenRecords;
        {
            RaygenRecordTri rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenPG, &rec));
            rec.data.vertex = (float3 *) tri_memory.vertexBuffer[0].d_pointer();
            rec.data.index = (int3 *) tri_memory.indexBuffer[0].d_pointer();
            rec.data.poly = (flowgpu::Polygon *) tri_memory.polyBuffer[0].d_pointer();
            rec.data.facetProbabilities = (float2 *) tri_memory.facprobBuffer[0].d_pointer();
            //raygenRecords.push_back(rec);
            sbt_memory.raygenRecordsBuffer.alloc(sizeof(rec));
            sbt_memory.raygenRecordsBuffer.upload(&rec, 1);
        }
        //state.raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        state.sbt.raygenRecord = sbt_memory.raygenRecordsBuffer.d_pointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        int numObjects = (int) model->triangle_meshes.size();
        std::vector<MissRecordTri> missRecords;
        {
            MissRecordTri rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.missPG, &rec));
            //rec.data = nullptr; /* for now ... */
            //rec.data_poly = nullptr; /* for now ... */
            rec.data.vertex = (float3 *) tri_memory.vertexBuffer[0].d_pointer();
            rec.data.index = (int3 *) tri_memory.indexBuffer[0].d_pointer();
            rec.data.poly = (flowgpu::Polygon *) tri_memory.polyBuffer[0].d_pointer();
            //rec.data.texcoord = (float2 *) tri_memory.texcoordBuffer[0].d_pointer();

            //missRecords.push_back(rec);
            sbt_memory.missRecordsBuffer.alloc(sizeof(rec));
            sbt_memory.missRecordsBuffer.upload(&rec, 1);
        }
        //missRecordsBuffer.alloc_and_upload(missRecords);
        state.sbt.missRecordBase = sbt_memory.missRecordsBuffer.d_pointer();
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof(MissRecordTri));
        state.sbt.missRecordCount = RayType::RAY_TYPE_COUNT;

        // ------------------------------------------------------------------
        // build hitgroup records
        // ------------------------------------------------------------------
        std::vector<HitgroupRecordTri> hitgroupRecords;
        hitgroupRecords.resize(RayType::RAY_TYPE_COUNT * FacetType::FACET_TYPE_COUNT);
        for (int meshID = 0; meshID < numObjects; meshID++) {
            // all meshes use the same code, so all same hit group
            OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPG[FacetType::FACET_TYPE_SOLID],
                                                 &hitgroupRecords[FacetType::FACET_TYPE_SOLID]));
            //rec.data.color  = gdt::float3(0.5,1.0,0.5);
            //rec.data.vertex = (float3*)vertexBuffer[meshID].d_pointer();
            hitgroupRecords[FacetType::FACET_TYPE_SOLID].data.vertex = (float3 *) tri_memory.vertexBuffer[meshID].d_pointer();
            hitgroupRecords[FacetType::FACET_TYPE_SOLID].data.index = (int3 *) tri_memory.indexBuffer[meshID].d_pointer();
            hitgroupRecords[FacetType::FACET_TYPE_SOLID].data.poly = (flowgpu::Polygon *) tri_memory.polyBuffer[meshID].d_pointer();
            hitgroupRecords[FacetType::FACET_TYPE_SOLID].data.texcoord =  (float2 *) tri_memory.texcoordBuffer[meshID].d_pointer();

            //hitgroupRecords.push_back(rec);

#ifdef WITH_TRANS
            OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPG[FacetType::FACET_TYPE_TRANS],
                                                 &hitgroupRecords[FacetType::FACET_TYPE_TRANS]));
            hitgroupRecords[FacetType::FACET_TYPE_TRANS].data.vertex = (float3 *) tri_memory.vertexBuffer[meshID].d_pointer();
            hitgroupRecords[FacetType::FACET_TYPE_TRANS].data.index = (int3 *) tri_memory.indexBuffer[meshID].d_pointer();
            hitgroupRecords[FacetType::FACET_TYPE_TRANS].data.poly = (flowgpu::Polygon *) tri_memory.polyBuffer[meshID].d_pointer();
            hitgroupRecords[FacetType::FACET_TYPE_TRANS].data.texcoord =  (float2 *) tri_memory.texcoordBuffer[meshID].d_pointer();

            //hitgroupRecords.push_back(rec); // TODO: for now use the same data for all facet types
#endif
        }
        //sbt_memory.hitgroupRecordsBuffer.alloc(RayType::RAY_TYPE_COUNT * FacetType::FACET_TYPE_COUNT * sizeof(rec));
        //sbt_memory.hitgroupRecordsBuffer.upload(&rec,RayType::RAY_TYPE_COUNT * FacetType::FACET_TYPE_COUNT);
        sbt_memory.hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);

        state.sbt.hitgroupRecordBase = sbt_memory.hitgroupRecordsBuffer.d_pointer();
        state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( sizeof(HitgroupRecordTri));
        state.sbt.hitgroupRecordCount = RayType::RAY_TYPE_COUNT * FacetType::FACET_TYPE_COUNT;
    }


    /*! upload some parts only on start */
    void SimulationOptiX::initSimulation() {
        const uint32_t launchSize = state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y;
        const uint32_t nbHCBins = EXTRAFACETCOUNTERS;

        CuFacetHitCounter *hitCounter = new CuFacetHitCounter[nbHCBins * state.launchParams.simConstants.nbFacets]();
        uint32_t *missCounter = new uint32_t[state.launchParams.simConstants.nbFacets];
        std:memset(missCounter, 0, state.launchParams.simConstants.nbFacets * sizeof(uint32_t));
        facet_memory.hitCounterBuffer.upload(hitCounter, nbHCBins * state.launchParams.simConstants.nbFacets);
        facet_memory.missCounterBuffer.upload(missCounter, state.launchParams.simConstants.nbFacets);

        Log::console_msg(4, "nbTextures {}\n",model->facetTex.size());
        Log::console_msg(4, "nbTexels {}\n",model->textures.size());
        Log::console_msg(4, "nbTexInc {}\n",model->texInc.size());

        delete[] hitCounter;
        delete[] missCounter;

#ifdef DEBUGCOUNT
        uint32_t *detVal = new uint32_t[NCOUNTBINS]();
        uint32_t *uVal = new uint32_t[NCOUNTBINS]();
        uint32_t *vVal = new uint32_t[NCOUNTBINS]();

        memory_debug.detBuffer.upload(detVal,NCOUNTBINS*1);
        memory_debug.uBuffer.upload(uVal,NCOUNTBINS*1);
        memory_debug.vBuffer.upload(vVal,NCOUNTBINS*1);

        delete[] detVal;
        delete[] uVal;
        delete[] vVal;
#endif
#ifdef DEBUGPOS
        float3 *pos = new float3[NBPOSCOUNTS * 1]();
        uint32_t *offset = new uint32_t[1]();
        uint16_t *posTypes = new uint16_t[NBPOSCOUNTS * 1]();
        memory_debug.posBuffer.upload(pos, NBPOSCOUNTS * 1);
        memory_debug.posOffsetBuffer.upload(offset, 1);
        memory_debug.posTypeBuffer.upload(posTypes, NBPOSCOUNTS * 1);

        delete[] pos;
        delete[] offset;
        delete[] posTypes;

#endif
#ifdef DEBUGLEAKPOS
        float3 *leakPos = new float3[NBCOUNTS * state.launchParams.simConstants.size.x *
                                     state.launchParams.simConstants.size.y]();
        float3 *leakDir = new float3[NBCOUNTS * state.launchParams.simConstants.size.x *
                                     state.launchParams.simConstants.size.y]();
        uint32_t *leakOffset = new uint32_t[state.launchParams.simConstants.size.x *
                                            state.launchParams.simConstants.size.y]();
        memory_debug.leakPosBuffer.upload(leakPos, NBCOUNTS * state.launchParams.simConstants.size.x *
                                                   state.launchParams.simConstants.size.y * 1);
        memory_debug.leakDirBuffer.upload(leakDir, NBCOUNTS * state.launchParams.simConstants.size.x *
                                                   state.launchParams.simConstants.size.y * 1);
        memory_debug.leakPosOffsetBuffer.upload(leakOffset, state.launchParams.simConstants.size.x *
                                                            state.launchParams.simConstants.size.y * 1);

        delete[] leakPos;
        delete[] leakDir;
        delete[] leakOffset;
#endif
#ifdef DEBUGMISS
        uint32_t *miss = new uint32_t[NMISSES * state.launchParams.simConstants.size.x *
                                      state.launchParams.simConstants.size.y]();
        memory_debug.missBuffer.upload(miss, NMISSES * state.launchParams.simConstants.size.x *
                                             state.launchParams.simConstants.size.y * 1);

        delete[] miss;
#endif
        state.launchParamsBuffer.upload(&state.launchParams, 1);
    }

    /*! render one frame */
    void SimulationOptiX::launchMolecules() {
        // Initialize RNG
        //const uint32_t launchSize = state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y;
        //const uint32_t nbRand  = 10*launchSize;
        //float *randomNumbers = new float[nbRand];
        //CuFacetHitCounter *hitCounter = new CuFacetHitCounter[launchSize*state.launchParams.simConstants.nbFacets]();

        //RandWrapper::getRand(randomNumbers, nbRand);
        //curandState_t *states;
        //crng::initializeRand(state.launchParams.frame.size.x*state.launchParams.frame.size.y, states, randomNumbers);
        //crng::generateRand(state.launchParams.frame.size.x*state.launchParams.frame.size.y, states, randomNumbers);
        //state.launchParams.randomNumbers = randomNumbers;
        //randBuffer.upload(randomNumbers,nbRand);
        //hitCounterBuffer.upload(hitCounter,launchSize*state.launchParams.simConstants.nbFacets);
        //state.launchParamsBuffer.upload(&launchParams,1);

        /*for(auto& stream : state.cuStreams)
            OPTIX_CHECK(optixLaunch(
                                            state.pipeline,stream,

                                            state.launchParamsBuffer.d_pointer(),
                                    state.launchParamsBuffer.sizeInBytes,
                                    &state.sbt,

                                            state.launchParams.simConstants.size.x/state.cuStreams.size(),
                                    state.launchParams.simConstants.size.y,
                                    1
            ));*/
try{
        OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                state.pipeline, state.stream,
                /*! parameters and SBT */
                state.launchParamsBuffer.d_pointer(),
                state.launchParamsBuffer.sizeInBytes,
                &state.sbt,
                /*! dimensions of the launch: */
                settings->kernelDimensions[0],
                settings->kernelDimensions[1],/*
                state.launchParams.simConstants.size.x,
                state.launchParams.simConstants.size.y,*/
                1
        ));

        /*OPTIX_CHECK(optixLaunch(*//*! pipeline we're launching launch: *//*
                state.pipeline,state.stream2,
                *//*! parameters and SBT *//*
                state.launchParamsBuffer.d_pointer(),
                state.launchParamsBuffer.sizeInBytes,
                &state.sbt,
                *//*! dimensions of the launch: *//*
                state.launchParams.simConstants.size.x/2,
                state.launchParams.simConstants.size.y,
                1
        ));*/

        // sync - make sure the frame is rendered before we download and
        // display (obviously, for a high-performance application you
        // want to use streams and double-buffering, but for this simple
        // example, this will have to do)
        CUDA_SYNC_CHECK();
    }
    catch (std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        ProcessSleep(10000);
    }
        //delete[] randomNumbers;
    }

    void SimulationOptiX::generateRand() {
        /*const uint32_t launchSize = state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y;
        const short nbRand  = NB_RAND*launchSize;
        float *randomNumbers = new float[nbRand];
        RandWrapper::getRand(randomNumbers, nbRand);
        uint32_t *randomOffset = new uint32_t[launchSize]();

        //curandState_t *states;
        //crng::initializeRand(state.launchParams.frame.size.x*state.launchParams.frame.size.y, states, randomNumbers);
        //crng::generateRand(state.launchParams.frame.size.x*state.launchParams.frame.size.y, states, randomNumbers);
        //state.launchParams.randomNumbers = randomNumbers;
        randBuffer.upload(randomNumbers,nbRand);
        delete[] randomNumbers;
        delete[] randomOffset;*/

#ifdef RNG_BULKED
        /*const unsigned int launchSize = state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y;
        const unsigned int nbRandperThread = NB_RAND(settings->cyclesRNG, state.launchParams.simConstants.maxDepth);
        //const unsigned int nbRand = nbRandperThread * launchSize;
        //CUDABuffer stateBuff, randomBuff;
        //curandState_t *states;
        //TODO: Try upload or host API
        //printf( "Pre : %p --> %p \n", (void*)randBuffer.d_ptr, (void*)&randBuffer.d_ptr);
        //crng::testRand(&randomBuff.d_ptr, launchSize);
        //crng::initializeRandHost(launchSize, (float**) &randomBuff.d_ptr);
        crng::generateRandHost(launchSize, (RN_T *) sim_memory.randBuffer.d_ptr, nbRandperThread);
        //printf( "Post: %p --> %p --> %p\n", (void*)randBuffer.d_ptr, (void*)&randBuffer.d_ptr, randBuffer.d_pointer());

        //std::cout<< " --- print Rand --- \n");
        //crng::printDevDataAtHost(sim_memory.randBuffer.d_ptr, nbRand);
        //std::cout<< " --- ---------- --- \n");
        //crng::initializeRand(launchSize, stateBuff.d_ptr, randomBuff.d_ptr);
        //crng::generateRand(launchSize, (curandState_t *)stateBuff.d_ptr, (float*)randomBuff.d_ptr);

        state.launchParams.randomNumbers = (RN_T *) sim_memory.randBuffer.d_ptr;
        //randBuffer.upload(randomNumbers,nbRand);

        crng::offsetBufferZeroInit(launchSize, (void *) sim_memory.randOffsetBuffer.d_ptr);

        //state.launchParamsBuffer.upload(&launchParams,1);*/
        const unsigned int launchSize = state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y;
        const unsigned int nbRandperThread = NB_RAND(settings->cyclesRNG, state.launchParams.simConstants.maxDepth);

        crng::generateRandHostAndBuffer(launchSize, static_cast<RN_T*>(sim_memory.randBuffer.d_ptr), nbRandperThread, static_cast<unsigned int*>(sim_memory.randOffsetBuffer.d_ptr));
        state.launchParams.randomNumbers = static_cast<RN_T*>(sim_memory.randBuffer.d_ptr);
#else
        state.launchParams.randomNumbers = static_cast<curandState_t*>(sim_memory.randBuffer.d_ptr);
#endif


    }


    void SimulationOptiX::resetDeviceData(const unsigned int newSize_[2]) {
        uint2 newSize = make_uint2(newSize_[0],newSize_[1]);

#ifdef RNG_BULKED
        const uint32_t nbRand = NB_RAND(settings->cyclesRNG, state.launchParams.simConstants.maxDepth);
#else
        const uint32_t nbRand = NB_RAND(1, state.launchParams.simConstants.maxDepth);
#endif
        // resize our cuda frame buffer
        if (!sim_memory.moleculeBuffer.d_pointer())
            return;

        sim_memory.moleculeBuffer.initDeviceData(newSize.x * newSize.y * sizeof(MolPRD));

        if (!model->cdfs_1.empty() && model->cdfs_1.size() == model->cdfs_2.size()) {
            facet_memory.cdf1Buffer.upload(model->cdfs_1.data(), model->cdfs_1.size());
            facet_memory.cdf2Buffer.upload(model->cdfs_2.data(), model->cdfs_2.size());
        }

        // Texture
        if (!model->textures.empty()) {
            facet_memory.textureBuffer.upload(model->facetTex.data(), model->facetTex.size());
            facet_memory.texelBuffer.upload(model->textures.data(), model->textures.size());
            facet_memory.texIncBuffer.upload(model->texInc.data(), model->texInc.size());
        }

        // Profile
        if (!model->profiles.empty()) {
            facet_memory.profileBuffer.upload(model->profiles.data(), model->profiles.size());
        }

        state.launchParamsBuffer.upload(&state.launchParams, 1);
    }


    void SimulationOptiX::initLaunchParams(const uint2 &newSize) {
#ifdef RNG_BULKED
        const uint32_t nbRand = NB_RAND(settings->cyclesRNG, settings->recursiveMaxDepth);
#else
        const uint32_t nbRand = NB_RAND(1, settings->recursiveMaxDepth);
#endif
        // resize our cuda frame buffer
        // TODO: one counter per thread is a problem for memory
        sim_memory.moleculeBuffer.resize(newSize.x * newSize.y * sizeof(MolPRD));
        sim_memory.moleculeBuffer.initDeviceData(newSize.x * newSize.y * sizeof(MolPRD));
#ifdef RNG_BULKED
        sim_memory.randBuffer.resize(nbRand * newSize.x * newSize.y * sizeof(RN_T));
        sim_memory.randOffsetBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
#else
        sim_memory.randBuffer.resize(newSize.x * newSize.y * sizeof(curandState_t));
#endif
        facet_memory.hitCounterBuffer.resize(
                model->nbFacets_total * EXTRAFACETCOUNTERS * sizeof(CuFacetHitCounter));
        facet_memory.missCounterBuffer.resize(model->nbFacets_total * sizeof(uint32_t));
        //facet_memory.textureBuffer.resize(model->textures.size() * sizeof(TextureCell));

// Texture

        if (!model->cdfs_1.empty() && model->cdfs_1.size() == model->cdfs_2.size()) {
            facet_memory.cdf1Buffer.alloc_and_upload(model->cdfs_1);
            facet_memory.cdf2Buffer.alloc_and_upload(model->cdfs_2);
        }

        if (!model->textures.empty()) {
            facet_memory.textureBuffer.alloc_and_upload(model->facetTex);
            facet_memory.texelBuffer.alloc_and_upload(model->textures);
            facet_memory.texIncBuffer.alloc_and_upload(model->texInc);
        }

        // Profile
        if (!model->profiles.empty()) {
            facet_memory.profileBuffer.alloc_and_upload(model->profiles);
        }

        // update the launch parameters that we'll pass to the optix
        // launch:
        state.launchParams.simConstants.useMaxwell = model->wp.useMaxwellDistribution;//settings->useMaxwellDistribution;
        state.launchParams.simConstants.gasMass = model->wp.gasMass;//settings->gasMass;
        state.launchParams.simConstants.nbRandNumbersPerThread = nbRand;
        state.launchParams.simConstants.scene_epsilon = SCENE_EPSILON;
        state.launchParams.simConstants.maxDepth = settings->recursiveMaxDepth;
        state.launchParams.simConstants.size = newSize;
        state.launchParams.simConstants.nbFacets = model->nbFacets_total;
        // TODO: Total nb for indices and vertices
        state.launchParams.simConstants.nbVertices = model->nbVertices_total;
#ifdef BOUND_CHECK
        state.launchParams.simConstants.nbTexel = model->nbTexel_total;
        state.launchParams.simConstants.nbProfSlices = model->nbProfSlices_total;
#endif
        state.launchParams.simConstants.offset_center_magnitude = settings->offsetMagnitude;
        state.launchParams.simConstants.offset_normal_magnitude = settings->offsetMagnitudeN;

        state.launchParams.perThreadData.currentMoleculeData = (MolPRD *) sim_memory.moleculeBuffer.d_pointer();
        state.launchParams.perThreadData.randBufferOffset = (uint32_t *) sim_memory.randOffsetBuffer.d_pointer();
#ifdef RNG_BULKED
#ifdef DEBUG
        crng::initializeRandHost(newSize.x * newSize.y, (RN_T **) &sim_memory.randBuffer.d_ptr, nbRand);
#else
        crng::initializeRandHost(newSize.x * newSize.y, (RN_T **) &sim_memory.randBuffer.d_ptr,  nbRand, time(nullptr));
#endif // DEBUG
        state.launchParams.randomNumbers = (RN_T *) sim_memory.randBuffer.d_pointer();
#else
        crng::initializeRandDevice_ref(newSize.x * newSize.y, sim_memory.randBuffer.d_ptr,  time(nullptr));
        state.launchParams.randomNumbers = (curandState_t *) sim_memory.randBuffer.d_pointer();
        crng::generateRandDevice(newSize.x * newSize.y,(curandState_t *) sim_memory.randBuffer.d_pointer());
        crng::generateRandDevice(newSize.x * newSize.y,(curandState_t *) state.launchParams.randomNumbers);
#endif // RNG_BULKED
        //crng::initializeRandHost(newSize.x * newSize.y, (float **) &randBuffer.d_ptr);
        state.launchParams.hitCounter = (CuFacetHitCounter *) facet_memory.hitCounterBuffer.d_pointer();
        state.launchParams.sharedData.missCounter = (uint32_t *) facet_memory.missCounterBuffer.d_pointer();

        if (!facet_memory.textureBuffer.isNullptr())
            state.launchParams.sharedData.facetTextures = (flowgpu::FacetTexture *) facet_memory.textureBuffer.d_pointer();
        if (!facet_memory.texelBuffer.isNullptr())
            state.launchParams.sharedData.texels = (flowgpu::Texel *) facet_memory.texelBuffer.d_pointer();
        if (!facet_memory.texIncBuffer.isNullptr())
            state.launchParams.sharedData.texelInc = (float *) facet_memory.texIncBuffer.d_pointer();
        if (!facet_memory.profileBuffer.isNullptr())
            state.launchParams.sharedData.profileSlices = (flowgpu::Texel *) facet_memory.profileBuffer.d_pointer();
        if (!facet_memory.cdf1Buffer.isNullptr())
            state.launchParams.sharedData.cdfs1 = (float *) facet_memory.cdf1Buffer.d_pointer();
        if (!facet_memory.cdf2Buffer.isNullptr())
            state.launchParams.sharedData.cdfs2 = (float *) facet_memory.cdf2Buffer.d_pointer();
#ifdef DEBUGCOUNT
        memory_debug.detBuffer.resize(NCOUNTBINS*sizeof(uint32_t));
        memory_debug.uBuffer.resize(NCOUNTBINS*sizeof(uint32_t));
        memory_debug.vBuffer.resize(NCOUNTBINS*sizeof(uint32_t));

        state.launchParams.debugCounter.detCount  = (uint32_t*)memory_debug.detBuffer.d_pointer();
        state.launchParams.debugCounter.uCount  = (uint32_t*)memory_debug.uBuffer.d_pointer();
        state.launchParams.debugCounter.vCount  = (uint32_t*)memory_debug.vBuffer.d_pointer();
#endif

#ifdef DEBUGPOS
        memory_debug.posBuffer.resize(1 * NBPOSCOUNTS * sizeof(float3));
        memory_debug.posOffsetBuffer.resize(1 * sizeof(uint32_t));
        memory_debug.posTypeBuffer.resize(1 * NBPOSCOUNTS * sizeof(uint16_t));
        state.launchParams.perThreadData.positionsBuffer_debug = (float3 *) memory_debug.posBuffer.d_pointer();
        state.launchParams.perThreadData.posOffsetBuffer_debug = (uint32_t *) memory_debug.posOffsetBuffer.d_pointer();
        state.launchParams.perThreadData.positionsType_debug = (uint16_t *) memory_debug.posTypeBuffer.d_pointer();
#endif

#ifdef DEBUGLEAKPOS
        memory_debug.leakPosBuffer.resize(newSize.x * newSize.y * NBCOUNTS * sizeof(float3));
        memory_debug.leakDirBuffer.resize(newSize.x * newSize.y * NBCOUNTS * sizeof(float3));
        memory_debug.leakPosOffsetBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
        state.launchParams.perThreadData.leakPositionsBuffer_debug = (float3 *) memory_debug.leakPosBuffer.d_pointer();
        state.launchParams.perThreadData.leakDirectionsBuffer_debug = (float3 *) memory_debug.leakDirBuffer.d_pointer();
        state.launchParams.perThreadData.leakPosOffsetBuffer_debug = (uint32_t *) memory_debug.leakPosOffsetBuffer.d_pointer();
#endif

#ifdef DEBUGMISS
        memory_debug.missBuffer.resize(NMISSES * newSize.x * newSize.y * sizeof(uint32_t));
        state.launchParams.perThreadData.missBuffer = (uint32_t *) memory_debug.missBuffer.d_pointer();
#endif

        state.launchParamsBuffer.alloc(sizeof(state.launchParams));
        state.launchParamsBuffer.upload(&state.launchParams, 1);
    }

    /*! resize buffers to given amount of threads */
    // initlaunchparams
    void SimulationOptiX::resize(const uint2 &newSize) {
#ifdef RNG_BULKED
        const uint32_t nbRand = NB_RAND(settings->cyclesRNG, state.launchParams.simConstants.maxDepth);
#else
        const uint32_t nbRand = NB_RAND(1, state.launchParams.simConstants.maxDepth);
#endif
        state.launchParams.simConstants.size = newSize;

        // resize our cuda frame buffer
        sim_memory.moleculeBuffer.resize(newSize.x * newSize.y * sizeof(MolPRD));
        sim_memory.moleculeBuffer.initDeviceData(newSize.x * newSize.y * sizeof(MolPRD));
#ifdef RNG_BULKED
        sim_memory.randBuffer.resize(nbRand * newSize.x * newSize.y * sizeof(RN_T));
        sim_memory.randOffsetBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

        crng::destroyRandHost((RN_T **) &sim_memory.randBuffer.d_ptr);
#ifdef DEBUG
        crng::initializeRandHost(newSize.x * newSize.y, (RN_T **) &sim_memory.randBuffer.d_ptr, nbRand);
#else
        crng::initializeRandHost(newSize.x * newSize.y, (RN_T **) &sim_memory.randBuffer.d_ptr,  nbRand, time(nullptr));
#endif // DEBUG
#else // RNG_BULKED
        sim_memory.randBuffer.resize(newSize.x * newSize.y * sizeof(curandState_t));
        crng::destroyRandDevice((curandState_t **) &sim_memory.randBuffer.d_ptr);
        crng::initializeRandDevice_ref(newSize.x * newSize.y, sim_memory.randBuffer.d_ptr,  time(nullptr));
#endif // RNG_BULKED



#ifdef DEBUGPOS
        memory_debug.posBuffer.resize(1 * NBPOSCOUNTS * sizeof(float3));
        memory_debug.posOffsetBuffer.resize(1 * sizeof(uint32_t));
        memory_debug.posTypeBuffer.resize(1 * NBPOSCOUNTS * sizeof(uint16_t));
#endif

#ifdef DEBUGLEAKPOS
        memory_debug.leakPosBuffer.resize(newSize.x * newSize.y * NBCOUNTS * sizeof(float3));
        memory_debug.leakDirBuffer.resize(newSize.x * newSize.y * NBCOUNTS * sizeof(float3));
        memory_debug.leakPosOffsetBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
#endif
#ifdef DEBUGMISS
        memory_debug.missBuffer.resize(NMISSES * newSize.x * newSize.y * sizeof(uint32_t));
#endif


        state.launchParamsBuffer.upload(&state.launchParams, 1);
    }

    /*! download the rendered color buffer and return the total amount of hits (= followed rays) */
    void SimulationOptiX::downloadDataFromDevice(HostData *hostData) {
#ifdef WITHDESORPEXIT
        if (!sim_memory.moleculeBuffer.isNullptr())
            sim_memory.moleculeBuffer.download(hostData->hitData.data(), state.launchParams.simConstants.size.x *
                                                                         state.launchParams.simConstants.size.y);
#endif
        facet_memory.hitCounterBuffer.download(hostData->facetHitCounters.data(),
                                               model->nbFacets_total * EXTRAFACETCOUNTERS);
        facet_memory.missCounterBuffer.download(hostData->leakCounter.data(), model->nbFacets_total);

        if (!facet_memory.texelBuffer.isNullptr())
            facet_memory.texelBuffer.download(hostData->texels.data(), model->textures.size());

        if (!facet_memory.profileBuffer.isNullptr())
            facet_memory.profileBuffer.download(hostData->profileSlices.data(), model->profiles.size());

#ifdef DEBUGCOUNT
        memory_debug.detBuffer.download(hostData->detCounter.data(), NCOUNTBINS);
        memory_debug.uBuffer.download(hostData->uCounter.data(), NCOUNTBINS);
        memory_debug.vBuffer.download(hostData->vCounter.data(), NCOUNTBINS);
#endif

#ifdef DEBUGPOS
        memory_debug.posBuffer.download(hostData->positions.data(), NBPOSCOUNTS * 1);
        memory_debug.posOffsetBuffer.download(hostData->posOffset.data(), 1);
        memory_debug.posTypeBuffer.download(hostData->posType.data(), NBPOSCOUNTS * 1);
#endif
#ifdef DEBUGLEAKPOS
        memory_debug.leakPosBuffer.download(hostData->leakPositions.data(),
                                            NBCOUNTS * state.launchParams.simConstants.size.x *
                                            state.launchParams.simConstants.size.y);
        memory_debug.leakDirBuffer.download(hostData->leakDirections.data(),
                                            NBCOUNTS * state.launchParams.simConstants.size.x *
                                            state.launchParams.simConstants.size.y);
        memory_debug.leakPosOffsetBuffer.download(hostData->leakPosOffset.data(),
                                                  state.launchParams.simConstants.size.x *
                                                  state.launchParams.simConstants.size.y);
#endif
    }

    /*! download the rendered color buffer and return the total amount of hits (= followed rays) */
    void SimulationOptiX::resetDeviceBuffers() {
        //sim_memory.moleculeBuffer.download(hit, state.launchParams.simConstants.size.x * state.launchParams.simConstants.size.y);
        facet_memory.hitCounterBuffer.initDeviceData(
                model->nbFacets_total * EXTRAFACETCOUNTERS * sizeof(flowgpu::CuFacetHitCounter));
        facet_memory.missCounterBuffer.initDeviceData(model->nbFacets_total * sizeof(uint32_t));

        if (!facet_memory.texelBuffer.isNullptr())
            facet_memory.texelBuffer.initDeviceData(model->textures.size() * sizeof(flowgpu::Texel));
        if (!facet_memory.profileBuffer.isNullptr())
            facet_memory.profileBuffer.initDeviceData(model->profiles.size() * sizeof(flowgpu::Texel));
#ifdef DEBUGPOS
        memory_debug.posBuffer.initDeviceData(NBPOSCOUNTS * 1 * sizeof(float3));
        memory_debug.posOffsetBuffer.initDeviceData(1 * sizeof(uint32_t));
        memory_debug.posTypeBuffer.initDeviceData(NBPOSCOUNTS * 1 * sizeof(uint16_t));
#endif
    }

    void SimulationOptiX::updateHostData(HostData *tempData) {
#ifdef WITHDESORPEXIT
        sim_memory.moleculeBuffer.upload(tempData->hitData.data(), state.launchParams.simConstants.size.x *
                                                                   state.launchParams.simConstants.size.y);
#endif
    }

    void SimulationOptiX::cleanup() {
        OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(state.raygenPG));
        OPTIX_CHECK(optixProgramGroupDestroy(state.missPG));
        for (auto &hitPG : state.hitgroupPG)
            OPTIX_CHECK(optixProgramGroupDestroy(hitPG));
#if defined(DEBUG)
        OPTIX_CHECK(optixProgramGroupDestroy(state.exceptionPG));
#endif
        OPTIX_CHECK(optixModuleDestroy(state.modules.rayModule));
#if !defined(WITHTRIANGLES)
        OPTIX_CHECK(optixModuleDestroy(state.modules.geometryModule));
#endif
        OPTIX_CHECK(optixModuleDestroy(state.modules.traceModule));
#if defined(DEBUG)
        OPTIX_CHECK(optixModuleDestroy(state.modules.exceptionModule));
#endif
        OPTIX_CHECK(optixDeviceContextDestroy(state.context));

        CUDA_CHECK(cudaStreamDestroy(state.stream));
#ifdef MULTI_STREAMS
        CUDA_CHECK( cudaStreamDestroy( state.stream2                 ) );
        for(auto& stream : state.cuStreams){
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        state.cuStreams.clear();
#endif

        for (int meshID = 0; meshID < tri_memory.vertexBuffer.size(); meshID++) {
            tri_memory.vertexBuffer[meshID].free();
            tri_memory.texcoordBuffer[meshID].free();
            tri_memory.indexBuffer[meshID].free();
            tri_memory.sbtIndexBuffer[meshID].free();
            tri_memory.polyBuffer[meshID].free();
            tri_memory.facprobBuffer[meshID].free();
        }

        for (int meshID = 0; meshID < poly_memory.aabbBuffer.size(); meshID++) {
            if (!poly_memory.aabbBuffer.empty() && poly_memory.aabbBuffer.size() > meshID)
                poly_memory.aabbBuffer[meshID].free();
            if (!poly_memory.vertex2Buffer.empty() && poly_memory.vertex2Buffer.size() > meshID)
                poly_memory.vertex2Buffer[meshID].free();
            if (!poly_memory.vertex2x64Buffer.empty() && poly_memory.vertex2x64Buffer.size() > meshID)
                poly_memory.vertex2x64Buffer[meshID].free();
            if (!poly_memory.vertexBuffer.empty() && poly_memory.vertexBuffer.size() > meshID)
                poly_memory.vertexBuffer[meshID].free();
            if (!poly_memory.indexBuffer.empty() && poly_memory.indexBuffer.size() > meshID)
                poly_memory.indexBuffer[meshID].free();
            if (!poly_memory.sbtIndexBuffer.empty() && poly_memory.sbtIndexBuffer.size() > meshID)
                poly_memory.sbtIndexBuffer[meshID].free();
            if (!poly_memory.polyBuffer.empty() && poly_memory.polyBuffer.size() > meshID)
                poly_memory.polyBuffer[meshID].free();
            if (!poly_memory.cdfBuffer.empty() && poly_memory.cdfBuffer.size() > meshID)
                poly_memory.cdfBuffer[meshID].free();
            if (!poly_memory.facprobBuffer.empty() && poly_memory.facprobBuffer.size() > meshID)
                poly_memory.facprobBuffer[meshID].free();
        }
        sbt_memory.raygenRecordsBuffer.free();
        sbt_memory.missRecordsBuffer.free();
        sbt_memory.hitgroupRecordsBuffer.free();

        sim_memory.moleculeBuffer.free();
#ifdef RNG_BULKED
        crng::destroyRandHost((RN_T **) &sim_memory.randBuffer.d_ptr);
        sim_memory.randOffsetBuffer.free();
#else
        crng::destroyRandDevice((curandState_t **) &sim_memory.randBuffer.d_ptr);
#endif
        facet_memory.hitCounterBuffer.free();
        facet_memory.missCounterBuffer.free();

        if (!facet_memory.cdf1Buffer.isNullptr())
            facet_memory.cdf1Buffer.free();
        if (!facet_memory.cdf2Buffer.isNullptr())
            facet_memory.cdf2Buffer.free();
        if (!facet_memory.textureBuffer.isNullptr())
            facet_memory.textureBuffer.free();
        if (!facet_memory.texelBuffer.isNullptr())
            facet_memory.texelBuffer.free();
        if (!facet_memory.texIncBuffer.isNullptr())
            facet_memory.texIncBuffer.free();
        if (!facet_memory.profileBuffer.isNullptr())
            facet_memory.profileBuffer.free();
#ifdef DEBUGCOUNT
        memory_debug.detBuffer.free();
        memory_debug.uBuffer.free();
        memory_debug.vBuffer.free();
#endif
#ifdef DEBUGPOS
        memory_debug.posBuffer.free();
        memory_debug.posOffsetBuffer.free();
        memory_debug.posTypeBuffer.free();
#endif
#ifdef DEBUGLEAKPOS
        memory_debug.leakPosBuffer.free();
        memory_debug.leakDirBuffer.free();
        memory_debug.leakPosOffsetBuffer.free();
#endif
#ifdef DEBUGMISS
        memory_debug.missBuffer.free();
#endif
        state.asBuffer.free();
        state.launchParamsBuffer.free();
    }

} // ::flowgpu
