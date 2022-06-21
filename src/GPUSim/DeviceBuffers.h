//
// Created by pbahr on 05/02/2020.
//

#ifndef MOLFLOW_PROJ_DEVICEBUFFERS_H
#define MOLFLOW_PROJ_DEVICEBUFFERS_H

#include "CUDABuffer.h"

namespace flowgpu {
    struct SBTMemory{
        CUDABuffer raygenRecordsBuffer;
        CUDABuffer missRecordsBuffer;
        CUDABuffer hitgroupRecordsBuffer;
        CUDABuffer exceptionRecordsBuffer;
    };


    struct SimulationMemory_perThread{
        CUDABuffer moleculeBuffer;
        CUDABuffer randBuffer;
        CUDABuffer randOffsetBuffer;

    };

    /*!
     * @brief Result buffers for per-facet data
     */
    struct SimulationMemory_perFacet{
        CUDABuffer hitCounterBuffer;
        CUDABuffer missCounterBuffer;

        CUDABuffer cdf1Buffer;
        CUDABuffer cdf2Buffer;
        CUDABuffer textureBuffer;
        CUDABuffer texelBuffer;
        CUDABuffer texIncBuffer;

        CUDABuffer profileBuffer;
    };

    /*! one buffer per input mesh */
    struct DeviceTriangleMemory {
        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> texcoordBuffer;
        std::vector<CUDABuffer> indexBuffer;
        std::vector<CUDABuffer> sbtIndexBuffer;
        std::vector<CUDABuffer> polyBuffer;

        std::vector<CUDABuffer> facprobBuffer;
    };

    struct DevicePolygonMemory {
        std::vector<CUDABuffer> aabbBuffer;
        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> vertex2Buffer;
        std::vector<CUDABuffer> vertex2x64Buffer;
        std::vector<CUDABuffer> indexBuffer;
        std::vector<CUDABuffer> sbtIndexBuffer;
        std::vector<CUDABuffer> polyBuffer;

        std::vector<CUDABuffer> facprobBuffer;
        std::vector<CUDABuffer> cdfBuffer;
    };

#if defined(DEBUGCOUNT) || defined(DEBUGPOS) || defined(DEBUGLEAKPOS) || defined(DEBUGMISS)
struct DeviceMemoryDebug {
#ifdef DEBUGCOUNT
        CUDABuffer detBuffer;
            CUDABuffer uBuffer;
            CUDABuffer vBuffer;
#endif //DEBUGCOUNT

#ifdef DEBUGPOS
        CUDABuffer posBuffer;
        CUDABuffer posOffsetBuffer;
        CUDABuffer posTypeBuffer;
#endif //DEBUGPOS

#ifdef DEBUGLEAKPOS
        CUDABuffer leakPosBuffer;
        CUDABuffer leakDirBuffer;
        CUDABuffer leakPosOffsetBuffer;
#endif //DEBUGLEAKPOS

#ifdef DEBUGMISS
        CUDABuffer missBuffer;
#endif //DEBUGMISS
    };
#endif //DEBUG

    class DeviceBuffers {

    };
}

#endif //MOLFLOW_PROJ_DEVICEBUFFERS_H