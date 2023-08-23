//
// Created by pbahr on 28/09/2020.
//

#ifndef MOLFLOW_PROJ_PERRAYDATA_H
#define MOLFLOW_PROJ_PERRAYDATA_H

#define DEBUG_BARY 1

#include <optix.h>

namespace flowgpu {

// attributes of the molecule that have effects for tracing or post processing
    struct MolPRD {
        // molecule data
        double velocity;
        int currentDepth;
#ifdef GPUNBOUNCE
        unsigned int nbBounces; //TODO: Check if info is needed
#endif
        float orientationRatio; // for low flux mode
        // post hit data
        float hitT; // distance in molecule path
        float3 hitPos;
        float3 postHitDir;
        unsigned int hitFacetId;

        unsigned int inSystem;
        unsigned int facetHitSide;

        double rndOrigin[2]{};
        double rndDirection[2]{};
        // flags - post launch processing TODO: convert all into one uint32_t ?
#ifdef WITHDESORPEXIT
        unsigned int hasToTerminate;
#endif
#ifdef DEBUG_BARY
        float2 barycentrics;
        float2 barycentrics_old;
#endif
    };

    // Alias the PerRayData pointer and an uint2 for the payload split and merge operations. Generates just move instructions.
    typedef union {
        MolPRD *ptr;
        uint2 dat;
    } Payload;

    __forceinline__ __device__
    uint2 splitPointer(MolPRD *ptr) {
        Payload payload;

        payload.ptr = ptr;

        return payload.dat;
    }

    __forceinline__ __device__
    MolPRD *mergePointer(unsigned int p0, unsigned int p1) {
        Payload payload;

        payload.dat.x = p0;
        payload.dat.y = p1;

        return payload.ptr;
    }

}
#endif //MOLFLOW_PROJ_PERRAYDATA_H
