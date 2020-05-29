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
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "optix7.h"
#include "OptixPolygon.h"
#include "GPUDefines.h"

#ifdef DEBUGCOUNT
#define NCOUNTBINS 100
#define NBCOUNTS 100
#define DETLOW (-0.2f)
#define DETHIGH 0.2f
#define ULOW (-0.2f)
#define UHIGH 1.2f
#define VLOW (-0.2f)
#define VHIGH 1.2f
#endif

namespace flowgpu {

    enum RayType
    {
        RAY_TYPE_MOLECULE  = 0,
        RAY_TYPE_COUNT
    };

    struct TriangleMeshSBTData {
        float3 *vertex;
        int3 *index;
        flowgeom::Polygon *poly;
    };

    struct TriangleRayGenData {
        float3 *vertex;
        int3 *index;
        flowgeom::Polygon *poly;

        // -- data for launch parameters --
        // --------------------------------
        // probability for facet selection
        float2* facetProbabilities;
        //double* prob_lower;
        //double* prob_upper;
        // CFD for velocity calculation (temperature, v-bin)
        float* cdfs;
        //double *cdf_x; //index of map
        //double *cdf_val; //value of map
    };

    struct PolygonRayGenData {
        float3 *vertex;
        float2 *vertex2;
        uint32_t *index;
        flowgeom::Polygon *poly;

        // -- data for launch parameters --
        // --------------------------------
        // probability for facet selection
        float2* facetProbabilities;
        //double* prob_lower;
        //double* prob_upper;
        // CFD for velocity calculation (temperature, v-bin)
        float* cdfs;
        //double *cdf_x; //index of map
        //double *cdf_val; //value of map
    };

    struct PolygonMeshSBTData {
        float3 *vertex;
        float2 *vertex2;
        uint32_t *index;
        flowgeom::Polygon *poly;
    };

    // attributes of the molecule that have effects for tracing or post processing
    struct MolPRD
    {
        // molecule data
        float velocity;
        int currentDepth;
        //int nbBounces; //TODO: Check if info is needed
        float orientationRatio; // for low flux mode

        // post hit data
        float hitT; // distance in molecule path
        float3 hitPos;
        float3 postHitDir;
        int   hitFacetId;

        // flags - post launch processing TODO: convert all into one uint32_t ?
        int inSystem;
    };

    // Globalhitbuffer
    /*FacetHitBuffer globalHits;               // Global counts (as if the whole geometry was one extra facet)
    size_t hitCacheSize;              // Number of valid hits in cache
    size_t lastHitIndex;					//Index of last recorded hit in gHits (turns over when reaches HITCACHESIZE)
    HIT hitCache[HITCACHESIZE];       // Hit history
    LEAK leakCache[LEAKCACHESIZE];      // Leak history
    size_t  lastLeakIndex;		  //Index of last recorded leak in gHits (turns over when reaches LEAKCACHESIZE)
    size_t  leakCacheSize;        //Number of valid leaks in the cache
    size_t  nbLeakTotal;*/

    // Facethitbuffer

    //TODO: Molflow counter
    struct CuFacetHitCounter{
        CuFacetHitCounter() :
        nbDesorbed(0) , nbMCHit(0) , nbHitEquiv(0.0f) ,
        nbAbsEquiv(0.0f) , sum_1_per_ort_velocity(0.0f) ,
        sum_1_per_velocity(0.0f) , sum_v_ort(0.0f)
        {};
        // Counts
        uint32_t nbMCHit;                   // Number of hits
        uint32_t nbDesorbed;                // Number of desorbed molec
        float nbAbsEquiv;                   // Equivalent number of absorbed molecules
        float nbHitEquiv;                   //Equivalent number of hits, used for low-flux impingement rate and density calculation
        float sum_1_per_ort_velocity;       // sum of reciprocials of orthogonal velocity components, used to determine the density, regardless of facet orientation
        float sum_1_per_velocity;           //For average molecule speed calculation
        float sum_v_ort;                    // sum of orthogonal speeds of incident velocities, used to determine the pressure
    };

    struct LaunchParams
    {

        struct {
            uint32_t *missCounter;
            flowgeom::FacetTexture *facetTextures;
            float *texelInc;
            flowgeom::Texel *texels;
            flowgeom::Texel *profileSlices;
        } sharedData;

        struct {
            MolPRD *currentMoleculeData;
            uint32_t* randBufferOffset; /*! offset to remember which random number is next */
#ifdef DEBUGPOS
            uint32_t* posOffsetBuffer_debug;
            float3* positionsBuffer_debug;
#endif
#ifdef DEBUGMISS
            uint32_t* missBuffer; //first value is the amount of primitives N, followed by N primitive IDs
#endif
        } perThreadData;

        struct {
            //uint32_t *colorBuffer;
            uint2     size;
            uint32_t nbFacets;
            uint32_t nbIndices;
            uint32_t nbVertices;

            // globalSettings
            bool useMaxwell;
            float gasMass;

            uint32_t maxDepth; // for recursion
            float scene_epsilon; // to prevent self intersection (currently unused due to other techniques)
            uint32_t nbRandNumbersPerThread;
        } simConstants;

        RN_T* randomNumbers;
        CuFacetHitCounter* hitCounter;

#ifdef DEBUGCOUNT
        struct {
            uint32_t* detCount;
            uint32_t* uCount;
            uint32_t* vCount;
        } debugCounter;
#endif
        OptixTraversableHandle traversable;
    };



} // ::flowgpu