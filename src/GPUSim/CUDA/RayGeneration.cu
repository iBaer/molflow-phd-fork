// Created by pbahr

#include <math_constants.h>

#include "LaunchParams.h"
#include "GPUDefines.h" // for NB_RAND
#include "helper_math.h"
#include <LaunchParams.h>
#include "CommonFunctions.cuh"




/*// Helper routines
thrust::device_vector<float> rand_vec();
//void initialize(thrust::device_vector<float>& v){
void initialize(){
    rand_vec().resize(1000);
    thrust::default_random_engine rng(123456);
    thrust::uniform_real_distribution<float> dist(0,1); // [0,1]
    for(size_t i = 0; i < rand_vec.size(); i++)
        rand_vec[i] = dist(rng);
}*/

namespace flowgpu {

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    //extern "C" __constant__ LaunchParams optixLaunchParams;

    extern "C" __device__ int point_in_polygon(float u, float v, const flowgpu::Polygon& poly) {
        // Fast method to check if a point is inside a polygon or not.
        // Works with convex and concave polys, orientation independent
        const PolygonRayGenData* rayGenData = (PolygonRayGenData*) optixGetSbtDataPointer();
        //const int   primID = optixGetPrimitiveIndex();

        //const Polygon& poly  = sbtData.poly[primID];

        const int nbSizeMinusOne = poly.nbVertices - 1;
        const float2* polyPoints = rayGenData->vertex2;

        int n_updown = 0;
        int n_found = 0;

        float2 p;
        p.x = u;
        p.y = v;

#ifdef BOUND_CHECK
/*
        if(poly.indexOffset < 0 || poly.indexOffset+poly.nbVertices >= optixLaunchParams.simConstants.nbVertices){
            printf("[%d] indexOffset %u -- %u >= %u is out of bounds\n", poly.parentIndex, poly.indexOffset, poly.indexOffset+poly.nbVertices, optixLaunchParams.simConstants.nbVertices);
        }*/
#endif
        for (int j = 0; j < nbSizeMinusOne; j++) {
            const float2& p1 = polyPoints[poly.indexOffset + j];
            const float2& p2 = polyPoints[poly.indexOffset + j + 1];

            //printf("[%d / %u] p1 %4.2f %4.2f / p2 %4.2f %4.2f \n",threadIdx.x,j,p1.x,p1.y,p2.x,p2.y);

            if (p.x<p1.x != p.x<p2.x) {
                float slope = (p2.y - p1.y) / (p2.x - p1.x);
                //printf("[%d / %u] slope %4.2f --> %4.2f < %4.2f \n",threadIdx.x,j,slope, (slope * p.x - p.y), (slope * p1.x - p1.y));

                if ((slope * p.x - p.y) < (slope * p1.x - p1.y)) {
                    n_updown++;
                }
                else {
                    n_updown--;
                }
                n_found++;
            }
        }

        //Last point. Repeating code because it's the fastest and this function is heavily used
        const float2& p1 = polyPoints[poly.indexOffset + nbSizeMinusOne];
        const float2& p2 = polyPoints[poly.indexOffset + 0];
        if (p.x<p1.x != p.x<p2.x) {
            //printf("[%d / %u] p1 %4.2f %4.2f / p2 %4.2f %4.2f \n",threadIdx.x,0,p1.x,p1.y,p2.x,p2.y);

            float slope = (p2.y - p1.y) / (p2.x - p1.x);
            //printf("[%d / %u] slope %4.2f --> %4.2f < %4.2f \n",threadIdx.x,0,slope, (slope * p.x - p.y), (slope * p1.x - p1.y));

            if ((slope * p.x - p.y) < (slope * p1.x - p1.y)) {
                n_updown++;
            }
            else {
                n_updown--;
            }
            n_found++;
        }

        if(((n_found / 2) & 1) ^ ((n_updown / 2) & 1)){
            return 1;
        }
        else{
            return 0;
        }
    }

    static __forceinline__ __device__
    int getSourceFacet(MolPRD& hitData,
#ifdef WITHTRIANGLES
                       const TriangleRayGenData* rayGenData,
#else
            const PolygonRayGenData* rayGenData,
#endif
#ifdef RNG_BULKED
    const RN_T* randFloat, unsigned int& randInd, unsigned int& randOffset)
#else
    curandState_t* states)
#endif
    {
#if defined(RNG_BULKED) && defined(BOUND_CHECK)
        if(randInd + randOffset >= optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y){
            printf("randInd %u is out of bounds\n", randInd + randOffset);
        }
        else if(randFloat[(unsigned int)(randInd + randOffset)] < 0.0 || randFloat[(unsigned int)(randInd + randOffset)] > 1.0){
            printf("randFloat %lf is out of bounds [%u + %u]\n", randFloat[(unsigned int)(randInd + randOffset)], randInd, randOffset);
        }
#endif
#ifdef RNG_BULKED
        float facetRnd = randFloat[(unsigned int)(randInd + randOffset++)];
#else
        float facetRnd = generate_rand(states, getWorkIndex());
#endif

        int facIndex = 0;
        bool found = false;
        do{
            OOB_CHECK_INT("facIndex", facIndex, optixLaunchParams.simConstants.nbFacets);

            found = (rayGenData->facetProbabilities[facIndex].y >= facetRnd); // find probability interval rand lies in
            if(!found)
                facIndex++;

            /*if(found){
                if(facIndex==0)
                    printf("[%d] %u/%u with last prob %8.6f > facetRnd %8.6f and rand index %u+%u\n", fbIndex, facIndex, optixLaunchParams.simConstants.nbFacets,
                           rayGenData->facetProbabilities[facIndex].y,facetRnd, randInd,randOffset-1);
                else
                    printf("found %u/%u with last prob %8.6f > facetRnd %8.6f \n", facIndex, optixLaunchParams.simConstants.nbFacets,
                       rayGenData->facetProbabilities[facIndex].y,facetRnd);
            }*/
        }while(!found && facIndex<optixLaunchParams.simConstants.nbFacets);

        OOB_CHECK_INT("Post facIndex", facIndex, optixLaunchParams.simConstants.nbFacets);

        hitData.hitFacetId = facIndex;
        return facIndex;

    }

    static __forceinline__ __device__
    float3 getNewOrigin(MolPRD& hitData,
#ifdef WITHTRIANGLES
                        const TriangleRayGenData* rayGenData,
#else
            const PolygonRayGenData* rayGenData,
#endif
                        const int& facIndex,
#ifdef RNG_BULKED
    const RN_T* randFloat, unsigned int& randInd, unsigned int& randOffset)
#else
    curandState_t* states)
#endif
{

#ifdef WITHTRIANGLES


#if defined(RNG_BULKED)
        OOB_CHECK("randInd", randInd + randOffset, optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y);
#endif

#ifdef RNG_BULKED
        hitData.rndOrigin[0] = randFloat[(unsigned int)(randInd + randOffset++)];
        hitData.rndOrigin[1] = randFloat[(unsigned int)(randInd + randOffset++)];
#else
        hitData.rndOrigin[0] = generate_rand(states, getWorkIndex());
        hitData.rndOrigin[1] = generate_rand(states, getWorkIndex());
#endif

        RN_T r1_sqrt =
                #ifdef RNG64
                    sqrt((double)hitData.rndOrigin[0]);
                #else
                    sqrtf(hitData.rndOrigin[0]);
                #endif

#if defined(RNG_BULKED)
        OOB_CHECK("randInd", randInd + randOffset, optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y);
#endif

        RN_T r2 = hitData.rndOrigin[1];

#ifdef BOUND_CHECK
        if(rayGenData->index[facIndex].x < 0 || rayGenData->index[facIndex].x >= optixLaunchParams.simConstants.nbVertices
        || rayGenData->index[facIndex].y < 0 || rayGenData->index[facIndex].y >= optixLaunchParams.simConstants.nbVertices
        || rayGenData->index[facIndex].z < 0 || rayGenData->index[facIndex].z >= optixLaunchParams.simConstants.nbVertices){
            printf("rayGenData->index[facIndex] %u,%u,%u >= %u is out of bounds\n", rayGenData->index[facIndex].x, rayGenData->index[facIndex].y, rayGenData->index[facIndex].z, optixLaunchParams.simConstants.nbVertices);
        }
#endif

        float3 vertA = rayGenData->vertex[rayGenData->index[facIndex].x];
        float3 vertB = rayGenData->vertex[rayGenData->index[facIndex].y];
        float3 vertC = rayGenData->vertex[rayGenData->index[facIndex].z];

#ifdef RNG64
        return make_float3((1.0 - r1_sqrt) * vertA + r1_sqrt * (1.0 - r2) * vertB + r1_sqrt * r2 * vertC); //rayGenData
#else
        return (1.0f-r1_sqrt) * vertA + r1_sqrt * (1.0f - r2) * vertB + r1_sqrt * r2 * vertC; //rayGenData
#endif

#else //WITHTRIANGLES

        // start position of particle (U,V) -> (x,y,z)
        float uDir = 0.0f, vDir = 0.0f;

        short isInPoly = 0;
        unsigned int loopNb = 0;

        while(!isInPoly && loopNb < NB_INPOLYCHECKS){
#ifdef BOUND_CHECK
            if(randInd + randOffset < 0 || randInd + randOffset >= optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y){
                printf("randInd %u is out of bounds\n", randInd + randOffset);
            }
#endif
            uDir = randFloat[(unsigned int)(randInd + randOffset++)]; // float3(randFloat[randInd++],randFloat[randInd++],randFloat[randInd++])

#ifdef BOUND_CHECK
            if(randInd + randOffset < 0 || randInd + randOffset >= optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y){
                printf("randInd %u is out of bounds\n", randInd + randOffset);
            }
#endif
            vDir = randFloat[(unsigned int)(randInd + randOffset++)];

            isInPoly = point_in_polygon(uDir,vDir, rayGenData->poly[facIndex]);

            ++loopNb;
        }

        // if no suitable starting point was found
        if(!isInPoly){
            uDir = 0.5;
            vDir = 0.5;
        }

        hitData.rndOrigin[0] = uDir;
        hitData.rndOrigin[1] = vDir;

        //rayOrigin =
        return rayGenData->poly[facIndex].O + uDir * rayGenData->poly[facIndex].U + vDir * rayGenData->poly[facIndex].V;
#endif //WITHTRIANGLES
    }

    static __forceinline__ __device__
    void initMoleculeInSystem(const unsigned int bufferIndex, MolPRD& hitData, float3& rayDir , float3& rayOrigin)
    {
        hitData = optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex];
        rayDir = hitData.postHitDir;
        rayOrigin = hitData.hitPos;
#ifdef WITHTRIANGLES
        const TriangleRayGenData* rayGenData = (TriangleRayGenData*) optixGetSbtDataPointer();
#else
        const PolygonRayGenData *rayGenData = (PolygonRayGenData *) optixGetSbtDataPointer();
#endif
        if(rayGenData->poly[hitData.hitFacetId].facProps.endangered_neighbor) { // with offset to center
            //rayOrigin = offset_to_center(hitData.hitPos, hitData.hitFacetId, rayGenData->poly[hitData.hitFacetId]);
            if(hitData.barycentrics.x < 1e-4 || hitData.barycentrics.y < 1e-4 || 1.0 - hitData.barycentrics.x - hitData.barycentrics.y < 1e-4) {
                    //float3 offsetted = offset_to_center(hitData.hitPos, hitData.hitFacetId, rayGenData->poly[hitData.hitFacetId]);
               if(1){ //if(rayGenData->poly[hitData.hitFacetId].parentIndex == 4 || rayGenData->poly[hitData.hitFacetId].parentIndex == 9) {
                    /*printf("Offset diff : %e , %e , %e x %e , %e , %e = D( %e , %e , %e ) \n",
                           hitData.hitPos.x, hitData.hitPos.y, hitData.hitPos.z, offsetted.x , offsetted.y , offsetted.z,
                           hitData.hitPos.x - offsetted.x, hitData.hitPos.y - offsetted.y, hitData.hitPos.z - offsetted.z);*/
                    float u = hitData.barycentrics.x;
                    float v = hitData.barycentrics.y;
                    float w = 1.0f - hitData.barycentrics.x - hitData.barycentrics.y;
                    float3 vertA = rayGenData->vertex[rayGenData->index[hitData.hitFacetId].x];
                    float3 vertB = rayGenData->vertex[rayGenData->index[hitData.hitFacetId].y];
                    float3 vertC = rayGenData->vertex[rayGenData->index[hitData.hitFacetId].z];
                    /*float wu = u * vertA.x + v * vertB.x + w * vertC.x;
                    float wv = u * vertA.y + v * vertB.y + w * vertC.y;
                    float ww = u * vertA.z + v * vertB.z + w * vertC.z;*/

                   hitData.barycentrics_old = hitData.barycentrics;
                   float3 bary_off = offset_to_center_bary(hitData.barycentrics, vertA, vertB, vertC, hitData.hitPos, hitData.hitFacetId, rayGenData->poly[hitData.hitFacetId]);
                    float wu = hitData.barycentrics.x * vertA.x + hitData.barycentrics.y * vertB.x + (1.0f-hitData.barycentrics.x-hitData.barycentrics.y) * vertC.x;
                    float wv = hitData.barycentrics.x * vertA.y + hitData.barycentrics.y * vertB.y + (1.0f-hitData.barycentrics.x-hitData.barycentrics.y) * vertC.y;
                    float ww = hitData.barycentrics.x * vertA.z + hitData.barycentrics.y * vertB.z + (1.0f-hitData.barycentrics.x-hitData.barycentrics.y) * vertC.z;
                    float wu_off = bary_off.x * vertA.x + bary_off.y * vertB.x + bary_off.z * vertC.x;
                    float wv_off = bary_off.x * vertA.y + bary_off.y * vertB.y + bary_off.z * vertC.y;
                    float ww_off = bary_off.x * vertA.z + bary_off.y * vertB.z + bary_off.z * vertC.z;
                    //printf("[%d (%d)] Offset diff : %e , %e , %e : ( %e , %e , %e ) -> ( %e , %e , %e ) --- [D( %e , %e , %e )]\n",


                    optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = make_float3(wu_off, wv_off, ww_off);
                    hitData.hitPos = make_float3(wu_off, wv_off, ww_off);
                    hitData.barycentrics = make_float2(bary_off.x, bary_off.y);
                   optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].barycentrics = hitData.barycentrics;
                   optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].barycentrics_old = hitData.barycentrics_old;

                   rayOrigin = hitData.hitPos;
                   optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = hitData.hitPos;
                   /*if(hitData.barycentrics.x < 1e-5 || hitData.barycentrics.y < 1e-5 || 1.0 - hitData.barycentrics.x - hitData.barycentrics.y < 1e-5) {
                       printf("[%d (%d)] Offset diff : %e , %e , %e --- %e , %e , %e [D( %e , %e , %e )]\n",
                              hitData.hitFacetId, rayGenData->poly[hitData.hitFacetId].parentIndex,
                              hitData.barycentrics.x, hitData.barycentrics.y,
                              1.0f - hitData.barycentrics.x - hitData.barycentrics.y,
                              bary_off.x, bary_off.y, bary_off.z,
                               *//*wu, wv, ww,
                                wu_off, wv_off, ww_off,*//*
                              hitData.barycentrics.x - bary_off.x, hitData.barycentrics.y - bary_off.y,
                              (1.0f - hitData.barycentrics.x - hitData.barycentrics.y) - bary_off.z
                       );
                       printf("[%d (%d)] Offset real : %e , %e , %e \n",
                              hitData.hitFacetId, rayGenData->poly[hitData.hitFacetId].parentIndex, hitData.hitPos.x, hitData.hitPos.y, hitData.hitPos.z
                       );
                   }*/

               }
                /*optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = offset_to_center(
                        hitData.hitPos, hitData.hitFacetId, rayGenData->poly[hitData.hitFacetId]);*/
            }
        }
#ifdef DEBUG
        //printf("--- in[%d] %4.2f , %4.2f , %4.2f - %4.2f , %4.2f , %4.2f ---\n", blockDim.x * blockIdx.x + threadIdx.x,optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].hitPos.x, optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].hitPos.y, optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].hitPos.z,optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].postHitDir.x, optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].postHitDir.y, optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].postHitDir.z);
#endif
    }


    // Calculate new direction, but keep old position for molecule that was stuck due to self-intersection
    static __forceinline__ __device__
    void initMoleculePostSelfIntersection(const unsigned int bufferIndex, MolPRD& hitData, float3& rayDir , float3& rayOrigin)
    {
        hitData = optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex];

/*#if defined(RNG_BULKED)
        RN_T* randFloat = optixLaunchParams.randomNumbers;
        unsigned int randInd = optixLaunchParams.simConstants.nbRandNumbersPerThread*(bufferIndex);
        unsigned int randOffset = optixLaunchParams.perThreadData.randBufferOffset[bufferIndex];
#else

#endif*/

        OOB_CHECK_INT("hitData.hitFacetId", hitData.hitFacetId, optixLaunchParams.simConstants.nbFacets);
#ifdef WITHTRIANGLES
        const TriangleRayGenData* rayGenData = (TriangleRayGenData*) optixGetSbtDataPointer();
#else
        const PolygonRayGenData* rayGenData = (PolygonRayGenData*) optixGetSbtDataPointer();
#endif

        //rayDir = hitData.postHitDir;
        /*if(rayGenData->poly[hitData.hitFacetId].facProps.is2sided && rayGenData->poly[hitData.hitFacetId].facProps.opacity == 0.0f){
            // todo: just offset later, no direction change
            if(bufferIndex == 0) printf(" Self intersection back facet hit %d\n",rayGenData->poly[hitData.hitFacetId].parentIndex);
            rayDir = getNewReverseDirection(hitData, rayGenData->poly[hitData.hitFacetId], randFloat, randInd,
                                            randOffset);
        }
        else *//*if(
                (hitData.facetHitSide == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE)
                )*//*{
            rayDir = getNewDirection(hitData, rayGenData->poly[hitData.hitFacetId], randFloat, randInd, randOffset);
            if(hitData.facetHitSide != OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE)
                if(bufferIndex == 0)
                    printf(" Self intersection back facet hit %d\n",rayGenData->poly[hitData.hitFacetId].parentIndex);

        }*/
        /*else {
            if(bufferIndex == 0) printf(" Self intersection back facet hit %d\n",rayGenData->poly[hitData.hitFacetId].parentIndex);
            rayDir = getNewReverseDirection(hitData, rayGenData->poly[hitData.hitFacetId], randFloat, randInd,
                                            randOffset);
        }*/
        //rayDir = getNewDirection(hitData,rayGenData->poly[hitData.hitFacetId],randFloat,randInd,randOffset);


        /*if(hitData.facetHitSide == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE) {
            hitData.hitPos = offset_ray(hitData.hitPos, -1.0f*rayGenData->poly[hitData.hitFacetId].N);
        }
        else{
            hitData.hitPos = offset_ray_old(hitData.hitPos,1.0f* rayGenData->poly[hitData.hitFacetId].N);
        }*/

        rayOrigin = hitData.hitPos;
        //optixLaunchParams.perThreadData.randBufferOffset[bufferIndex] = randOffset; // set value back to buffer
    }

// Calculate new position for molecule with an offset
static __forceinline__ __device__
void initMoleculeTransparentHit(const unsigned int bufferIndex, MolPRD& hitData, float3& rayDir , float3& rayOrigin)
{
    hitData = optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex];

    //rayDir = hitData.postHitDir;
    rayDir = hitData.postHitDir;
    rayOrigin = hitData.hitPos;

}

    // --------------------------------------
    // increase facet counters for desorption
    // --------------------------------------
    static __forceinline__ __device__
    void increaseHitCounterDesorption(CuFacetHitCounter& hitCounter, const MolPRD &hitData,
                                      const float3 &rayDir, const float3 &polyNormal)
    {

        //const float hitEquiv = 0.0f; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
        //const float absEquiv = 0.0f; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
        //atomicAdd(&optixLaunchParams.hitCounter[counterIdx].nbMCHit,1);
        //++optixLaunchParams.hitCounter[counterIdx].nbMCHit;
        //const unsigned int counterIdx = facIndex + ((bufferIndex)%(EXTRAFACETCOUNTERS))*optixLaunchParams.simConstants.nbFacets;

        //atomicAdd(&hitCounter.nbMCHit, static_cast<uint32_t>(0));
        //atomicAdd(&hitCounter.nbHitEquiv, 0.0f);

#ifdef HIT64
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781;
        const FLOAT_T velocity = hitData.velocity;
        const FLOAT_T ortVelocity = velocity*fabsf(dot(rayDir, polyNormal));

        atomicAdd(&hitCounter.nbDesorbed, static_cast<uint64_cu>(1));
        atomicAdd(&hitCounter.sum_1_per_ort_velocity, (2.0 / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, 1.0 / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;
#else
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f;
        const FLOAT_T velocity = hitData.velocity;
        const FLOAT_T ortVelocity = velocity*fabsf(dot(rayDir, polyNormal));

        atomicAdd(&hitCounter.nbDesorbed, static_cast<uint32_t>(1));
        atomicAdd(&hitCounter.sum_1_per_ort_velocity, (2.0f / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, 1.0f / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;
#endif
    }

    // --------------------------------------
    // increase texture counters for desorption
    // --------------------------------------
    static __forceinline__ __device__
    void RecordDesorptionTexture(const flowgpu::Polygon& poly, const MolPRD& hitData, const float3& rayOrigin, const float3& rayDir){

        const float2 hitLocation = getHitLocation_old(poly,rayOrigin);

        flowgpu::FacetTexture& facetTex = optixLaunchParams.sharedData.facetTextures[poly.texProps.textureOffset];

        const auto tu = (unsigned int)(__saturatef(hitLocation.x) * facetTex.texWidth_precise); // saturate for rounding errors that can result for values larger than 1.0
        const auto tv = (unsigned int)(__saturatef(hitLocation.y) * facetTex.texHeight_precise);
        unsigned int add = tu + tv * (facetTex.texWidth);
        //printf("Hit at %lf , %lf for tex %lf , %lf\n", hitLocation.x, hitLocation.y, facetTex.texWidth_precise, facetTex.texHeight_precise);

#ifdef BOUND_CHECK
        if(facetTex.texelOffset + add >= optixLaunchParams.simConstants.nbTexel){printf("facetTex.texelOffset + add %u >= %u is out of bounds\n", facetTex.texelOffset + add, optixLaunchParams.simConstants.nbTexel);}
#endif

        flowgpu::Texel& tex = optixLaunchParams.sharedData.texels[facetTex.texelOffset + add];

#ifdef HIT64
        const FLOAT_T velocity_factor = 2.0;
        const FLOAT_T ortSpeedFactor = 1.0;
        const FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781) * hitData.velocity*fabs(dot(rayDir, poly.N)); //surface-orthogonal velocity component
        atomicAdd(&tex.countEquiv, static_cast<uint64_cu>(1));
        atomicAdd(&tex.sum_1_per_ort_velocity, 1.0 * velocity_factor / ortVelocity);
        atomicAdd(&tex.sum_v_ort_per_area, 1.0 * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add]); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const FLOAT_T velocity_factor = 2.0f;
        const FLOAT_T ortSpeedFactor = 1.0f;
        FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) * hitData.velocity*fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component
        atomicAdd(&tex.countEquiv, static_cast<uint32_t>(1));
        atomicAdd(&tex.sum_1_per_ort_velocity, 1.0f * velocity_factor / ortVelocity);
        atomicAdd(&tex.sum_v_ort_per_area, 1.0f * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add]); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif


        //printf("Desorbing on facet#%u %u + %u = %u\n", poly.parentIndex, tu, tv, add);

    }

    // --------------------------------------
    // increase texture counters for desorption
    // --------------------------------------
    static __forceinline__ __device__
    void RecordDesorptionProfile(const flowgpu::Polygon& poly, const MolPRD& hitData, const float3& rayOrigin, const float3& rayDir){

        const float2 hitLocation = getHitLocation_old(poly,rayOrigin);

        unsigned int add = 0;
        if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileU){
            add = (unsigned int)(__saturatef(hitLocation.x) * PROFILE_SIZE);
        }
        else if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileV){
            add = (unsigned int)(__saturatef(hitLocation.y) * PROFILE_SIZE);
        }


#ifdef BOUND_CHECK
        if(poly.profProps.profileOffset + add >= optixLaunchParams.simConstants.nbProfSlices){
            printf("[DES] poly.profProps.profileOffset + add %u >= %u is out of bounds\n", poly.profProps.profileOffset + add, optixLaunchParams.simConstants.nbProfSlices);}
#endif
        flowgpu::Texel& tex = optixLaunchParams.sharedData.profileSlices[poly.profProps.profileOffset + add];

#ifdef HIT64
        const FLOAT_T velocity_factor = 2.0;
        const FLOAT_T ortSpeedFactor = 1.0;
        const FLOAT_T ortVelocity = hitData.velocity*fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint64_t>(1));
        atomicAdd(&tex.sum_1_per_ort_velocity, velocity_factor / ortVelocity);
        atomicAdd(&tex.sum_v_ort_per_area, ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781)); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const FLOAT_T velocity_factor = 2.0f;
        const FLOAT_T ortSpeedFactor = 1.0f;
        const FLOAT_T ortVelocity = hitData.velocity*fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint32_t>(1));
        atomicAdd(&tex.sum_1_per_ort_velocity, velocity_factor / ortVelocity);
        atomicAdd(&tex.sum_v_ort_per_area, ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f)); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif
    }

static __forceinline__ __device__
void recordDesorption(const unsigned int& counterIdx, const flowgpu::Polygon& poly, const MolPRD& hitData, const float3& rayDir, const float3& rayOrigin)
{
#ifdef BOUND_CHECK
    if(counterIdx >= optixLaunchParams.simConstants.nbFacets * EXTRAFACETCOUNTERS){
            printf("facIndex %u >= %u is out of bounds\n", counterIdx, optixLaunchParams.simConstants.nbFacets * EXTRAFACETCOUNTERS);
        }
#endif
    increaseHitCounterDesorption(optixLaunchParams.hitCounter[counterIdx],hitData,rayDir, poly.N);
#ifdef WITH_TEX
    if (poly.texProps.textureFlags & flowgpu::TEXTURE_FLAGS::countDes)
            RecordDesorptionTexture(poly, hitData, rayOrigin, rayDir);
#endif
#ifdef WITH_PROF
    if (poly.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile)
            RecordDesorptionProfile(poly, hitData, rayOrigin, rayDir);
#endif
}

    // Calculate new direction, but keep old position
    static __forceinline__ __device__
    void initMoleculeFromStart(const unsigned int bufferIndex, MolPRD& hitData, float3& rayDir , float3& rayOrigin) {
        hitData.hitPos = make_float3(-999.9f, -999.9f, -999.9f);
        hitData.postHitDir = make_float3(-999.9f, -999.9f, -999.9f);
        hitData.hitT = -999.0f;
        hitData.velocity = -999.0f;
        hitData.currentDepth = 0;
#ifdef GPUNBOUNCE
        hitData.nbBounces = 0;
#endif
        hitData.inSystem = NEW_PARTICLE;
        hitData.orientationRatio = 1.0f;

        hitData.rndOrigin[0] = -999999.9;
        hitData.rndOrigin[1] = -999999.9;
        hitData.rndDirection[0] = -999999.9;
        hitData.rndDirection[1] = -999999.9;

#ifdef DEBUG_BARY
        hitData.barycentrics = {-999999.9, -999999.9};
        hitData.barycentrics_old = {-999999.9, -999999.9};
#endif
#ifdef WITHTRIANGLES
        const TriangleRayGenData* rayGenData = (TriangleRayGenData*) optixGetSbtDataPointer();
#else
        const PolygonRayGenData *rayGenData = (PolygonRayGenData *) optixGetSbtDataPointer();
#endif
#if defined(RNG_BULKED)
        const RN_T *randFloat = optixLaunchParams.randomNumbers;
        unsigned int randInd = optixLaunchParams.simConstants.nbRandNumbersPerThread * (bufferIndex);
        unsigned int randOffset = optixLaunchParams.perThreadData.randBufferOffset[bufferIndex];

        const int facIndex = getSourceFacet(hitData, rayGenData, randFloat, randInd, randOffset);
#else
        curandState_t* states = (curandState_t*) optixLaunchParams.randomNumbers;
        const int facIndex = getSourceFacet(hitData, rayGenData, states);
#endif
        hitData.hitFacetId = facIndex;

#ifdef BOUND_CHECK
        if(facIndex < 0 || facIndex >= optixLaunchParams.simConstants.nbFacets){
            printf("facIndex %u >= %u is out of bounds\n", facIndex, optixLaunchParams.simConstants.nbFacets);
        }
#endif

#if defined(RNG_BULKED)
        rayOrigin = getNewOrigin(hitData, rayGenData, facIndex, randFloat, randInd, randOffset);
        rayDir = getNewDirection(hitData, rayGenData->poly[facIndex], randFloat, randInd, randOffset);
        // Write back cached local variables to shared memory
        optixLaunchParams.perThreadData.randBufferOffset[bufferIndex] = randOffset;
#else
        rayOrigin = getNewOrigin(hitData, rayGenData, facIndex, states);
        rayDir = getNewDirection(hitData, rayGenData->poly[facIndex], states);
#endif

        if(optixLaunchParams.simConstants.useMaxwell){
            RN_T rnd_val;
#if defined(RNG_BULKED)
            OOB_CHECK("randInd", randInd + randOffset, optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y);
#endif

#ifdef RNG_BULKED
            rnd_val = randFloat[(unsigned int)(randInd + randOffset++)];
            optixLaunchParams.perThreadData.randBufferOffset[bufferIndex] = randOffset;
#else
            rnd_val = generate_rand(states, getWorkIndex());
#endif
            hitData.velocity = randomVelo(rayGenData->poly[facIndex], rnd_val);

        }
        else {
            hitData.velocity = getNewVelocity(rayGenData->poly[facIndex], optixLaunchParams.simConstants.gasMass);
        }
        // --------------------------------------
        // increase facet counters for desorption
        // --------------------------------------
        const unsigned int counterIdx =
                facIndex + ((bufferIndex) % (EXTRAFACETCOUNTERS)) * optixLaunchParams.simConstants.nbFacets;
        recordDesorption(counterIdx, rayGenData->poly[facIndex], hitData, rayDir, rayOrigin);
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__startFromSource()
    {
        const unsigned int bufferIndex = getWorkIndex();

        OOB_CHECK("bufferIndex", bufferIndex, (optixLaunchParams.simConstants.nbRandNumbersPerThread * optixLaunchParams.simConstants.size.x * optixLaunchParams.simConstants.size.y));
        float3 rayOrigin;
        float3 rayDir;

        MolPRD& hitData = optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex];

        //printf("[%d] Particle status: %d v %d\n", bufferIndex, hitData.hasToTerminate, hitData.inSystem);
#ifdef WITHDESORPEXIT
        if(hitData.hasToTerminate==2){
            return;
        }
#endif
        switch (hitData.inSystem) {
            case NEW_PARTICLE: {
                /*
                 * start from a source
                 */
                initMoleculeFromStart(bufferIndex, hitData, rayDir, rayOrigin);
                break;
            }
            case TRANSPARENT_HIT: {
                initMoleculeTransparentHit(bufferIndex, hitData, rayDir, rayOrigin);
                break;
            }
            case ACTIVE_PARTICLE: {
                /* if molecule is still in the system (bounce etc.)
                 * load data from thread buffer
                 */
                initMoleculeInSystem(bufferIndex, hitData, rayDir, rayOrigin);
                break;
            }
            case SELF_INTERSECTION: {
                /*
                 * if molecule suffered from self intersection with starting facet, try again with new direction
                 */
                initMoleculePostSelfIntersection(bufferIndex, hitData, rayDir, rayOrigin);
                break;
            }
            default: {
                printf("Unknown launch status!: %u\n",hitData.inSystem);
                break;
            }
        }
#if defined(DEBUGPOS)
    if(bufferIndex==0){
        const unsigned int posIndexOffset = optixLaunchParams.perThreadData.posOffsetBuffer_debug[bufferIndex]++;
        if(posIndexOffset<NBPOSCOUNTS){
            const unsigned int posIndex = bufferIndex*NBPOSCOUNTS+posIndexOffset;
            //printf("[%d] my pos is %d\n", bufferIndex, posIndex);
            optixLaunchParams.perThreadData.positionsBuffer_debug[posIndex] = rayOrigin;
            optixLaunchParams.perThreadData.positionsType_debug[posIndex] = 1;
        }
    }
#endif

        {
#ifdef WITHTRIANGLES
    const flowgpu::TriangleRayGenData* rayGenData = (flowgpu::TriangleRayGenData*) optixGetSbtDataPointer();
#else
    const flowgpu::PolygonRayGenData* rayGenData = (flowgpu::PolygonRayGenData*) optixGetSbtDataPointer();
#endif
#ifdef BOUND_CHECK
            if(hitData.hitFacetId >= optixLaunchParams.simConstants.nbFacets){
                printf("[ActiveStart] facIndex %u >= %u is out of bounds (%u)\n", hitData.hitFacetId, optixLaunchParams.simConstants.nbFacets, hitData.inSystem);
            }
#endif
            apply_offset(rayGenData->poly[hitData.hitFacetId], hitData, rayOrigin);
            /*if(rayGenData->poly[hitData.hitFacetId].facProps.endangered_neighbor) { // with offset to center
                rayOrigin = offset_to_center(hitData.hitPos, hitData.hitFacetId, rayGenData->poly[hitData.hitFacetId]);
            }*/
        }
#ifdef PAYLOAD_DIRECT
        int hi_vel = __double2hiint(hitData.velocity);
        int lo_vel = __double2loint(hitData.velocity);
#else
        uint2 payload = splitPointer(&hitData);
#endif
#ifdef DEBUG
        if(isnan(rayDir.x)) {
            unsigned int randInd = optixLaunchParams.simConstants.nbRandNumbersPerThread*(bufferIndex);
            unsigned int randOffset = optixLaunchParams.perThreadData.randBufferOffset[bufferIndex];

            DEBUG_PRINT("RG[%d] ray dir NaN -> %d / %d + %d\n",bufferIndex,hitData.inSystem, randInd, randOffset);
        }
#endif


        optixTrace(optixLaunchParams.traversable,
               rayOrigin,
               rayDir,
               0.f,//optixLaunchParams.simConstants.scene_epsilon,//1.e-6f,//0.f,    // tmin // TODO: Choose scene_epsilon wisely
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
                   RAY_FLAGS,
                   RayType::RAY_TYPE_MOLECULE,             // SBT offset
                   RayType::RAY_TYPE_COUNT,               // SBT stride
                   RayType::RAY_TYPE_MOLECULE,             // missSBTIndex
               //u0, u1 , u2, u3);
               //float3_as_args(hitData.hitPos),

#ifdef PAYLOAD_DIRECT
               // direct
               reinterpret_cast<unsigned int&>(hitData.currentDepth),
               reinterpret_cast<unsigned int&>(hitData.inSystem),
               reinterpret_cast<unsigned int&>(hitData.hitFacetId),
               reinterpret_cast<unsigned int&>(hitData.hitT),
               reinterpret_cast<unsigned int&>( hi_vel ),
               reinterpret_cast<unsigned int&>( lo_vel ),
               reinterpret_cast<unsigned int&>(hitData.nbBounces),
               reinterpret_cast<unsigned int&>(hitData.facetHitSide)
#else
                // ptr
                   payload.x, payload.y
#endif
        );

        //hitData.hitPos = hitData.hitOri + hitData.hitT * hitData.hitDir;
#ifdef PAYLOAD_DIRECT
        hitData.velocity = __hiloint2double(hi_vel,lo_vel );
#else
        // update particle state in memory after ray generation
        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex] = hitData;
#endif
#if defined(DEBUG) && defined(GPUNBOUNCE)
        if(bufferIndex == optixLaunchParams.simConstants.size.x - 1 && (hitData.nbBounces % (int)1e4 == 1e4 - 1))
            printf("[%d] has new launch status -> %d and bounces %d / %d\n",bufferIndex,hitData.inSystem,optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].nbBounces,hitData.nbBounces);
#endif
/*
        // and write to thread buffer ...
        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].velocity = hitData.velocity;
        hitData.currentDepth = hitData.currentDepth;
#ifdef GPUNBOUNCE
#ifdef DEBUG
        if(bufferIndex == optixLaunchParams.simConstants.size.x - 1 && (hitData.nbBounces % (int)1e4 == 1e4 - 1))
            printf("[%d] has new launch status -> %d and bounces %d / %d\n",bufferIndex,hitData.inSystem,optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].nbBounces,hitData.nbBounces);
#endif
        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].nbBounces = hitData.nbBounces;
#endif
        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].inSystem = hitData.inSystem;
        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitFacetId = hitData.hitFacetId;
        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitT = hitData.hitT;

        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].facetHitSide = hitData.facetHitSide;
*/

    }
} // ::flowgpu
