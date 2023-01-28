// Created by pbahr

#include "helper_math.h"

#include "LaunchParams.h"
#include "GPUDefines.h" // for NB_RAND
#include "CommonFunctions.cuh"

using namespace flowgpu;

namespace flowgpu {

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    //extern "C" __constant__ LaunchParams optixLaunchParams;

#if defined(PAYLOAD_DIRECT)
    static __device__ __inline__ MolPRD getMolPRD()
    {
        MolPRD prd;
        prd.currentDepth = optixGetPayload_0();
        prd.inSystem = optixGetPayload_1();
        prd.hitFacetId = optixGetPayload_2();
        prd.hitT = int_as_float( optixGetPayload_3() );
        prd.velocity = __hiloint2double( optixGetPayload_4(),optixGetPayload_5() );

#ifdef GPUNBOUNCE
        prd.nbBounces = optixGetPayload_6();
#endif
        prd.facetHitSide = optixGetPayload_7();
        return prd;
    }

    static __device__ __inline__ void setMolPRD( const MolPRD &prd )
    {
        optixSetPayload_0( prd.currentDepth );
        optixSetPayload_1( prd.inSystem );
        optixSetPayload_2( prd.hitFacetId );
        optixSetPayload_3( float_as_int(prd.hitT) );
        optixSetPayload_4( __double2hiint(prd.velocity) );
        optixSetPayload_5( __double2loint(prd.velocity) );

#ifdef GPUNBOUNCE
        optixSetPayload_6( prd.nbBounces );
#endif
        optixSetPayload_7( prd.facetHitSide );

    }
#endif

    // --------------------------------------
    // increase facet counters for absorption
    // --------------------------------------
    static __forceinline__ __device__
    void increaseHitCounterAbsorp(CuFacetHitCounter& hitCounter, const FLOAT_T hitEquiv, const FLOAT_T absEquiv,  const FLOAT_T ortVelocity, const FLOAT_T velFactor, const FLOAT_T velocity)
    {
        //printf("-- %d -> Absorp ----- \n",prd.currentDepth);
        // 	IncreaseFacetCounter(iFacet, sHandle->currentParticle.flightTime,
        // 	1, 0, 1, 2.0 / ortVelocity, (sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity);

#ifdef HIT64
        atomicAdd(&hitCounter.nbMCHit,static_cast<uint64_t>(1));
        atomicAdd(&hitCounter.nbHitEquiv, hitEquiv);
        atomicAdd(&hitCounter.nbAbsEquiv, absEquiv);
        atomicAdd(&hitCounter.sum_1_per_ort_velocity, (1.0 / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, (hitEquiv) / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;

#else
        atomicAdd(&hitCounter.nbMCHit,static_cast<uint32_t>(1));
        atomicAdd(&hitCounter.nbHitEquiv, hitEquiv);
        atomicAdd(&hitCounter.nbAbsEquiv, absEquiv);
        atomicAdd(&hitCounter.sum_1_per_ort_velocity, (1.0f / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, (hitEquiv) / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;
#endif
    }

    // --------------------------------------
    // increase facet counters for bounce
    // --------------------------------------
    static __forceinline__ __device__
    void increaseHitCounterBounce(CuFacetHitCounter &hitCounter, const FLOAT_T hitEquiv, const FLOAT_T ortVelocity, const FLOAT_T velFactor, const FLOAT_T velocity)
    {
        // 	IncreaseFacetCounter(iFacet, sHandle->currentParticle.flightTime, 1, 0, 0, 1.0 / ortVelocity, (sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity);

#ifdef HIT64
        const FLOAT_T oriRatio = 1.0;

        atomicAdd(&hitCounter.nbMCHit, static_cast<uint64_t>(1));
        atomicAdd(&hitCounter.nbHitEquiv, hitEquiv);

        atomicAdd(&hitCounter.sum_1_per_ort_velocity, oriRatio * (1.0 / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, oriRatio *velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, (hitEquiv) / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;
#else
        const float oriRatio = 1.0f;

        atomicAdd(&hitCounter.nbMCHit, static_cast<uint32_t>(1));
        atomicAdd(&hitCounter.nbHitEquiv, hitEquiv);

        atomicAdd(&hitCounter.sum_1_per_ort_velocity, oriRatio * (1.0f / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, oriRatio *velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, (hitEquiv) / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;
#endif
    }

    // Same but without increased hit counter
    static __forceinline__ __device__
    void increaseHitCounterPostBounce(CuFacetHitCounter &hitCounter, const FLOAT_T hitEquiv, const FLOAT_T ortVelocity, const FLOAT_T velFactor, const FLOAT_T velocity)
    {
#ifdef HIT64
        const FLOAT_T oriRatio = 1.0;
        // 		IncreaseFacetCounter(iFacet, sHandle->currentParticle.flightTime, 0, 0, 0, 1.0 / ortVelocity, (sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity);
        atomicAdd(&hitCounter.sum_1_per_ort_velocity, oriRatio * (1.0 / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, oriRatio * velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
#else
        const float oriRatio = 1.0f;
        // 		IncreaseFacetCounter(iFacet, sHandle->currentParticle.flightTime, 0, 0, 0, 1.0 / ortVelocity, (sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity);
        atomicAdd(&hitCounter.sum_1_per_ort_velocity, oriRatio * (1.0f / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, oriRatio * velFactor*ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
#endif
    }

    // --------------------------------------
    // increase facet counters for bounce
    // --------------------------------------
    static __forceinline__ __device__
    void increaseTransparentCounter(CuFacetHitCounter &hitCounter, const FLOAT_T hitEquiv, const FLOAT_T ortVelocity, const FLOAT_T velFactor, const FLOAT_T velocity)
    {
        /*IncreaseFacetCounter(facet, currentParticle.flightTime +
                                    facet->colDist / 100.0 / currentParticle.velocity,
                                    1, 0, 0,
                             2.0 / (currentParticle.velocity * directionFactor),
                             2.0 * (wp.useMaxwellDistribution ? 1.0 : 1.1781) * currentParticle.velocity *
                             directionFactor);*/

#ifdef HIT64
        const FLOAT_T oriRatio = 1.0;

        atomicAdd(&hitCounter.nbMCHit, static_cast<uint64_t>(1));
        atomicAdd(&hitCounter.nbHitEquiv, hitEquiv);

        atomicAdd(&hitCounter.sum_1_per_ort_velocity, oriRatio * (2.0 / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, oriRatio * 2.0 * velFactor * ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, (hitEquiv) / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;
#else
        const float oriRatio = 1.0f;

        atomicAdd(&hitCounter.nbMCHit, static_cast<uint32_t>(1));
        atomicAdd(&hitCounter.nbHitEquiv, hitEquiv);

        atomicAdd(&hitCounter.sum_1_per_ort_velocity, oriRatio * (2.0f / ortVelocity));//prd.oriRatio * sum_1_per_v;
        atomicAdd(&hitCounter.sum_v_ort, oriRatio * 2.0f * velFactor * ortVelocity);//(sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity; //prd.oriRatio * sum_v_ort;
        atomicAdd(&hitCounter.sum_1_per_velocity, (hitEquiv) / velocity);//(hitEquiv + static_cast<double>(desorb)) / prd.velocity;
#endif
    }

    // --------------------------------------
    // increase texture counters for absorption (same as desorp)
    // --------------------------------------
    static __forceinline__ __device__
    void RecordAbsorpTexture(const flowgpu::Polygon &poly, MolPRD &hitData, const float3& rayOrigin, const float3& rayDir){

        const flowgpu::TriangleMeshSBTData &sbtData = *(const flowgpu::TriangleMeshSBTData*)optixGetSbtDataPointer();
        const float2 hitLocation = getHitLocation(optixGetTriangleBarycentrics(), sbtData.texcoord, optixGetPrimitiveIndex() * 3);

        flowgpu::FacetTexture& facetTex = optixLaunchParams.sharedData.facetTextures[poly.texProps.textureOffset];

        const auto tu = (unsigned int)(__saturatef(hitLocation.x) * facetTex.texWidth_precise); // saturate for rounding errors that can result for values larger than 1.0
        const auto tv = (unsigned int)(__saturatef(hitLocation.y) * facetTex.texHeight_precise);
        unsigned int add = tu + tv * (facetTex.texWidth);

        //printf("Hit at %lf , %lf for tex %lf , %lf\n", hitLocation.x, hitLocation.y, facetTex.texWidth_precise, facetTex.texHeight_precise);
#ifdef BOUND_CHECK
        if(facetTex.texelOffset + add >= optixLaunchParams.simConstants.nbTexel){
            printf("[ABS] facetTex.texelOffset + add %u + %u>= %u is out of bounds [%u , %u] - [%d , %d] (%lf * %lf)x(%12.10lf * %12.10lf)\n", facetTex.texelOffset, add, optixLaunchParams.simConstants.nbTexel,tu,tv,facetTex.texWidth,facetTex.texHeight, hitLocation.x, facetTex.texWidth_precise, hitLocation.y, facetTex.texHeight_precise);}
#endif
        flowgpu::Texel& tex = optixLaunchParams.sharedData.texels[facetTex.texelOffset + add];

#ifdef HIT64
        const FLOAT_T velocity_factor = 2.0;
        const FLOAT_T ortSpeedFactor = 1.0;
        const FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781) * hitData.velocity*fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint64_t>(1));
        atomicAdd(&tex.sum_1_per_ort_velocity, velocity_factor / ortVelocity);
        atomicAdd(&tex.sum_v_ort_per_area, ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add]); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const float velocity_factor = 2.0f;
        const float ortSpeedFactor = 1.0f;
        const float ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) * hitData.velocity*fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint32_t>(1));
        atomicAdd(&tex.sum_1_per_ort_velocity, velocity_factor / ortVelocity);
        atomicAdd(&tex.sum_v_ort_per_area, ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add]); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif
    }


    // --------------------------------------
    // increase texture counters for reflections
    // --------------------------------------
    static __forceinline__ __device__
    unsigned int RecordBounceTexture(const flowgpu::Polygon& poly, MolPRD& hitData, const float3& rayOrigin, const float3& rayDir){

        const flowgpu::TriangleMeshSBTData &sbtData = *(const flowgpu::TriangleMeshSBTData*)optixGetSbtDataPointer();
        const float2 hitLocation = getHitLocation(optixGetTriangleBarycentrics(), sbtData.texcoord, optixGetPrimitiveIndex() * 3);

        flowgpu::FacetTexture& facetTex = optixLaunchParams.sharedData.facetTextures[poly.texProps.textureOffset];

        const auto tu = (unsigned int)(__saturatef(hitLocation.x) * facetTex.texWidth_precise); // saturate for rounding errors that can result for values larger than 1.0
        const auto tv = (unsigned int)(__saturatef(hitLocation.y) * facetTex.texHeight_precise);
        unsigned int add = tu + tv * (facetTex.texWidth);

        //printf("Pre Bounce Tex: %f = %f * %f * %f\n",ortVelocity,(optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) ,hitData.velocity,fabsf(dot(rayDir, poly.N)));
#ifdef BOUND_CHECK
        if(facetTex.texelOffset + add >= optixLaunchParams.simConstants.nbTexel){
            printf("[HIT] facetTex.texelOffset + add %u + %u>= %u is out of bounds [%u , %u] - [%d , %d] (%lf * %lf)x(%12.10lf * %12.10lf)\n", facetTex.texelOffset, add, optixLaunchParams.simConstants.nbTexel,tu,tv,facetTex.texWidth,facetTex.texHeight, hitLocation.x, facetTex.texWidth_precise, hitLocation.y, facetTex.texHeight_precise);}
#endif
        flowgpu::Texel& tex = optixLaunchParams.sharedData.texels[facetTex.texelOffset + add];

#ifdef HIT64
        const FLOAT_T velocity_factor = 1.0;
        const FLOAT_T ortSpeedFactor = 1.0;
        const FLOAT_T oriRatio = 1.0; // TODO: Part of particle
        const FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781) * hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint64_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (FLOAT_T)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (FLOAT_T)(oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add])); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const FLOAT_T velocity_factor = 1.0f;
        const FLOAT_T ortSpeedFactor = 1.0f;
        const FLOAT_T oriRatio = 1.0f; // TODO: Part of particle
        const FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) * hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint32_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (float)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (float)(oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add])); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif

#if DEBUG
        /*if(facetTex.texelOffset + add == 4){
            printf("Pre Bounce Texoffset4[%d]: %f\n", poly.parentIndex,optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add]);
        }*/
        if(optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add] >= 1e5)
            printf("Pre Bounce Tex[%d]: %f , %f - %f (%f) -> [%u + %u]\n", poly.parentIndex,
                oriRatio * velocity_factor / ortVelocity,oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add],
                ortVelocity,
                optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add], facetTex.texelOffset, add);
#endif
        // Return Texel index for reuse in post bounce processing
        return facetTex.texelOffset + add;
    }

    // --------------------------------------
    // increase texture counters for reflections
    // --------------------------------------
    static __forceinline__ __device__
    void RecordTransparentTexture(const flowgpu::Polygon& poly, MolPRD& hitData, const float3& rayOrigin, const float3& rayDir){

        const flowgpu::TriangleMeshSBTData &sbtData = *(const flowgpu::TriangleMeshSBTData*)optixGetSbtDataPointer();
        const float2 hitLocation = getHitLocation(optixGetTriangleBarycentrics(), sbtData.texcoord, optixGetPrimitiveIndex() * 3);

        flowgpu::FacetTexture& facetTex = optixLaunchParams.sharedData.facetTextures[poly.texProps.textureOffset];

        const unsigned int tu = (unsigned int)(__saturatef(hitLocation.x) * facetTex.texWidth_precise); // saturate for rounding errors that can result for values larger than 1.0
        const unsigned int tv = (unsigned int)(__saturatef(hitLocation.y) * facetTex.texHeight_precise);

        const unsigned int add = tu + tv * (facetTex.texWidth);

#ifdef BOUND_CHECK
        if(facetTex.texelOffset + add >= optixLaunchParams.simConstants.nbTexel){
            //float fixHitU = detU/__saturatef(det);
            //float fixHitV = detV/__saturatef(det);
            const unsigned int fixTU = (unsigned int)(__saturatef(hitLocation.x) * facetTex.texWidth_precise);
            const unsigned int fixTV = (unsigned int)(__saturatef(hitLocation.y) * facetTex.texHeight_precise);
            const unsigned int fixAdd = fixTU + fixTV * (facetTex.texWidth);

            printf("[TR] facetTex.texelOffset + add %u + %u >= %u is out of bounds [%u , %u] - [%d , %d] (%lf * %lf)x(%12.10lf * %12.10lf)\n", facetTex.texelOffset, add, optixLaunchParams.simConstants.nbTexel,tu,tv,facetTex.texWidth,facetTex.texHeight, hitLocation.x, facetTex.texWidth_precise, hitLocation.y, facetTex.texHeight_precise);
            printf("[TR] %u  [%u , %u] - [%d , %d] (%lf * %lf)x(%12.10lf * %12.10lf)\n",fixAdd , fixTU,fixTV,facetTex.texWidth,facetTex.texHeight, hitLocation.x, facetTex.texWidth_precise, hitLocation.y, facetTex.texHeight_precise);

            const float3 b = rayOrigin - poly.O;
            float det = poly.U.x * poly.V.y - poly.U.y * poly.V.x; // TODO: Pre calculate
            float detU = b.x * poly.V.y - b.y * poly.V.x;
            float detV = poly.U.x * b.y - poly.U.y * b.x;

            if(fabsf(det)>EPS32/* || detU/det > 1.0 || detV/det > 1.0*/)
                printf("[1][TP][Hit] Using 1\n" );

                printf("[1][TP][Hit] Dangerous determinant calculated for texture hit: %lf : %lf : %lf -> %lf : %lf\n",det,detU,detV,detU/det,detV/det);

                det = poly.U.y * poly.V.z - poly.U.z * poly.V.y; // TODO: Pre calculate
                detU = b.y * poly.V.z - b.z * poly.V.y;
                detV = poly.U.y * b.z - poly.U.z * b.y;
                if(fabsf(det)>EPS32/* || detU/det > 1.0 || detV/det > 1.0*/)
                    printf("[2][TP][Hit] Using 2\n" );

                    printf("[2][TP][Hit] Dangerous determinant calculated for texture hit: %lf : %lf : %lf -> %lf : %lf\n",det,detU,detV,detU/det,detV/det);
                    det = poly.U.z * poly.V.x - poly.U.x * poly.V.z; // TODO: Pre calculate
                    detU = b.z * poly.V.x - b.x * poly.V.z;
                    detV = poly.U.z * b.x - poly.U.x * b.z;
                    if(fabsf(det)>EPS32/* || detU/det > 1.0 || detV/det > 1.0*/)
                        printf("[3][TP][Hit] Using 3\n" );

                        printf("[3][TP][Hit] Dangerous determinant calculated for texture hit: %lf : %lf : %lf -> %lf : %lf\n",det,detU,detV,detU/det,detV/det);



        }
#endif
        flowgpu::Texel& tex = optixLaunchParams.sharedData.texels[facetTex.texelOffset + add];


#ifdef HIT64
        const FLOAT_T velocity_factor = 2.0;
        const FLOAT_T ortSpeedFactor = 2.0;
        const FLOAT_T oriRatio = 1.0;
        const FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781) * hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint64_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (FLOAT_T)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (FLOAT_T)(oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add])); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const float velocity_factor = 2.0f;
        const float ortSpeedFactor = 2.0f;
        const float oriRatio = 1.0f; // TODO: Part of particle
        const float ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) * hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint32_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (float)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (float)(oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add])); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif

        return;
    }

    // Same but without increased hit counter
    static __forceinline__ __device__
    void RecordPostBounceTexture(const flowgpu::Polygon& poly, MolPRD& hitData, const float3& rayDir, unsigned int texelIndex){
#ifdef HIT64
        const FLOAT_T velocity_factor = 1.0;
        const FLOAT_T ortSpeedFactor = 1.0;
        const FLOAT_T oriRatio = 1.0; // TODO: Part of particle
        const FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781) * hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component
        //printf("PostBounce Tex: %f = %f * %f * %f\n",ortVelocity,(optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) ,hitData.velocity,fabsf(dot(rayDir, poly.N)));

        flowgpu::Texel& tex = optixLaunchParams.sharedData.texels[texelIndex];
        atomicAdd(&tex.sum_1_per_ort_velocity, (FLOAT_T)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (FLOAT_T)(oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[texelIndex])); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const FLOAT_T velocity_factor = 1.0f;
        const FLOAT_T ortSpeedFactor = 1.0f;
        const FLOAT_T oriRatio = 1.0f; // TODO: Part of particle
        const FLOAT_T ortVelocity = (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) * hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component
        //printf("PostBounce Tex: %f = %f * %f * %f\n",ortVelocity,(optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) ,hitData.velocity,fabsf(dot(rayDir, poly.N)));

        flowgpu::Texel& tex = optixLaunchParams.sharedData.texels[texelIndex];
        atomicAdd(&tex.sum_1_per_ort_velocity, (float)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (float)(oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[texelIndex])); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif
        //printf("PostBounce Tex[%d]: %f , %f - %f (%f)\n", poly.parentIndex, oriRatio * velocity_factor / ortVelocity,oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[texelIndex],ortVelocity,optixLaunchParams.sharedData.texelInc[texelIndex]);
#if DEBUG
        if(optixLaunchParams.sharedData.texelInc[texelIndex] >= 1e5)
        printf("PostBounce Tex[%d]: %f , %f - %f (%f)\n", poly.parentIndex,
                oriRatio * velocity_factor / ortVelocity,
                oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[texelIndex],
                ortVelocity,
                optixLaunchParams.sharedData.texelInc[texelIndex]);
#endif
    }

    // --------------------------------------
    // increase texture counters for absorption (same as desorp)
    // --------------------------------------
    static __forceinline__ __device__
    void RecordAbsorpProfile(const flowgpu::Polygon &poly, MolPRD &hitData, const float3& rayOrigin, const float3& rayDir){

        const flowgpu::TriangleMeshSBTData &sbtData = *(const flowgpu::TriangleMeshSBTData*)optixGetSbtDataPointer();
        const float2 hitLocation = getHitLocation(optixGetTriangleBarycentrics(), sbtData.texcoord, optixGetPrimitiveIndex() * 3);

        unsigned int add = 0;
        if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileU){
            add = (unsigned int)(__saturatef(hitLocation.x) * PROFILE_SIZE);
        }
        else if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileV){
            add = (unsigned int)(__saturatef(hitLocation.y) * PROFILE_SIZE);
        }

        //printf("Hit at %lf , %lf for tex %lf , %lf\n", hitLocation.x, hitLocation.y, facetTex.texWidth_precise, facetTex.texHeight_precise);

#ifdef BOUND_CHECK
        if(poly.profProps.profileOffset + add >= optixLaunchParams.simConstants.nbProfSlices){
            printf("[ABS] poly.profProps.profileOffset + add %u >= %u is out of bounds\n", poly.profProps.profileOffset + add, optixLaunchParams.simConstants.nbProfSlices);}
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

    // --------------------------------------
    // increase profile counters for reflections
    // --------------------------------------
    static __forceinline__ __device__
    unsigned int RecordBounceProfile(const flowgpu::Polygon& poly, MolPRD& hitData, const float3& rayOrigin, const float3& rayDir){

        const flowgpu::TriangleMeshSBTData &sbtData = *(const flowgpu::TriangleMeshSBTData*)optixGetSbtDataPointer();
        const float2 hitLocation = getHitLocation(optixGetTriangleBarycentrics(), sbtData.texcoord, optixGetPrimitiveIndex() * 3);

        unsigned int add = 0;
        if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileU){
            add = (unsigned int)(__saturatef(hitLocation.x) * PROFILE_SIZE);

        }
        else if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileV){
            add = (unsigned int)(__saturatef(hitLocation.y) * PROFILE_SIZE);
        }

        //printf("Pre Bounce Tex: %f = %f * %f * %f\n",ortVelocity,(optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) ,hitData.velocity,fabsf(dot(rayDir, poly.N)));
#ifdef BOUND_CHECK
        if(poly.profProps.profileOffset + add >= optixLaunchParams.simConstants.nbProfSlices){
            printf("[HIT] poly.profProps.profileOffset + add %u [%lf -> %lf] >= %u is out of bounds\n", poly.profProps.profileOffset + add, hitLocation.x, __saturatef(hitLocation.x), optixLaunchParams.simConstants.nbProfSlices);
        }
#endif
        flowgpu::Texel& tex = optixLaunchParams.sharedData.profileSlices[poly.profProps.profileOffset + add];

#ifdef HIT64
        const FLOAT_T velocity_factor = 1.0;
        const FLOAT_T ortSpeedFactor = 1.0;
        const FLOAT_T oriRatio = 1.0; // TODO: Part of particle
        const FLOAT_T ortVelocity = hitData.velocity * fabs(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint64_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (FLOAT_T)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (FLOAT_T)(oriRatio * ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781))); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const FLOAT_T velocity_factor = 1.0f;
        const FLOAT_T ortSpeedFactor = 1.0f;
        const FLOAT_T oriRatio = 1.0f; // TODO: Part of particle
        const FLOAT_T ortVelocity = hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        //DEBUG_PRINT("Adding %f = %f * %f * %f", oriRatio *velocity_factor*ortVelocity, oriRatio, velocity_factor, ortVelocity);

        atomicAdd(&tex.countEquiv, static_cast<uint32_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (FLOAT_T)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (FLOAT_T)(oriRatio * ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f))); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif
        //printf("Pre Bounce Tex[%d]: %f , %f - %f (%f)\n", poly.parentIndex, oriRatio * velocity_factor / ortVelocity,oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add],ortVelocity,optixLaunchParams.sharedData.texelInc[facetTex.texelOffset + add]);

        // Return Texel index for reuse in post bounce processing
        return poly.profProps.profileOffset + add;
    }

    // Same but without increased hit counter
    static __forceinline__ __device__
    void RecordPostBounceProfile(const flowgpu::Polygon& poly, MolPRD& hitData, const float3& rayDir, unsigned int texelIndex){
#ifdef HIT64
        const FLOAT_T velocity_factor = 1.0;
        const FLOAT_T ortSpeedFactor = 1.0;
        const FLOAT_T oriRatio = 1.0; // TODO: Part of particle
        const FLOAT_T ortVelocity =  hitData.velocity * fabs(dot(rayDir, poly.N)); //surface-orthogonal velocity component
        //printf("PostBounce Tex: %f = %f * %f * %f\n",ortVelocity,(optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) ,hitData.velocity,fabsf(dot(rayDir, poly.N)));

        flowgpu::Texel& tex = optixLaunchParams.sharedData.profileSlices[texelIndex];
        atomicAdd(&tex.sum_1_per_ort_velocity, (FLOAT_T)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (FLOAT_T)(oriRatio * ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781))); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const float velocity_factor = 1.0f;
        const float ortSpeedFactor = 1.0f;
        const float oriRatio = 1.0f; // TODO: Part of particle
        const float ortVelocity =  hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component
        //printf("PostBounce Tex: %f = %f * %f * %f\n",ortVelocity,(optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) ,hitData.velocity,fabsf(dot(rayDir, poly.N)));

        flowgpu::Texel& tex = optixLaunchParams.sharedData.profileSlices[texelIndex];
        atomicAdd(&tex.sum_1_per_ort_velocity, (float)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (float)(oriRatio * ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f))); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif
        //printf("PostBounce Tex[%d]: %f , %f - %f (%f)\n", poly.parentIndex, oriRatio * velocity_factor / ortVelocity,oriRatio * ortSpeedFactor * ortVelocity * optixLaunchParams.sharedData.texelInc[texelIndex],ortVelocity,optixLaunchParams.sharedData.texelInc[texelIndex]);
    }

    // --------------------------------------
    // increase profile counters for reflections
    // --------------------------------------
    static __forceinline__ __device__
    unsigned int RecordTransparentProfile(const flowgpu::Polygon& poly, MolPRD& hitData, const float3& rayOrigin, const float3& rayDir){

        const flowgpu::TriangleMeshSBTData &sbtData = *(const flowgpu::TriangleMeshSBTData*)optixGetSbtDataPointer();
        const float2 hitLocation = getHitLocation(optixGetTriangleBarycentrics(), sbtData.texcoord, optixGetPrimitiveIndex() * 3);

        unsigned int add = 0;
        if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileU){
            add = (unsigned int)(__saturatef(hitLocation.x) * PROFILE_SIZE);
        }
        else if(poly.profProps.profileType == flowgpu::PROFILE_FLAGS::profileV){
            add = (unsigned int)(__saturatef(hitLocation.y) * PROFILE_SIZE);
        }

        //printf("Pre Bounce Tex: %f = %f * %f * %f\n",ortVelocity,(optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f) ,hitData.velocity,fabsf(dot(rayDir, poly.N)));
#ifdef BOUND_CHECK
        if(poly.profProps.profileOffset + add >= optixLaunchParams.simConstants.nbProfSlices){
            printf("[TRA] poly.profProps.profileOffset + add %u >= %u is out of bounds\n", poly.profProps.profileOffset + add, optixLaunchParams.simConstants.nbProfSlices);}
#endif
        flowgpu::Texel& tex = optixLaunchParams.sharedData.profileSlices[poly.profProps.profileOffset + add];

#ifdef HIT64
        const FLOAT_T velocity_factor = 2.0;
        const FLOAT_T ortSpeedFactor = 2.0;
        const FLOAT_T oriRatio = 1.0; // TODO: Part of particle
        const FLOAT_T ortVelocity = hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint64_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (FLOAT_T)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (FLOAT_T)(oriRatio * ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781))); // sum ortho_velocity[m/s] / cell_area[cm2]
#else
        const float velocity_factor = 2.0f;
        const float ortSpeedFactor = 2.0f;
        const float oriRatio = 1.0f; // TODO: Part of particle
        const float ortVelocity = hitData.velocity * fabsf(dot(rayDir, poly.N)); //surface-orthogonal velocity component

        atomicAdd(&tex.countEquiv, static_cast<uint32_t>(oriRatio));
        atomicAdd(&tex.sum_1_per_ort_velocity, (float)(oriRatio * velocity_factor / ortVelocity));
        atomicAdd(&tex.sum_v_ort_per_area, (float)(oriRatio * ortSpeedFactor * ortVelocity * (optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f))); // sum ortho_velocity[m/s] / cell_area[cm2]
#endif

        // Return Texel index for reuse in post bounce processing
        return poly.profProps.profileOffset + add;
    }

    static __forceinline__ __device__
    void recordAbsorption(const unsigned int& counterIdx, const flowgpu::Polygon& poly, MolPRD& prd, const float3& rayDir, const float3 rayOrigin)
    {
#ifdef HIT64
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781;
        const FLOAT_T ortVelocity = prd.velocity*fabsf(dot(rayDir, poly.N));
        const FLOAT_T hitEquiv = 1.0; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
        const FLOAT_T absEquiv = 1.0; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
#else
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f;
        const FLOAT_T ortVelocity = prd.velocity*fabsf(dot(rayDir, poly.N));
        const FLOAT_T hitEquiv = 1.0f; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
        const FLOAT_T absEquiv = 1.0f; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)

#endif

#ifdef BOUND_CHECK
        if(counterIdx >= optixLaunchParams.simConstants.nbFacets * EXTRAFACETCOUNTERS){
            printf("facIndex %u >= %u is out of bounds\n", counterIdx, optixLaunchParams.simConstants.nbFacets * EXTRAFACETCOUNTERS);
        }
#endif
        increaseHitCounterAbsorp(optixLaunchParams.hitCounter[counterIdx], hitEquiv, absEquiv, ortVelocity, velFactor, prd.velocity);

#ifdef WITH_TEX
        if (poly.texProps.textureFlags & flowgpu::TEXTURE_FLAGS::countAbs){
            RecordAbsorpTexture(poly, prd, rayOrigin, rayDir);
        }
#endif
#ifdef WITH_PROF
        if (poly.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile){
            RecordAbsorpProfile(poly, prd, rayOrigin, rayDir);
        }
#endif

// terminate if exit has been called
#ifdef WITHDESORPEXIT
#ifdef PAYLOAD_DIRECT
        if(optixLaunchParams.perThreadData.currentMoleculeData[getWorkIndex()].hasToTerminate==1){
            optixLaunchParams.perThreadData.currentMoleculeData[getWorkIndex()].hasToTerminate=2;
        }
#else
        if(prd.hasToTerminate==1){
            prd.hasToTerminate=2;
        }
#endif //PAYLOAD_DIRECT
#endif
    }

    static __forceinline__ __device__
    void recordBounce(const unsigned int& bufferIndex, const unsigned int& counterIdx, const flowgpu::Polygon& poly, MolPRD& prd, const float3& rayDir, const float3 rayOrigin,
#ifdef RNG_BULKED
            const RN_T* randFloat, unsigned int& randInd, unsigned int& randOffset)
#else
                      curandState_t* states)
#endif
    {
        // TODO: Save somewhere as a shared constant instead of repetively evaluating
#ifdef HIT64
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781;
        FLOAT_T ortVelocity = prd.velocity*fabsf(dot(rayDir, poly.N));
        const FLOAT_T hitEquiv = 1.0; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
        const FLOAT_T absEquiv = 1.0; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
#else
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f;
        FLOAT_T ortVelocity = prd.velocity*fabsf(dot(rayDir, poly.N));
        const FLOAT_T hitEquiv = 1.0f; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
        const FLOAT_T absEquiv = 1.0f; //1.0*prd.orientationRatio; // hit=1.0 (only changed for lowflux mode)
#endif

#ifdef BOUND_CHECK
        if(counterIdx >= optixLaunchParams.simConstants.nbFacets * EXTRAFACETCOUNTERS){
            printf("facIndex %u >= %u is out of bounds\n", counterIdx, optixLaunchParams.simConstants.nbFacets * EXTRAFACETCOUNTERS);
        }
#endif
        // 1. Increment counters
        increaseHitCounterBounce(optixLaunchParams.hitCounter[counterIdx], hitEquiv, ortVelocity, velFactor, prd.velocity);

#ifdef WITH_TEX
        unsigned int texelIndex = 1e8;
        if (poly.texProps.textureFlags & flowgpu::TEXTURE_FLAGS::countRefl){
            texelIndex = RecordBounceTexture(poly, prd, rayOrigin, rayDir);
        }
#endif // WITH_TEX
#ifdef WITH_PROF
        unsigned int profileIndex = 1e8;
        //printf("BOUNCE?? %d != %d == %d\n",(int)poly.profProps.profileType, (int)flowgpu::PROFILE_FLAGS::noProfile,(int)poly.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile);
        if (poly.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile){
            profileIndex = RecordBounceProfile(poly, prd, rayOrigin, rayDir);
        }
#endif

        // also increase a nbBounce counter if there is need (e.g. recursion)
        prd.inSystem = ACTIVE_PARTICLE;

        // 2. Relaunch particle

        //-----------
        // new ray velocity (for now only perfect thermalization (needs accomodationFactor)
        //-----------
        //TODO: Calculate new Velocity from CFD or directly
        //if (optixLaunchParams.simConstants.useMaxwell) prd.velocity = GenerateRandomVelocity(collidedFacet->sh.CDFid);
        //else
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
            prd.velocity = randomVelo(poly, rnd_val);

        }
        else {
            prd.velocity = getNewVelocity(poly, optixLaunchParams.simConstants.gasMass);
        }

#if defined(RNG_BULKED)
        OOB_CHECK("randInd", randInd + randOffset, optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y);
#endif
        if(prd.facetHitSide == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE || prd.inSystem != TRANSPARENT_HIT) {
            //if (bufferIndex == 0 && poly.parentIndex == 102) printf("[%d] Front direction!\n", poly.parentIndex);
#if defined(RNG_BULKED)
            prd.postHitDir = getNewDirection(prd, poly, randFloat, randInd, randOffset);
#else
            prd.postHitDir = getNewDirection(prd, poly, states);
#endif
        }
        else {
            /*if (bufferIndex == 0)
                printf("[%d] Back direction!\n", poly.parentIndex);*/
            //else if (bufferIndex == 0 && poly.parentIndex != 102) printf("[%d] Back direction!\n", poly.parentIndex);
#if defined(RNG_BULKED)
            prd.postHitDir = getNewReverseDirection(prd, poly, randFloat, randInd, randOffset);
#else
            prd.postHitDir = getNewReverseDirection(prd, poly, states);
#endif
            //prd.inSystem = ACTIVE_PARTIC;
        }

#if defined(DEBUG) && defined(RNG_BULKED)
        if(isnan(prd.postHitDir.x)) {
            DEBUG_PRINT("HIT[%d][%d] ray dir NaN -> %d / %d + %d\n",bufferIndex,prd.currentDepth, prd.inSystem, randInd, randOffset);
        }
#endif

        // 3. Increment counters for post bounce / outgoing particles
#ifdef HIT64
        ortVelocity = prd.velocity*fabs(dot(prd.postHitDir, poly.N));
#else
        ortVelocity = prd.velocity*fabsf(dot(prd.postHitDir, poly.N));
#endif

        increaseHitCounterPostBounce(optixLaunchParams.hitCounter[counterIdx], hitEquiv, ortVelocity, velFactor, prd.velocity);

#ifdef WITH_TEX
        if (poly.texProps.textureFlags & flowgpu::TEXTURE_FLAGS::countRefl)
                RecordPostBounceTexture(poly, prd, prd.postHitDir, texelIndex);
#endif
#ifdef WITH_PROF
        if (poly.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile){
                RecordPostBounceProfile(poly, prd, prd.postHitDir, profileIndex);
            }
#endif

#if defined(GPUNBOUNCE)
        ++optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].nbBounces;
        ++prd.nbBounces;
#endif
    }

    //------------------------------------------------------------------------------
    //
    // closest hit and miss programs for the molflow ray tracer standard molecules
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __anyhit__molecule()
    {
        // we don't need it
    }

    extern "C" __global__ void __closesthit__molecule_triangle()
    {

        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

        //const int   primID = optixGetPrimitiveIndex();
        const float3 ray_orig = optixGetWorldRayOrigin();
        const float3 ray_dir  = optixGetWorldRayDirection();
        const float  ray_t    = optixGetRayTmax();

        //const uint2 launchIndex = make_uint2(optixGetLaunchIndex());
        const unsigned int bufferIndex = getWorkIndex();

#ifdef DEBUGMISS
        const unsigned int missIndex = bufferIndex*NMISSES;
        optixLaunchParams.perThreadData.missBuffer[missIndex] = 0;
#endif
        const unsigned int facetHitKind = optixGetHitKind();

#ifdef PAYLOAD_DIRECT
        MolPRD myPrd = getMolPRD();
        MolPRD* prd = &myPrd;
#else
        MolPRD* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
#endif

        if(prd->hitFacetId == optixGetPrimitiveIndex()){
            // --------------------------------------
            // Self Intersection (SI)
            // Classify as SI when collided facet is the same as previously (hitFacetId)
            // 1. Label hit as SI
            // 2. Save intersection side (front/back)
            // 3. Record new ray origin/direction
            // --------------------------------------

            DEBUG_PRINT("[%d] source and goal facet equal %d : %8.6f,%8.6f,%8.6f -> %8.6f,%8.6f,%8.6f : [%.5e .. %.5e] (tri)\n",(blockDim.x * blockIdx.x + threadIdx.x), prd->hitFacetId, ray_orig.x,ray_orig.y,ray_orig.z,ray_dir.x,ray_dir.y,ray_dir.z, optixGetRayTmin(),ray_t);
            // if(prd->inSystem == 4) DEBUG_PRINT("[%d] [backface] source and goal facet equal %d : %8.6f,%8.6f,%8.6f -> %8.6f,%8.6f,%8.6f : [%.5e .. %.5e] (tri)\n",(blockDim.x * blockIdx.x + threadIdx.x), prd->hitFacetId, ray_orig.x,ray_orig.y,ray_orig.z,ray_dir.x,ray_dir.y,ray_dir.z, optixGetRayTmin(),ray_t);

            prd->facetHitSide = facetHitKind;
            if(facetHitKind == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE) {
                prd->inSystem = SELF_INTERSECTION;
                if(bufferIndex==0)
                    DEBUG_PRINT("[%u] back self hit detected -> %u -> %u for n=%d\n", bufferIndex, prd->inSystem, sbtData.poly[prd->hitFacetId].parentIndex,prd->nbBounces);
            }
            else{
                prd->inSystem = SELF_INTERSECTION;
                if(bufferIndex==0)
                    DEBUG_PRINT("[%u] front self hit detected -> %u -> %u for n=%d\n", bufferIndex, prd->inSystem, sbtData.poly[prd->hitFacetId].parentIndex,prd->nbBounces);
            }
            const flowgpu::Polygon& poly  = sbtData.poly[prd->hitFacetId];


            prd->hitPos = ray_orig;
            prd->postHitDir = ray_dir;
            //prd.currentDepth = 0;
#ifdef PAYLOAD_DIRECT
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = prd->hitPos;
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].postHitDir = prd->postHitDir;
            setMolPRD(myPrd);
#endif
            return;
        }

        // Now check for other types of hits

        // first load hit data
        prd->hitT = ray_t;
        prd->hitPos = ray_orig + ray_t * ray_dir;
        prd->hitFacetId = optixGetPrimitiveIndex();
        prd->facetHitSide = facetHitKind;

        const unsigned int counterIdx = prd->hitFacetId + (bufferIndex%(EXTRAFACETCOUNTERS)) * optixLaunchParams.simConstants.nbFacets;
        const flowgpu::Polygon& poly  = sbtData.poly[prd->hitFacetId];

        // In case of Front hit vs back hit
        if(facetHitKind == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE) {
            //prd->inSystem = TRANSPARENT_HIT;
        }
        else if(bufferIndex==0) {
            DEBUG_PRINT("[%u] back hit detected -> Mol: %u on F#%u\n", bufferIndex, prd->inSystem, poly.parentIndex);
        }

        // TODO: Consider maxwelldistribution or not and oriRatio for lowFlux and desorbtion/absorbtion
        //IncreaseFacetCounter(
        // iFacet, sHandle->currentParticle.flightTime, 1, 0, 0, 1.0 / ortVelocity,
        // (sHandle->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity);
        // TODO: if bounce
        // TODO: queue bounces back in pipeline instead of recursively following them
        // stop for some point due to recursion limits

#ifdef HIT64
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781;
        const FLOAT_T ortVelocity = prd->velocity*fabsf(dot(ray_dir, poly.N));
        const FLOAT_T hitEquiv = 1.0; //1.0*prd->orientationRatio; // hit=1.0 (only changed for lowflux mode)
        const FLOAT_T absEquiv = 1.0; //1.0*prd->orientationRatio; // hit=1.0 (only changed for lowflux mode)
#else
        const float velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f;
        const float ortVelocity = prd->velocity*fabsf(dot(ray_dir, poly.N));
        const float hitEquiv = 1.0f; //1.0*prd->orientationRatio; // hit=1.0 (only changed for lowflux mode)
        const float absEquiv = 1.0f; //1.0*prd->orientationRatio; // hit=1.0 (only changed for lowflux mode)
#endif

#ifndef WITH_TRANS
        if(poly.facProps.is2sided && poly.facProps.opacity == 0){
        //if(poly.facProps.is2sided){
            DEBUG_PRINT("[%d] 2sided facet hit %d / %d : %8.6f,%8.6f,%8.6f -> %8.6f,%8.6f,%8.6f : [%.5e .. %.5e] (tri)\n",(blockDim.x * blockIdx.x + threadIdx.x), prd->hitFacetId, poly.parentIndex, ray_orig.x,ray_orig.y,ray_orig.z,ray_dir.x,ray_dir.y,ray_dir.z, optixGetRayTmin(),ray_t);
            // replace like with self intersection, but keep position etc.
            prd->inSystem = TRANSPARENT_HIT;

#ifdef PAYLOAD_DIRECT
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = prd->hitPos;
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].postHitDir = ray_dir;
#endif

            increaseTransparentCounter(optixLaunchParams.hitCounter[counterIdx], hitEquiv, ortVelocity, velFactor, prd->velocity);
            if (poly.texProps.textureFlags & flowgpu::TEXTURE_FLAGS::countTrans) {
                RecordTransparentTexture(poly, *prd, prd->hitPos, ray_dir);
            }

#ifdef PAYLOAD_DIRECT
            setMolPRD(myPrd);
#else
            prd->postHitDir = ray_dir;
#endif
            return;
        }
#endif // WITH_TRANS


#ifdef RNG_BULKED
        const RN_T* randFloat = optixLaunchParams.randomNumbers;
        unsigned int randInd = (optixLaunchParams.simConstants.nbRandNumbersPerThread)*(bufferIndex);
        unsigned int randOffset = optixLaunchParams.perThreadData.randBufferOffset[bufferIndex];
#endif

        //TODO: AtomicAdd for smaller hitCounter structure, just per threadIdx
        if(poly.facProps.stickingFactor>=0.99999f || ((poly.facProps.stickingFactor > 0.0f) && (
#ifdef RNG_BULKED
                randFloat[(unsigned int)(randInd + randOffset++)]
#else
                generate_rand(optixLaunchParams.randomNumbers, bufferIndex)
#endif
< (poly.facProps.stickingFactor)))){
            //--------------------------
            //-------- ABSORPTION ------
            //--------------------------

            recordAbsorption(counterIdx,poly,*prd, ray_dir, prd->hitPos);
#ifdef RNG_BULKED
            optixLaunchParams.perThreadData.randBufferOffset[bufferIndex] = randOffset;
#endif
            prd->inSystem = NEW_PARTICLE;
            prd->currentDepth = 0;
#ifdef GPUNBOUNCE
            /*optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].nbBounces =*/ prd->nbBounces = 0;
#ifdef PAYLOAD_DIRECT
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].nbBounces = 0;
#endif // PAYLOAD_DIRECT
#endif // GPUNBOUNCE

#ifdef PAYLOAD_DIRECT
            setMolPRD(myPrd);
#endif
            return;
        }
        // Process a memory error
        else if(poly.facProps.stickingFactor<-0.00001f){
            //

            printf("-- %lf Err ----- \n",poly.facProps.stickingFactor);
            return;
        }
        // Process a bounce/reflection
        else{

            //--------------------------
            //-------- REFLECTION ------
            //--------------------------
#ifdef RNG_BULKED
            recordBounce(bufferIndex, counterIdx, poly, *prd, ray_dir, prd->hitPos, randFloat, randInd, randOffset);
            // Write temporary local variables back to shared memory
            optixLaunchParams.perThreadData.randBufferOffset[bufferIndex] = randOffset;
#else
            recordBounce(bufferIndex, counterIdx, poly, *prd, ray_dir, prd->hitPos, optixLaunchParams.randomNumbers);
#endif
#ifdef PAYLOAD_DIRECT
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].facetHitSide = prd->facetHitSide;
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = prd->hitPos;
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].postHitDir = prd->postHitDir;
#endif

            // No recursion or end of recursion
            if(prd->currentDepth >= optixLaunchParams.simConstants.maxDepth){
                //printf("-- [%d] Max Bounces reached on depth %d, resetting for next call ... ----- \n",fbIndex, prd->currentDepth);
                prd->currentDepth = 0;
            }
            // WITH RECURSION
            else {
                ++prd->currentDepth;

#if defined(DEBUGPOS)
                if(bufferIndex==0){
                    const unsigned int posIndexOff = optixLaunchParams.perThreadData.posOffsetBuffer_debug[(unsigned int)(bufferIndex)]++;
                    if(posIndexOff<NBPOSCOUNTS){
                        const unsigned int posIndex = bufferIndex*NBPOSCOUNTS+posIndexOff;
                        //printf("[%d] my pos is %d\n", (unsigned int)(ix+iy*optixLaunchParams.simConstants.size.x), posIndex);
                        optixLaunchParams.perThreadData.positionsBuffer_debug[posIndex] = prd->hitPos;
                        optixLaunchParams.perThreadData.positionsType_debug[posIndex] = 3;
                    }
                }
#endif

#if defined(DEBUG) && defined(RNG_BULKED)
                if(isnan(optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].postHitDir.x)) {
                    unsigned int randInd = optixLaunchParams.simConstants.nbRandNumbersPerThread*(bufferIndex);
                    unsigned int randOffset = optixLaunchParams.perThreadData.randBufferOffset[bufferIndex];

                    DEBUG_PRINT("TP[%d][%d] ray dir NaN -> %d / %d + %d\n",bufferIndex,prd->currentDepth, prd->inSystem, randInd, randOffset);
                }
#endif

                {
#ifdef BOUND_CHECK
                    if(prd->hitFacetId >= optixLaunchParams.simConstants.nbFacets){
                        printf("[RecursiveReflection] facIndex %u >= %u is out of bounds (%u)\n", prd->hitFacetId, optixLaunchParams.simConstants.nbFacets, prd->inSystem);
                    }
#endif
                    apply_offset(poly, *prd, optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos);
                    if(poly.facProps.endangered_neighbor) { // with offset to center
                        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = offset_to_center(prd->hitPos, prd->hitFacetId, poly);
                    }
                }

#ifdef PAYLOAD_DIRECT
                int hi_vel = __double2hiint(prd->velocity);
                int lo_vel = __double2loint(prd->velocity);
#else
                // Pass the current payload registers through to the shadow ray.
                unsigned int p0 = optixGetPayload_0();
                unsigned int p1 = optixGetPayload_1();
#endif
                optixTrace(optixLaunchParams.traversable,
                           optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos,
                           optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].postHitDir,
                           0.f,//1e-4f,//0.f,    // tmin
                           1e20f,  // tmax
                           0.0f,   // rayTime
                           OptixVisibilityMask(255),
                           RAY_FLAGS,
                           RayType::RAY_TYPE_MOLECULE,             // SBT offset
                           RayType::RAY_TYPE_COUNT,               // SBT stride
                           RayType::RAY_TYPE_MOLECULE,             // missSBTIndex
#ifdef PAYLOAD_DIRECT
                        // direct
               reinterpret_cast<unsigned int&>(prd->currentDepth),
               reinterpret_cast<unsigned int&>(prd->inSystem),
               reinterpret_cast<unsigned int&>(prd->hitFacetId),
               reinterpret_cast<unsigned int&>(prd->hitT),
               reinterpret_cast<unsigned int&>( hi_vel ),
               reinterpret_cast<unsigned int&>( lo_vel ),
               reinterpret_cast<unsigned int&>(prd->nbBounces),
               reinterpret_cast<unsigned int&>(prd->facetHitSide)
#else
                        // ptr
                           p0, p1
#endif
                );
#ifdef PAYLOAD_DIRECT
                prd->velocity = __hiloint2double(hi_vel,lo_vel );
#endif
            }
#ifdef PAYLOAD_DIRECT
            setMolPRD(myPrd);
#endif
        } // end bounce
    }

#ifdef WITH_TRANS
    extern "C" __global__ void __closesthit__transparent_triangle()
    {

        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

        //const int   primID = optixGetPrimitiveIndex();
        const float3 ray_orig = optixGetWorldRayOrigin();
        const float3 ray_dir  = optixGetWorldRayDirection();
        const float  ray_t    = optixGetRayTmax();

        //const uint2 launchIndex = make_uint2(optixGetLaunchIndex());
        const unsigned int bufferIndex = getWorkIndex();

        /*#ifdef DEBUG
            printf("[%u] transparent hit detected\n", bufferIndex);
        #endif*/

#ifdef PAYLOAD_DIRECT
        MolPRD myPrd = getMolPRD();
        MolPRD* prd = &myPrd;
#else
        MolPRD* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
#endif
        const int facetHitKind = optixGetHitKind();
        const int hitFacetId = optixGetPrimitiveIndex();

        // self intersection
        // just try to offset continously
        if(prd->hitFacetId == hitFacetId){
#ifdef DEBUG
            /*if(bufferIndex == 0)
                printf("[%u->%u] transparent hit self inter\n", bufferIndex, prd->nbBounces);
            */
#endif
            prd->hitFacetId = hitFacetId;
            prd->hitT = ray_t;
            prd->postHitDir = ray_dir;
            prd->hitPos = ray_orig + ray_t * ray_dir;
            //prd->inSystem = SELF_INTERSECTION;
            prd->inSystem = TRANSPARENT_HIT;
            optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = ray_orig;

            setMolPRD(myPrd);
            return;
        }

        prd->hitT = ray_t;
        prd->hitPos = ray_orig + ray_t * ray_dir;

        // first add facet hits
        const unsigned int counterIdx = hitFacetId + (bufferIndex%(EXTRAFACETCOUNTERS)) * optixLaunchParams.simConstants.nbFacets;

        //Register (orthogonal) velocity
        const flowgpu::Polygon& poly  = sbtData.poly[hitFacetId];

        /*#ifdef DEBUG
            printf("[%u] trans 2side check\n", bufferIndex);
        #endif*/

#ifdef DEBUG
        /*if(facetHitKind != OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE && !poly.facProps.is2sided) {
            //__miss__molecule();

            printf("[%u][%u] Back face hit on 1-sided facet should never happen!\n",bufferIndex,poly.parentIndex);
            prd->hitFacetId = hitFacetId;
            prd->facetHitSide = facetHitKind;
            prd->inSystem = NEW_PARTICLE;
            setMolPRD(prd);
            return;
        }*/
        if(facetHitKind != OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE && !poly.facProps.is2sided) {
            //__miss__molecule();

            //printf("[%u][%u] Back face hit on 1-sided facet should never happen!\n",bufferIndex,poly.parentIndex);
            /*prd->hitFacetId = hitFacetId;
            prd->facetHitSide = facetHitKind;
            prd->inSystem = NEW_PARTICLE;
            setMolPRD(prd);
            return;*/
        }
#endif
        /*#ifdef DEBUG
            printf("[%u] trans hit!!\n", bufferIndex);
        #endif*/

#ifdef HIT64
        const FLOAT_T velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0 : 1.1781;
        const FLOAT_T ortVelocity = prd->velocity*fabsf(dot(ray_dir, poly.N));
        const FLOAT_T hitEquiv = 1.0; //1.0*prd->orientationRatio; // hit=1.0 (only changed for lowflux mode)
#else
        const float velFactor = optixLaunchParams.simConstants.useMaxwell ? 1.0f : 1.1781f;
        const float ortVelocity = prd->velocity*fabsf(dot(ray_dir, poly.N));
        const float hitEquiv = 1.0f; //1.0*prd->orientationRatio; // hit=1.0 (only changed for lowflux mode)
#endif

        // replace like with self intersection, but keep position etc.
        prd->inSystem = TRANSPARENT_HIT;

        /*if(optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE)
            prd->inSystem = TRANSPARENT_HIT;
        else
            prd->inSystem = TRANSPARENT_BACK_HIT;*/

        // Only increase counters if not dealing with self intersection
        if(prd->hitFacetId != hitFacetId){

            //printf("Registering trans hit on status %u\n",prd->inSystem);

            increaseTransparentCounter(optixLaunchParams.hitCounter[counterIdx], hitEquiv, ortVelocity, velFactor, prd->velocity);
#ifdef WITH_TEX
            if (poly.texProps.textureFlags & flowgpu::TEXTURE_FLAGS::countTrans) {
                RecordTransparentTexture(poly, *prd, prd->hitPos, ray_dir);
            }
#endif // WITH_TEX
#ifdef WITH_PROF
            //printf("BOUNCE? %d != %d == %d\n",(int)poly.profProps.profileType, (int)flowgpu::PROFILE_FLAGS::noProfile,(int)poly.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile);
            if (poly.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile){
                RecordTransparentProfile(poly, *prd, prd->hitPos, ray_dir);
            }
#endif
        }
        prd->hitFacetId = hitFacetId;
        prd->postHitDir = ray_dir;
        prd->facetHitSide = facetHitKind;

        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].hitPos = prd->hitPos;
        optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].postHitDir = ray_dir;

        /*#ifdef DEBUG
            printf("[%u] trans end!!\n", bufferIndex);
        #endif*/
        setMolPRD(myPrd);
    }
#endif // WITH_TRANS

#if defined(DEBUGMISS)
    // Parallelogram intersection based on the SDK optixWhitted example
    extern "C" __device__ void is_parallelogram_miss(uint32_t primID)
    {
        const PolygonMeshSBTData &sbtData = *(const PolygonMeshSBTData*)optixGetSbtDataPointer();

        //const int   primID = optixGetPrimitiveIndex();

        float3 ray_orig = optixGetWorldRayOrigin();
        float3 ray_dir = optixGetWorldRayDirection();

        /*if(blockDim.x * blockIdx.x + threadIdx.x == 3 && ray_dir.x > -0.26 && ray_dir.x < -0.24)
            primID=162;*/

        ray_dir = make_float3(-1.0,-1.0,-1.0) * ray_dir;

        const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

        const flowgpu::Polygon& poly  = sbtData.poly[primID];
        const float det = dot(poly.Nuv, ray_dir);
        printf("[%d] intersection det = %12.10f\n", primID, det);
        if(det > 0.0) {
            const float iDet = 1.0 / det;
            float3 intZ = ray_orig - poly.O;

            const float u = iDet * DET33(intZ.x, poly.V.x, ray_dir.x,
                                         intZ.y, poly.V.y, ray_dir.y,
                                         intZ.z, poly.V.z, ray_dir.z);
            printf("[%d] intersection   u = %12.10f\n",primID, u);

            const float v = iDet * DET33(poly.U.x, intZ.x, ray_dir.x,
                                         poly.U.y, intZ.y, ray_dir.y,
                                         poly.U.z, intZ.z, ray_dir.z);
            printf("[%d] intersection   v = %12.10f\n",primID, v);

            const float d = iDet * dot(poly.Nuv, intZ);
            printf("[%d] intersection   d = %12.10f\n",primID, d);

        }
    }
#endif //DEBUGMISS

    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // reset molecule position to start from source again and count the leak
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__molecule()
    {
        //int3 &prd = *(int3*)getPRD<int3>();

#ifdef PAYLOAD_DIRECT
        MolPRD myPrd = getMolPRD();
        MolPRD* prd = &myPrd;
#else
        MolPRD* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
#endif

        if(prd->inSystem == 0) // skip miss on desorption
            return;

#if defined(DEBUGMISS)
        const float3 ray_orig = optixGetWorldRayOrigin();
        const float3 ray_dir  = optixGetWorldRayDirection();
        const float  ray_t    = optixGetRayTmax();


        const unsigned int fbIndex = getWorkIndex();
        const unsigned int missIndex = fbIndex*NMISSES;
        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();


#if !defined(GPUNBOUNCE)
        DEBUG_PRINT("(%d) miss[%d -> %d -> %d][%d] "
               "(%12.10f , %12.10f , %12.10f) --> (%12.10f , %12.10f , %12.10f) = %e \n",
               prd->inSystem, fbIndex, prd->hitFacetId, sbtData.poly[prd->hitFacetId].parentIndex, missIndex,
               ray_orig.x, ray_orig.y , ray_orig.z , ray_dir.x, ray_dir.y , ray_dir.z, ray_t);
#else
        DEBUG_PRINT("(%d , %d) miss[%d -> %d -> %d][%d] "
               "(%12.10f , %12.10f , %12.10f) --> (%12.10f , %12.10f , %12.10f) = %e \n",
               prd->inSystem, prd->nbBounces, fbIndex, prd->hitFacetId, sbtData.poly[prd->hitFacetId].parentIndex, missIndex,
               ray_orig.x, ray_orig.y , ray_orig.z , ray_dir.x, ray_dir.y , ray_dir.z, ray_t);
#endif

        for(int i=missIndex+1; i <= missIndex+optixLaunchParams.perThreadData.missBuffer[missIndex];i++){
            DEBUG_PRINT("miss[%d -> %d -> %d] at %d\n",
                    blockDim.x * blockIdx.x + threadIdx.x,
                    missIndex,
                    optixLaunchParams.perThreadData.missBuffer[missIndex],
                   optixLaunchParams.perThreadData.missBuffer[i]);

            is_parallelogram_miss(optixLaunchParams.perThreadData.missBuffer[i]);

        }


        optixLaunchParams.perThreadData.missBuffer[missIndex] = 0;
#endif //DEBUGMISS

/*optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].hitPos = prd->hitPos;
optixLaunchParams.perThreadData.currentMoleculeData[fbIndex].postHitDir = prd->postHitDir;*/

#ifdef DEBUGLEAKPOS
        {
            const float3 ray_orig = optixGetWorldRayOrigin();
            const float3 ray_dir  = optixGetWorldRayDirection();
            const float  ray_t    = optixGetRayTmax();


            const unsigned int fbIndex = getWorkIndex();



#if !defined(GPUNBOUNCE)
            DEBUG_PRINT("(%d) miss[%d -> %d] (%12.10f , %12.10f , %12.10f) --> (%12.10f , %12.10f , %12.10f) = %e \n",
               prd->inSystem, fbIndex, prd->hitFacetId,
               ray_orig.x, ray_orig.y , ray_orig.z , ray_dir.x, ray_dir.y , ray_dir.z, ray_t);
#else
            DEBUG_PRINT("(%d , %d) miss[%d -> %d] (%12.10f , %12.10f , %12.10f) --> (%12.10f , %12.10f , %12.10f) = %e \n",
               prd->inSystem, prd->nbBounces, fbIndex, prd->hitFacetId,
               ray_orig.x, ray_orig.y , ray_orig.z , ray_dir.x, ray_dir.y , ray_dir.z, ray_t);
#endif

            const unsigned int posIndexOff = optixLaunchParams.perThreadData.leakPosOffsetBuffer_debug[fbIndex]++;
            if(posIndexOff<NBCOUNTS){
                const unsigned int posIndex = (fbIndex)*NBCOUNTS+posIndexOff;
                //printf("[%d] my pos is %d\n", (unsigned int)(fbIndex), posIndex);
                optixLaunchParams.perThreadData.leakPositionsBuffer_debug[posIndex] = ray_orig;
                optixLaunchParams.perThreadData.leakDirectionsBuffer_debug[posIndex] = ray_dir;

            }
        }
#endif

        // Try to relaunch a molecule for some time until removing it from the system
        // Differentiate between physical and single precision leaks here

        //if(prd->inSystem > 20) {
        //optixLaunchParams.sharedData.missCounter += 1;
        atomicAdd(&optixLaunchParams.sharedData.missCounter[prd->hitFacetId], 1);

        // Reset particle
        initParticle(*prd);

// terminate if exit has been called
#ifdef WITHDESORPEXIT
#ifdef PAYLOAD_DIRECT
        if(optixLaunchParams.perThreadData.currentMoleculeData[getWorkIndex()].hasToTerminate==1){
            optixLaunchParams.perThreadData.currentMoleculeData[getWorkIndex()].hasToTerminate=2;
        }
#else
        if(prd->hasToTerminate==1){
            prd->hasToTerminate=2;
        }
#endif //PAYLOAD_DIRECT
#endif
        /*}
        else{
            prd->inSystem = max(3,prd->inSystem+1);
        }*/
#if defined(PAYLOAD_DIRECT)
        setMolPRD(myPrd);
#endif
    }


} // ::flowgpu
