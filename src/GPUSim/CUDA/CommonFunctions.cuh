//
// Created by pbahr on 09/03/2020.
// TODO: Find a better name
//

#ifndef MOLFLOW_PROJ_COMMONFUNCTIONS_CUH
#define MOLFLOW_PROJ_COMMONFUNCTIONS_CUH

#include <math_constants.h>
#include "cuda_runtime.h"
#include "helper_math.h"
#include "LaunchParams.h"
#include "GPUDefines.h"

#include <cooperative_groups.h>
#include <curand_kernel.h>

/*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
extern "C" __constant__ flowgpu::LaunchParams optixLaunchParams;


#define RAY_FLAGS OPTIX_RAY_FLAG_NONE \
                    | OPTIX_RAY_FLAG_DISABLE_ANYHIT \
                    | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
#define EPS32 1e-12f

#define DET33(_11,_12,_13,_21,_22,_23,_31,_32,_33)  \
  ((_11)*( (_22)*(_33) - (_32)*(_23) ) +            \
   (_12)*( (_23)*(_31) - (_33)*(_21) ) +            \
   (_13)*( (_21)*(_32) - (_31)*(_22) ))

#define DET33_ROW(_1,_2,_3)  \
  ((_1.x)*( (_2.y)*(_3.z) - (_2.z)*(_3.y) ) +            \
   (_2.x)*( (_3.y)*(_1.z) - (_3.z)*(_1.y) ) +            \
   (_3.x)*( (_1.y)*(_2.z) - (_1.z)*(_2.y) ))

#define DOT(v1, v2)  \
  ((v1.x)*(v2.x) + (v1.y)*(v2.y) + (v1.z)*(v2.z))

#define CROSS(a, b)   \
  (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)

#define float3_as_args(u) \
    reinterpret_cast<uint32_t&>((u).x), \
    reinterpret_cast<uint32_t&>((u).y), \
    reinterpret_cast<uint32_t&>((u).z)
#ifdef DEBUG
#define DEBUG 1
#endif
#if defined(DEBUG) && DEBUG > 0
#define DEBUG_PRINT(fmt, ...) printf("DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) /* Don't do anything in release builds */
#endif

// out of bounds print
#if defined(BOUND_CHECK) && defined(DEBUG)
#define OOB_CHECK_INT(name, val, upper_bound) if(val >= upper_bound || val < 0){printf("OutOfBounds: [%s] %d >= %d  : %s:%d:%s()\n", name, val, upper_bound, __FILE__, __LINE__, __func__);}
#define OOB_CHECK(name, val, upper_bound) if(val >= upper_bound){printf("OutOfBounds: [%s] %u >= %u  : %s:%d:%s()\n", name, val, upper_bound, __FILE__, __LINE__, __func__);}
#else
/* Don't do anything in release builds */
#define OOB_CHECK_INT(name, val, upper_bound)
#define OOB_CHECK(name, val, upper_bound)
#endif

namespace cg = cooperative_groups;

//const __device__ float offset_val = 1.0f/64.0f;
//const __device__ float offset_val_n = -1.0f/64.0f;
const __device__ float offset_val = 1.0f/1.0f;
const __device__ float offset_val_n = (-1.0f) * offset_val;
const __device__ float offset_valc = 2000.0f/1.0f; //offset value for center offset

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
static __forceinline__ __device__ RN_T generate_rand(curandState_t* states, unsigned int id) {
    /* Copy state to local memory for efficiency */
    curandState_t localState = states[id];
    /* curand works like rand - except that it takes a state as a parameter */
    RN_T rnd = curand_uniform(&localState);

    /* Copy state back to global memory */
    states[id] = localState;

    return rnd;
}

static __forceinline__ __device__
unsigned int getWorkIndex()
{
    const unsigned int ix = optixGetLaunchIndex().x;
    const unsigned int iy = optixGetLaunchIndex().y;
    const unsigned int fbIndex = ix + iy *optixGetLaunchDimensions().x;
    //const unsigned int fbIndex = ix+iy*optixLaunchParams.simConstants.size.x;
    //const unsigned int fbIndex = blockDim.x * blockIdx.x + threadIdx.x;
    return fbIndex;
}

static __forceinline__ __device__
void *unpackPointer( unsigned int i0, unsigned int i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

constexpr __device__ float origin()      { return 1.0f / 32.0f; }
constexpr __device__ float float_scale() { return 1.0f / 65536.0f; }
constexpr __device__ float int_scale()   { return 256.0f; }
// Normal points outward for rays exiting the surface, else is flipped.
static __forceinline__ __device__ float3 offset_ray(const float3 p, const float3 n){
    int3 of_i(make_int3(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z));
    float3 p_i(make_float3(
            int_as_float(float_as_int(p.x)+((p.x < 0) ? -of_i.x : of_i.x)),
            int_as_float(float_as_int(p.y)+((p.y < 0) ? -of_i.y : of_i.y)),
            int_as_float(float_as_int(p.z)+((p.z < 0) ? -of_i.z : of_i.z))));
    return float3(make_float3(
            fabsf(p.x) < origin() ? p.x+float_scale()*n.x : p_i.x,
            fabsf(p.y) < origin() ? p.y+float_scale()*n.y : p_i.y,
            fabsf(p.z) < origin() ? p.z+float_scale()*n.z : p_i.z));
}

static __forceinline__ __device__ float3 offset_ray_new(const float3 p, const float3 n){
    int3 of_i(make_int3(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z));
    float3 p_i(make_float3(
            int_as_float(float_as_int(p.x)+((p.x < 0) ? -of_i.x : of_i.x)),
            int_as_float(float_as_int(p.y)+((p.y < 0) ? -of_i.y : of_i.y)),
            int_as_float(float_as_int(p.z)+((p.z < 0) ? -of_i.z : of_i.z))));
    return float3(make_float3(p.x+float_scale()*n.x,
                              p.y+float_scale()*n.y,
                              p.z+float_scale()*n.z));
}

// Normal points outward for rays exiting the surface, else is flipped.
static __forceinline__ __device__ float3 offset_ray_none(const float3 p, const float3 n){
    int3 of_i(make_int3(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z));
    float3 p_i(make_float3(
            int_as_float(float_as_int(p.x)+((p.x < 0) ? -of_i.x : of_i.x)),
            int_as_float(float_as_int(p.y)+((p.y < 0) ? -of_i.y : of_i.y)),
            int_as_float(float_as_int(p.z)+((p.z < 0) ? -of_i.z : of_i.z))));
    return p;
}
static __forceinline__ __device__ float3 offset_to_center(const float3 p, unsigned int prim_idx, const flowgpu::Polygon& poly){

    // get triangle center
    //float3 center;
    const flowgpu::TriangleMeshSBTData &sbtData = *(const flowgpu::TriangleMeshSBTData*)optixGetSbtDataPointer();

    const float3 &c = poly.center;

    /*DEBUG_PRINT("NEW OFFSET[%d -> %d -> %d] "
                "(%12.10f , %12.10f , %12.10f) \n",
                idx0, idx1, idx2, Aa.x, Aa.y, Aa.z);*/

    //int3 of_i(make_int3(int_scale() * c.x, int_scale() * c.y, int_scale() * c.z));
    //const int3 of_i = make_float3(float_scale()*c.x, float_scale()*c.y, float_scale()*c.z);
    float3 dir_i = normalize(c - p);
    int3 of_i(make_int3(
            int_scale() * dir_i.x * offset_valc,
            int_scale() * dir_i.y * offset_valc,
            int_scale() * dir_i.z * offset_valc));
    float3 p_i(make_float3(
            int_as_float(float_as_int(p.x)+((p.x < 0) ? -of_i.x : of_i.x)),
            int_as_float(float_as_int(p.y)+((p.y < 0) ? -of_i.y : of_i.y)),
            int_as_float(float_as_int(p.z)+((p.z < 0) ? -of_i.z : of_i.z))));
    /*DEBUG_PRINT("NEW OFFSET (%12.10f , %12.10f , %12.1    0f) ->"
                "(%12.10f , %12.10f , %12.10f) \n",
                p.x,p.y,p.z,dir_i.x, dir_i.y, dir_i.z);*/
    return float3(make_float3(p.x+1.0f*float_scale()*dir_i.x * offset_valc,
                              p.y+1.0f*float_scale()*dir_i.y * offset_valc,
                              p.z+1.0f*float_scale()*dir_i.z * offset_valc));
    return float3(make_float3(
            fabsf(p.x) < origin() ? p.x+float_scale()*dir_i.x : p_i.x,
            fabsf(p.y) < origin() ? p.y+float_scale()*dir_i.y : p_i.y,
            fabsf(p.z) < origin() ? p.z+float_scale()*dir_i.z : p_i.z));

    return float3(make_float3(p.x+float_scale()*c.x, p.y+float_scale()*c.y, p.z+float_scale()*c.z));
}

// Normal points outward for rays exiting the surface, else is flipped.
static __forceinline__ __device__ void initParticle(flowgpu::MolPRD& prd){
    prd.velocity = -999.0;
    prd.hitPos = make_float3(-999.0);
    prd.postHitDir = make_float3(-999.0);
    prd.hitFacetId = 9999999;
    prd.hitT = -999.0f;
    prd.inSystem = flowgpu::NEW_PARTICLE;
#if defined(GPUNBOUNCE)
    prd.nbBounces = 0;
#endif

}

static __forceinline__ __device__ void apply_offset(const flowgpu::MolPRD& hitData, float3& rayOrigin){
#ifdef WITHTRIANGLES
    const flowgpu::TriangleRayGenData* rayGenData = (flowgpu::TriangleRayGenData*) optixGetSbtDataPointer();
#else
    const flowgpu::PolygonRayGenData* rayGenData = (flowgpu::PolygonRayGenData*) optixGetSbtDataPointer();
#endif

    const uint32_t facIndex = hitData.hitFacetId;
#ifdef BOUND_CHECK
    if(facIndex >= optixLaunchParams.simConstants.nbFacets){
                printf("[RayOffset] facIndex %u >= %u is out of bounds (%u)\n", facIndex, optixLaunchParams.simConstants.nbFacets, hitData.inSystem);
            }
#endif
    //do not offset a transparent hit
    //if(optixLaunchParams.perThreadData.currentMoleculeData[bufferIndex].inSystem != TRANSPARENT_HIT){
    float3 facNormal = rayGenData->poly[facIndex].N;
    /*if((hitData.facetHitSide == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE && hitData.inSystem != TRANSPARENT_HIT)
       || (hitData.facetHitSide == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE && hitData.inSystem == TRANSPARENT_HIT))*/

    // Don't flip normal on a normal backface hit (should only be allowed for 2sided transparent facets)
    if((/*rayGenData->poly[hitData.hitFacetId].facProps.is2sided && */rayGenData->poly[hitData.hitFacetId].facProps.opacity == 0.0f)
       && (hitData.facetHitSide == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE) )

        // previous backface hit
    {
        /*if(bufferIndex == 0)
            printf("[%d] reverting offset -> %d -> %d\n",bufferIndex,hitData.inSystem, hitData.nbBounces);
        */facNormal *= (offset_val_n);
        //rayOrigin = offset_ray(rayOrigin, (-1.0f) * rayGenData->poly[facIndex].N);
    }
    else{
        facNormal *= (offset_val);
    }
    rayOrigin = offset_ray(rayOrigin,facNormal);
}

__device__
int atomicAggInc(int *ptr, int incVal)
{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;

    // elect the first active thread to perform atomic add
    if (g.thread_rank() == 0) {
        prev = atomicAdd(ptr, g.size() * incVal);
    }

    // broadcast previous value within the warp
    // and add each active thread’s rank to it
    prev = g.shfl(prev, 0);
    return g.thread_rank() + prev;
}

__device__
uint32_t atomicAggInc(uint32_t *ptr, uint32_t incVal)
{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;

    // elect the first active thread to perform atomic add
    if (g.thread_rank() == 0) {
        prev = atomicAdd(ptr, g.size() * incVal);
    }

    // broadcast previous value within the warp
    // and add each active thread’s rank to it
    prev = g.shfl(prev, 0);
    return g.thread_rank() + prev;
}

__device__
float atomicAggInc(float *ptr, float incVal)
{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;

    // elect the first active thread to perform atomic add
    if (g.thread_rank() == 0) {
        prev = atomicAdd(ptr, g.size() * incVal);
    }

    // broadcast previous value within the warp
    // and add each active thread’s rank to it
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}

static __forceinline__ __device__
double3 getOrigin_double(
#ifdef WITHTRIANGLES
        const flowgpu::TriangleRayGenData* rayGenData,
#else
                    const flowgpu::PolygonRayGenData* rayGenData,
#endif
                    const int& facIndex,
                    const double rnd1, const double rnd2)
{

#ifdef WITHTRIANGLES

    double r1_sqrt =
                #ifdef RNG64
                    sqrt((double)rnd1);
                #else
                    sqrtf(rnd1);
                #endif

        double r2 = rnd2;

        double3 vertA = rayGenData->vertex64[rayGenData->index[facIndex].x];
        double3 vertB = rayGenData->vertex64[rayGenData->index[facIndex].y];
        double3 vertC = rayGenData->vertex64[rayGenData->index[facIndex].z];

        double3 ret = (1.0 - r1_sqrt) * vertA + r1_sqrt * (1.0 - r2) * vertB + r1_sqrt * r2 * vertC;
        return ret; //rayGenData

#else //WITHTRIANGLES

    // start position of particle (U,V) -> (x,y,z)
    double uDir = rnd1, vDir = rnd2;

    return rayGenData->poly[facIndex].Ox64 + uDir * rayGenData->poly[facIndex].Ux64 + vDir * rayGenData->poly[facIndex].Vx64;
#endif //WITHTRIANGLES
    }

//TODO: Only non maxwell for now
static __forceinline__ __device__
FLOAT_T getNewVelocity(const flowgpu::Polygon& poly, const float& gasMass)
{
    return 145.469*sqrt((double)poly.facProps.temperature / gasMass);
}

//TODO: Only cosine for now
static __forceinline__ __device__
float3 getNewDirection(flowgpu::MolPRD& hitData, const flowgpu::Polygon& poly,
#ifdef RNG_BULKED
                       const RN_T* randFloat, unsigned int& randInd, unsigned int& randOffset)
#else
                        curandState_t* states)
#endif
{

    // generate ray direction
/*#ifdef BOUND_CHECK
    if(randInd + randOffset < 0 || randInd + randOffset >= optixLaunchParams.simConstants.nbRandNumbersPerThread*optixLaunchParams.simConstants.size.x*optixLaunchParams.simConstants.size.y){
            printf("randInd %u is out of bounds\n", randInd + randOffset);
        }
#endif*/


#ifdef RNG_BULKED
    hitData.rndDirection[0] = randFloat[(unsigned int) (randInd + randOffset++)];
    hitData.rndDirection[1] = randFloat[(unsigned int) (randInd + randOffset++)];
#else
    hitData.rndDirection[0] = generate_rand(states, getWorkIndex());
    hitData.rndDirection[1] = generate_rand(states, getWorkIndex());
#endif


    FLOAT_T theta = 0.0f;
#ifdef RNG64
    if(poly.desProps.desorbType == 2 || poly.desProps.desorbType == 0) {
        theta = acos(sqrt((double) hitData.rndDirection[0]));
    }
    else if (poly.desProps.desorbType == 3) {
        theta = acos(pow((double) hitData.rndDirection[0], (1.0 / (poly.desProps.cosineExponent + 1.0))));
    }
    else if (poly.desProps.desorbType == 1) {
        theta = acos((double) hitData.rndDirection[0]);
    }
#else
    if(poly.desProps.desorbType == 2 || poly.desProps.desorbType == 0) {
        theta = acosf(sqrtf(hitData.rndDirection[0]));
    }
    else if (poly.desProps.desorbType == 3) {
        theta = acosf(powf(hitData.rndDirection[0], (1.0f / (poly.desProps.cosineExponent + 1.0f))));
    }
    else if (poly.desProps.desorbType == 1) {
        theta = acosf(hitData.rndDirection[0]);
    }
#endif
    else{
        printf("Unsupported desorption type! %u on %d\n", poly.desProps.desorbType, poly.parentIndex);
        return make_float3(0.0f);
    }

    const FLOAT_T phi = hitData.rndDirection[1] *
#ifdef HIT64
                        2.0 * CUDART_PI;
#else
                        2.0f * CUDART_PI_F;
#endif

#ifdef RNG64
    const FLOAT_T u = sin(theta)*cos(phi);
    const FLOAT_T v = sin(theta)*sin(phi);
    const FLOAT_T n = cos(theta);
#else
    const float u = sinf(theta)*cosf(phi);
    const float v = sinf(theta)*sinf(phi);
    const float n = cosf(theta);
#endif


    /*float3 nU = rayGenData->poly[facIndex].nU;
    float3 nV = rayGenData->poly[facIndex].nV;
    float3 N = rayGenData->poly[facIndex].N;*/

#ifdef RNG64
    const double3 nU = make_double3(poly.nU);
    const double3 nV = make_double3(poly.nV);
    const double3 N = make_double3(poly.N);
    double3 ret = (u*nU + v*nV + n*N);
    return make_float3(u*nU + v*nV + n*N);
#else
    const float3 nU = poly.nU;
    const float3 nV = poly.nV;
    const float3 N = poly.N;

    return u*nU + v*nV + n*N;
#endif
    /*if (rayDir.x != 0.0) rayDir.x = 1.0 / rayDir.x;
    if (rayDir.y != 0.0) rayDir.y = 1.0 / rayDir.y;
    if (rayDir.z != 0.0) rayDir.z = 1.0 / rayDir.z;
    rayDir = float3(-1.0f,-1.0f,-1.0f) * rayDir;*/
}

static __forceinline__ __device__
double3 getDirection_double(flowgpu::MolPRD& hitData, const flowgpu::Polygon& poly,
                       const double rnd1, const double rnd2)
{
    double theta = 0.0;
    if(poly.desProps.desorbType == 2 || poly.desProps.desorbType == 0) {
        theta = acos(sqrt(rnd1));
    }
    else if (poly.desProps.desorbType == 3) {
        theta = acos(pow(rnd1, (1.0 / (poly.desProps.cosineExponent + 1.0))));
    }
    else if (poly.desProps.desorbType == 1) {
        theta = acos(rnd1);
    }
    else{
        printf("Unsupported desorption type! %u on %d , %d\n", poly.desProps.desorbType, hitData.hitFacetId, poly.parentIndex);
        return make_double3(0.0,0.0,0.0);
    }
    const double phi = rnd2 * 2.0 * CUDART_PI;

    const double u = sin(theta)*cos(phi);
    const double v = sin(theta)*sin(phi);
    const double n = cos(theta);

    const double3 nU = poly.nUx64;
    const double3 nV = poly.nVx64;
    const double3 N = poly.Nx64;

    //DEBUG_PRINT("Double Dir: %lf * [%lf,%lf,%lf] + %lf * [%lf,%lf,%lf] + %lf * [%lf,%lf,%lf]\n", u,nU.x , nU.y , nU.z ,v,nV.x , nV.y , nV.z , n,N.x , N.y , N.z);
    return u*nU + v*nV + n*N;
}

//TODO: Only cosine for now
static __forceinline__ __device__
float3 getNewReverseDirection(flowgpu::MolPRD& hitData, const flowgpu::Polygon& poly,
#ifdef RNG_BULKED
                              const RN_T* randFloat, unsigned int& randInd, unsigned int& randOffset)
#else
                              curandState_t* states)
#endif
{

    // generate ray direction
    FLOAT_T theta = 0.0;
#ifdef RNG64
    if(poly.desProps.desorbType == 2 || poly.desProps.desorbType == 0) {
        theta = acos(sqrt(hitData.rndDirection[0]));
    }
    else if (poly.desProps.desorbType == 3) {
        theta = acos(pow(hitData.rndDirection[0], (1.0 / (poly.desProps.cosineExponent + 1.0))));
    }
    else if (poly.desProps.desorbType == 1) {
        theta = acos(hitData.rndDirection[0]);
    }
#else
    if(poly.desProps.desorbType == 2 || poly.desProps.desorbType == 0) {
        theta = acosf(sqrtf((double) hitData.rndDirection[0]));
    }
    else if (poly.desProps.desorbType == 3) {
        theta = acosf(powf((double) hitData.rndDirection[0], (1.0f / (poly.desProps.cosineExponent + 1.0f))));
    }
    else if (poly.desProps.desorbType == 1) {
        theta = acosf((double) hitData.rndDirection[0]);
    }
#endif
    else{
        printf("Unsupported desorption type! %u on %d\n", poly.desProps.desorbType, poly.parentIndex);
        return make_float3(0.0f);
    }
    const FLOAT_T phi = hitData.rndDirection[1] *
#ifdef HIT64
            2.0 * CUDART_PI;
#else
            2.0f * CUDART_PI_F;
#endif

#ifdef RNG64
    const float u = sin(theta)*cos(phi);
    const float v = sin(theta)*sin(phi);
    const float n = cos(theta);
#else
    const float u = sinf(theta)*cosf(phi);
    const float v = sinf(theta)*sinf(phi);
    const float n = cosf(theta);
#endif
    const float3 nU = poly.nU;
    const float3 nV = poly.nV;
    const float3 N = poly.N; // reverse normal

    return u*nU + v*nV - n*N;
}

// Transform barycentrics to UV coordinates via texture coordinates
static __forceinline__ __device__
float2 getHitLocation(const float2 barycentrics, float2* texCoord, unsigned int texIndex)
{
    const float u = barycentrics.x; // beta and gamma
    const float v = barycentrics.y; // and gamma
    const float w = 1.0f - u - v; // and alpha

    float2 tex = w * texCoord[texIndex]
                 +         u * texCoord[texIndex+1]
                 +         v * texCoord[texIndex+2];

    return tex; //hitLocationU , hitLocationV
}

// Transformation to UV coordinates via Cramer's rule
// without direction info
static __forceinline__ __device__
float2 getHitLocation_old(const flowgpu::Polygon& poly, const float3& rayOrigin)
{
    const float3 b = rayOrigin - poly.O;

    float det = poly.U.x * poly.V.y - poly.U.y * poly.V.x; // Could be precalculated
    float detU = b.x * poly.V.y - b.y * poly.V.x;
    float detV = poly.U.x * b.y - poly.U.y * b.x;

    if(fabsf(det)<=EPS32){
        det = poly.U.y * poly.V.z - poly.U.z * poly.V.y;
        detU = b.y * poly.V.z - b.z * poly.V.y;
        detV = poly.U.y * b.z - poly.U.z * b.y;
        if(fabsf(det)<=EPS32){
            det = poly.U.z * poly.V.x - poly.U.x * poly.V.z;
            detU = b.z * poly.V.x - b.x * poly.V.z;
            detV = poly.U.z * b.x - poly.U.x * b.z;
#ifdef DEBUG
            if(fabsf(det)<=EPS32){
                    printf("[HitLoc] Dangerous determinant calculated: %e : %e : %e -> %e : %e\n",det,detU,detV,detU/det,detV/det);
            }
#endif
        }
    }

    return make_float2(detU/det, detV/det); //hitLocationU , hitLocationV
}

#endif //MOLFLOW_PROJ_COMMONFUNCTIONS_CUH
