//
// Created by pbahr on 04/11/2019.
//

#include <cuda.h>
#include "cudaRandom.cuh"
#include <math_constants.h>
#include <time.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include "GPUDefines.h" // for NB_RAND

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

    /* we have to initialize the state */
    const int id = threadIdx.x + blockIdx.x * 1;
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                id, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[id]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms_bits(curandState_t* states, unsigned int* numbers) {
    /* curand works like rand - except that it takes a state as a parameter */
    numbers[blockIdx.x] = curand(&states[blockIdx.x]) % 100;
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void random_floats(curandState_t* states, RN_T* numbers) {
    const int id = threadIdx.x + blockIdx.x * 1;
    /* Copy state to local memory for efficiency */
    curandState localState = states[id];
    /* curand works like rand - except that it takes a state as a parameter */
    for(int offset = 0; offset < NB_RAND; offset++)
        numbers[id + offset] = curand_uniform(&localState);

    /* Copy state back to global memory */
    states[id] = localState;
}

namespace crng {
/*
    cudaRandom::cudaRandom(unsigned int N) : kernelSize(N) {
        initializeRand();
    };

    cudaRandom::~cudaRandom() {
        */
/* free the memory we allocated for the states and numbers *//*

        cudaFree(states);
        cudaFree(randomNumbers);
    };
*/
    curandGenerator_t gen;
    int initializeRandHost(unsigned int kernelSize, RN_T **randomNumbersPtr, unsigned int seed) {
        //curandGenerator_t gen;
        /*float *//**devData,*//* *hostData;

        *//* Allocate n floats on host *//*
        hostData = (float *)calloc(n, sizeof(float));*/

        /*size_t available, total;
        cudaMemGetInfo(&available, &total);
        printf("Pre Available %d / %d\n",available,total);
        printf("Trying to allocate %d (Bytes)\n",NB_RAND*kernelSize*sizeof(float));*/

#ifdef RNG64
        printf("Generating 64bit RNG\n");
#else
        printf("Generating 32bit RNG\n");
#endif

        /* Allocate n floats on device */
        CUDA_CALL(cudaMalloc((void **)randomNumbersPtr, NB_RAND*kernelSize*sizeof(RN_T)));
        //printf("Allocating size for %d random number\n",NB_RAND*kernelSize);

        /* Create pseudo-random number generator */
#ifdef DEBUG
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
#else
        CURAND_CALL(curandCreateGenerator(&gen, (curandRngType)161));
#endif

        /* Set seed */
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));

        return EXIT_SUCCESS;
    }

    int generateRandHost(unsigned int kernelSize, RN_T *randomNumbers){
        /* Generate n floats on device */
#ifdef RNG64
        CURAND_CALL(curandGenerateUniformDouble(gen, randomNumbers, NB_RAND*kernelSize));
#else
        CURAND_CALL(curandGenerateUniform(gen, randomNumbers, NB_RAND*kernelSize));
#endif

        //CURAND_CALL(curandGenerate(gen, (unsigned int*)randomNumbers, NB_RAND*kernelSize));

        return EXIT_SUCCESS;
    }

    int destroyRandHost(RN_T **randomNumbersPtr){
        /* Cleanup */
        CURAND_CALL(curandDestroyGenerator(gen));
        CUDA_CALL(cudaFree(*randomNumbersPtr));
        return EXIT_SUCCESS;
    }

    int offsetBufferZeroInit(unsigned int kernelSize, void *randomOffsets){
        /* Generate n floats on device */
        CUDA_CALL(cudaMemset((unsigned int*)randomOffsets, 0, kernelSize*sizeof(unsigned int)));
        return EXIT_SUCCESS;
    }

    int testRand(void** devData, size_t n){
        //size_t n = 100;
        curandGenerator_t gen;
        /*float *//**devData,*//* *hostData;

        *//* Allocate n floats on host *//*
        hostData = (float *)calloc(n, sizeof(float));*/

        /* Allocate n floats on device */
        CUDA_CALL(cudaMalloc((void **)devData, n*sizeof(RN_T)));

        /* Create pseudo-random number generator */
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        /* Set seed */
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                                                       1234ULL));

        /* Generate n floats on device */
#ifdef RNG64
        CURAND_CALL(curandGenerateUniformDouble(gen, (RN_T*)*devData, n));
#else
        CURAND_CALL(curandGenerateUniform(gen, (RN_T*)*devData, n));
#endif
        /* Copy device memory to host *//*
        CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
                             cudaMemcpyDeviceToHost));

        *//* Show result *//*
        for(i = 0; i < n; i++) {
            if(i%100==0)
            printf("[%d] -- %1.4f ", i, hostData[i]);
        }
        printf("\n");*/

        /* Cleanup */
        CURAND_CALL(curandDestroyGenerator(gen));
        //CUDA_CALL(cudaFree(devData));
/*
        free(hostData);
*/


        return EXIT_SUCCESS;
    }

    int printDevDataAtHost(void* devData, size_t n){
        RN_T *hostData;

        /* Allocate n floats on host */
        hostData = (RN_T *)calloc(n, sizeof(RN_T));

        /* Copy device memory to host */
        CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(RN_T), cudaMemcpyDeviceToHost));

        /* Show result */
        unsigned int countSub = 0;
        unsigned int countSuper = 0;

        size_t i;
        for(i = 0; i < n; i++) {
            //if(i%100==0)
                //printf("[%d] -- %1.4f ", i, hostData[i]);
            if(hostData[i]==1.0f){
                printf("[%zd] -- %1.4f \n", i, hostData[i]);}
            else if(hostData[i]==0.0f)
                printf("[%zd] -- %1.4f \n", i, hostData[i]);
            else if(hostData[i]==0.25f)
                printf("[%zd] -- %1.4f \n", i, hostData[i]);
        }
        printf("\n");
        printf("[%6.4f/%d -- %6.4f/%d] ", (float)(countSub)/(countSub+countSuper),countSub, (float)(countSuper)/(countSub+countSuper),countSuper);
        printf("\n");
        free(hostData);
        return EXIT_SUCCESS;
    }

    int  initializeRand(unsigned int kernelSize, void* states, void* randomNumbers) {
        /* CUDA's random number library uses curandState_t to keep track of the seed value
         we will store a random state for every thread  */

        /* allocate space on the GPU for the random states */
        CUDA_CALL(cudaMalloc((void **) &states, kernelSize * sizeof(curandState_t)));

        unsigned int seed = time(0);
        /* we have to initialize the state */
        init << < kernelSize, 1 >> > (seed, (curandState_t*) states);

        CUDA_CALL(cudaMalloc((void **) &randomNumbers, NB_RAND * kernelSize * sizeof(RN_T))); // 10 rand per thread
        /* Set results to 0 */
        CUDA_CALL(cudaMemset((float *) randomNumbers, 0.0f, NB_RAND * kernelSize *sizeof(RN_T)));

        return EXIT_SUCCESS;
    }

    int  initializeRand(unsigned int kernelSize, curandState_t *states, RN_T *randomNumbers) {
        /* CUDA's random number library uses curandState_t to keep track of the seed value
         we will store a random state for every thread  */

        /* allocate space on the GPU for the random states */
        CUDA_CALL(cudaMalloc((void **) &states, kernelSize * sizeof(curandState_t)));

        unsigned int seed = time(0);
        /* we have to initialize the state */
        init << < kernelSize, 1 >> > (seed, states);

        CUDA_CALL(cudaMalloc((void **) &randomNumbers, NB_RAND * kernelSize * sizeof(RN_T))); // 10 rand per thread
        /* Set results to 0 */
        CUDA_CALL(cudaMemset(randomNumbers, 0.0f, NB_RAND * kernelSize *sizeof(RN_T)));

        return EXIT_SUCCESS;
    }

    void generateRand(unsigned int kernelSize, curandState_t *states, RN_T *randomNumbers) {
        /* invoke the kernel to get some random numbers */
        random_floats << < kernelSize, 1 >> > (states, randomNumbers);
    }

    void destroyRand(curandState_t *states, RN_T *randomNumbers){
        cudaFree(states);
        cudaFree(randomNumbers);
    }
}