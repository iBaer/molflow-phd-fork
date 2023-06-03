/*
Program:     MolFlow+ / Synrad+
Description: Monte Carlo simulator for ultra-high vacuum and synchrotron radiation
Authors:     Jean-Luc PONS / Roberto KERSEVAN / Marton ADY / Pascal BAEHR
Copyright:   E.S.R.F / CERN
Website:     https://cern.ch/molflow

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Full license text: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
*/

// common MF helper tools
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <memory>

#include "optix7.h"
#include "SimulationControllerGPU.h"
#include "helper_output.h"
#include "helper_math.h"
#include "GPUDefines.h"
#include "fmt/core.h"

#include "Model.h"
#include "HostData.h"
#include "SimulationOptiX.h"
#include "ModelReader.h"
#include "Helper/OutputHelper.h"
#include "Helper/Chronometer.h"
#include "GPUSettings.h"
#include <omp.h>

SimulationControllerGPU::SimulationControllerGPU(size_t parentPID, size_t procIdx, size_t nbThreads,
                                                 SimulationUnit *simUnit,
                                                 std::shared_ptr<ProcComm> pInfo)
        : SimulationController(parentPID, procIdx, nbThreads, nullptr, pInfo), /*data(), globalCounter(), */figures(),
          globFigures() {
    optixHandle = nullptr;
    model = nullptr;
    hasEnded = false;
    endCalled = false;

    simulation = simUnit;
    data = std::make_unique<HostData>();
    globalCounter = std::make_unique<GlobalCounter>();
}

SimulationControllerGPU::~SimulationControllerGPU() {
    if (optixHandle) {
        CloseSimulation();
    }
    /*if(model){
        delete model;
    }*/
    // model is deleted from SimulationGPU
}

// Get Simulation Data from device (+reset) and add to global simulation state
bool UpdateHits(SimulationControllerGPU& sim_con, GlobalSimuState& globState) {
    uint64_t nbHits = sim_con.GetSimulationData();
    // Export results
    // 1st convert from GPU types to CPU types
    // 2nd save with XML
    sim_con.ConvertSimulationData(globState);

    sim_con.ResetGlobalCounter();
    return true;
}

bool SimulationControllerGPU::runLoop() {
    bool eos;
    bool lastUpdateOk = false;

    //printf("Lim[%zu] %lu --> %lu\n",threadNum, localDesLimit, simulation->globState->globalHits.globalHits.hit.nbDesorbed);

    Chronometer run_chrono;
    run_chrono.Start();
    double timeStart = run_chrono.ElapsedMs();
    double timeLoopStart = timeStart;
    double timeEnd;

    bool simEos = false;

    // TODO: Remove defaults
    std::list<size_t> limits; // Number of desorptions: empty=use other end condition
    size_t nbLoops = 0;               // Number of Simulation loops
    size_t launchSize = 1;                  // Kernel launch size
    size_t nPrints = 10;
    size_t printPerN = 100000;
    double angle = 0.0; // timeLimit / -i or direct by -k
    double printEveryNMinutes = 0.0; // timeLimit / -i or direct by -k
    double timeLimit = 0.0;
    bool silentMode = false;
    constexpr double runForMS = 1000.0;
    uint64_t printPerNRuns = std::min(static_cast<uint64_t>(printPerN), static_cast<uint64_t>(nbLoops/nPrints)); // prevent n%0 operation
    printPerNRuns = std::max(printPerNRuns, static_cast<uint64_t>(1));

    // -i
    if(nbLoops==0 && printEveryNMinutes <= 0.0 && timeLimit > 1e-6){
        printEveryNMinutes = timeLimit / nPrints;
    }
    // ^^^

    auto start_total = std::chrono::steady_clock::now();
    auto t0 = start_total;
    double nextPrintAtMin = printEveryNMinutes;

    double raysPerSecondMax = 0.0;
    double desPerSecondMax = 0.0;

    size_t refreshForStop = std::numeric_limits<size_t>::max();
    size_t loopN = 0;

    nbLoops = 100000;
    printEveryNMinutes = 1.0/60.0;
    nextPrintAtMin = printEveryNMinutes;

    do {
        //setSimState(getSimStatus());
        //size_t desorptions = localDesLimit;//(localDesLimit > 0 && localDesLimit > particle->tmpState.globalHits.globalHits.hit.nbDesorbed) ? localDesLimit - particle->tmpState.globalHits.globalHits.hit.nbDesorbed : 0;

        /*simEos = */
        double t_run_start = run_chrono.ElapsedMs();
        do {
            RunSimulation();  // Run during 1 sec
            timeEnd = run_chrono.ElapsedMs();
            ++loopN;
        } while(timeEnd - t_run_start < runForMS && (model->ontheflyParams.desorptionLimit != 0 && refreshForStop >= loopN)); // run loop for a particular amount of time, to reduce overhead

        size_t timeOut = lastUpdateOk ? 0 : 100; //ms

        lastUpdateOk = UpdateHits(*this, *this->simulation->globState); // Update hit with 20ms timeout. If fails, probably an other subprocess is updating, so we'll keep calculating and try it later (latest when the simulation is stopped).

        size_t localDesLimit = 0;
        if(simulation->model->otfParams.desorptionLimit > 0){
            if(localDesLimit > simulation->globState->globalHits.globalHits.nbDesorbed)
                localDesLimit -= simulation->globState->globalHits.globalHits.nbDesorbed;
            else localDesLimit = 0;
        }
        if(model->ontheflyParams.desorptionLimit != 0) {
            // add remaining steps to current loop count, this is the new approx. stop until desorption limit is reached
            refreshForStop = globFigures.runCount + RemainingStepsUntilStop();
            Log::console_msg_master(3, " Stopping at {} / {} with {} x {} x {} des\n",
                                    globFigures.runCount, refreshForStop, globFigures.total_des, figures.total_des, simulation->globState->globalHits.globalHits.nbDesorbed);
        }

        timeLoopStart = run_chrono.ElapsedMs();

        //printf("[%zu] PUP: %lu , %lu , %lu\n",threadNum, desorptions,localDesLimit, particle->tmpState.globalHits.globalHits.hit.nbDesorbed);
        const bool eos_time = this->model->ontheflyParams.timeLimit != 0 &&
                              timeEnd - timeStart >= this->model->ontheflyParams.timeLimit * 1000;
        const bool eos_comm = (procInfo->masterCmd != COMMAND_START);
        const bool eos_des = this->model->ontheflyParams.desorptionLimit != 0 &&
                this->simulation->globState->globalHits.globalHits.nbDesorbed >= this->model->ontheflyParams.desorptionLimit * 1000;
        eos = simEos || eos_time || eos_comm || eos_des;
    } while (!eos);

    hasEnded = true;

    /*procInfo->RemoveAsActive(threadNum);
    if (!lastUpdateOk) {
        //printf("[%zu] Updating on finish!\n",threadNum);
        setSimState("Final update...");
        particle->UpdateHits(simulation->globState, simulation->globParticleLog,
                             20000); // Update hit with 20ms timeout. If fails, probably an other subprocess is updating, so we'll keep calculating and try it later (latest when the simulation is stopped).)
    }*/

    Log::console_msg_master(3, " EOS at {} / {} with {} x {} x {} des\n",
                            globFigures.runCount, refreshForStop, globFigures.total_des, figures.total_des, simulation->globState->globalHits.globalHits.nbDesorbed);

    return simEos;
}

/*auto t1 = std::chrono::steady_clock::now();
std::chrono::duration<double,std::ratio<60,1>> elapsedMinutes = t1 - start_total;

// Fetch end conditions
if(nbLoops > 0 && loopN >= nbLoops)
hasEnded = true;
else if(timeLimit >= 1e-6 && elapsedMinutes.count() >= timeLimit)
hasEnded = true;

if((!silentMode && ((printEveryNMinutes > 0.0 && elapsedMinutes.count() > nextPrintAtMin)) || refreshForStop <= loopN || hasEnded)){
if(printEveryNMinutes > 0.0 && elapsedMinutes.count() > nextPrintAtMin)
nextPrintAtMin += printEveryNMinutes;

auto t1 = std::chrono::steady_clock::now();
std::chrono::duration<double,std::milli> elapsed = t1 - t0;
t0 = t1;

uint64_t nbHits = GetSimulationData();
// Export results
// 1st convert from GPU types to CPU types
// 2nd save with XML
ConvertSimulationData(*this->simulation->globState);

// end on leak
*//*if(figures.total_leak)
break;//return figures.total_leak;*//*
if(model->ontheflyParams.desorptionLimit != 0) {
// add remaining steps to current loop count, this is the new approx. stop until desorption limit is reached
refreshForStop = loopN + RemainingStepsUntilStop();
std::cout << " Stopping at " << loopN << " / " << refreshForStop << std::endl;
}
if(hasEnded){
// if there is a next des limit, handle that
if(!limits.empty()) {
model->ontheflyParams.desorptionLimit = limits.front();
limits.pop_front();
hasEnded = false;
endCalled = false;
AllowNewParticles();
std::cout << " Handling next des limit " << model->ontheflyParams.desorptionLimit << std::endl;
}
}
static const uint64_t nRays = launchSize * printPerNRuns;
//double rpsRun = (double)nRays / elapsed.count() / 1000.0;
double rpsRun = (double)(nbHits) / elapsed.count() / 1000.0;
raysPerSecondMax = std::max(raysPerSecondMax,rpsRun);
//raysPerSecondSum += rpsRun;
std::cout << "--- Run #" << loopN + 1 << " \t- Elapsed Time: " << elapsed.count() / 1000.0 << " s \t--- " << rpsRun << " MRay/s ---" << std::endl;
printf("--- Trans Prob: %e\n",GetTransProb());
}

++loopN;*/

//! Staart a simulation
int SimulationControllerGPU::Start() {
    // Check simulation model and geometry one last time
    auto sane = simulation->SanityCheckModel(true);
    if(sane.first){
        loadOk = false;
    }

#if not defined(GPUCOMPABILITY)
    for(auto& thread : simThreads){
        if(!thread.particle)
            loadOk = false;
    }
#endif

    if(!loadOk) {
        if(sane.second)
            SetState(PROCESS_ERROR, sane.second->c_str());
        else
            SetState(PROCESS_ERROR, GetSimuStatus());
        return 1;
    }

#if not defined(GPUCOMPABILITY)
    if(simulation->model->accel.empty()){
    //if(RebuildAccel()){
        loadOk = false;
        SetState(PROCESS_ERROR, "Failed building acceleration structure!");
        return 1;
    }
#endif

    if (simulation->model->otfParams.desorptionLimit > 0) {
        if (simulation->totalDesorbed >=
            simulation->model->otfParams.desorptionLimit /
            simulation->model->otfParams.nbProcess) {
            ClearCommand();
            SetState(PROCESS_DONE, GetSimuStatus());
        }
    }

    if (GetLocalState() != PROCESS_RUN) {
        DEBUG_PRINT("[%d] COMMAND: START (%zd,%zu)\n", prIdx, procInfo->cmdParam, procInfo->cmdParam2);
        SetState(PROCESS_RUN, GetSimuStatus());
    }


    bool eos = false;
    bool lastUpdateOk = true;
    if (loadOk) {
        procInfo->InitActiveProcList();

        // Calculate remaining work
        size_t desPerThread = 0;
        size_t remainder = 0;
        if(simulation->model->otfParams.desorptionLimit > 0){
            if(simulation->model->otfParams.desorptionLimit > (simulation->globState->globalHits.globalHits.nbDesorbed)) {
                size_t limitDes_global = simulation->model->otfParams.desorptionLimit;
                desPerThread = limitDes_global / nbThreads;
                remainder = limitDes_global % nbThreads;
            }
        }

        runLoop();

        if (hasEnded) {
            if (GetLocalState() != PROCESS_ERROR) {
                // Max desorption reached
                ClearCommand();
                SetState(PROCESS_DONE, GetSimuStatus());
                DEBUG_PRINT("[%d] COMMAND: PROCESS_DONE (Max reached)\n", prIdx);
            }
        }
        else {
            if (simulation->model->otfParams.desorptionLimit > 0) {
                if (simulation->totalDesorbed >=
                    simulation->model->otfParams.desorptionLimit /
                    simulation->model->otfParams.nbProcess) {
                    ClearCommand();
                    SetState(PROCESS_DONE, GetSimuStatus());
                }
            }
        }
    } else {
        SetErrorSub("No geometry loaded");
        ClearCommand();
    }
    return 0;
}

// return true on error, false if load successful
bool SimulationControllerGPU::Load() {
    SetState(PROCESS_STARTING, "Loading simulation");

    auto sane = simulation->SanityCheckModel(false);
    if(!sane.first) {
        SetState(PROCESS_STARTING, "Loading simulation");
        bool loadError = false;
        auto &simModel = simulation->model;
        if(!simModel || flowgpu::loadFromSimModel(this->model, settings, *simModel)) {
            loadOk = false;
        }
        else {
            loadOk = !LoadSimulation(model, settings->kernelDimensions[0] * settings->kernelDimensions[1]);
        }
        Reset();
        SetRuntimeInfo();
    }
    SetReady(loadOk);

    return !loadOk;
}

//! Not supported, as ADS are not really customizable on GPU
int SimulationControllerGPU::RebuildAccel() {
    return 0;
}

//! Reset simulation
int SimulationControllerGPU::Reset() {
    DEBUG_PRINT("[%d] COMMAND: RESET (%zd,%zu)\n", prIdx, procInfo->cmdParam, procInfo->cmdParam2);
    SetState(PROCESS_STARTING, "Resetting local cache...", false, true);
    resetControls();
    ResetSimulation(true);
    SetReady(loadOk);
    return 0;
}

void SimulationControllerGPU::EmergencyExit() {}; // Killing threads

/**
 * Reset tmp results and load simulation
 * @return 1=could not load GPU Sim, 0=successfully loaded
 */
int SimulationControllerGPU::LoadSimulation(std::shared_ptr<flowgpu::Model> loaded_model, size_t launchSize) {
    if (loaded_model == nullptr)
        return 1;

    /*uint2 newSize = make_uint2(launchSize,1);
    if(newSize != kernelDimensions){
        kernelDimensions = newSize;
    }*/

    try {
        model = loaded_model;
        ResetSimulation(false);
        settings->kernelDimensions[0] = launchSize;
        settings->kernelDimensions[1] = 1;
        optixHandle = std::make_shared<flowgpu::SimulationOptiX>(loaded_model, settings);
    } catch (std::runtime_error &e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        std::cout << "Does GPUMolflow support this geometry yet?" << std::endl;
        exit(1);
    }

    optixHandle->initSimulation();
    Resize();

    return 0;
}

/**
 * Start one simulation circle
 * @return total desorbed particles
 */
uint64_t SimulationControllerGPU::RunSimulation() {

#ifdef RNG_BULKED
    // generate new numbers whenever necessary, recursion = TraceProcessing only, poly checks only for ray generation with polygons
    if (figures.runCount % (settings->cyclesRNG) == 0) {
#ifdef DEBUG
        //TODO: Print for certain verbosity levels
        // std::cout << "#flowgpu: generating random numbers at run #" << figures.runCount << std::endl;
#endif
        optixHandle->generateRand();
    }
#endif //RNG_BULKED

    try {
        optixHandle->launchMolecules();
        ++figures.runCount;
        ++globFigures.runCount;
        if (!endCalled && !hasEnded) {
            ++figures.runCountNoEnd;
            ++globFigures.runCountNoEnd;
        }
    } catch (std::runtime_error &e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return figures.total_des;
}

/**
 * Calculates rough estimate (N Steps) for when a desorption limit will be reached
 */
int SimulationControllerGPU::RemainingStepsUntilStop() {
    uint64_t diffDes = 0u;
    if (model->ontheflyParams.desorptionLimit > globFigures.total_des)
        diffDes = model->ontheflyParams.desorptionLimit - globFigures.total_des;
    size_t remainingDes = diffDes;

    // Minimum 100 steps, to not spam single steps on desorption stop
    // TODO : find more elegant way
    size_t remainingSteps = remainingDes / (settings->kernelDimensions[0] * settings->kernelDimensions[1]);
    if (diffDes > 0 && globFigures.desPerRun > 0)
        remainingSteps = std::ceil(0.9 * remainingDes / globFigures.desPerRun);
    if (endCalled) {
        // TODO: replace  kernelDimensions[0]*kernelDimensions[1]
        if (diffDes >= 0)
            remainingSteps = std::ceil(0.9 * settings->kernelDimensions[0] * settings->kernelDimensions[1] / figures.desPerRun_stop);
        if (remainingSteps < 100)
            remainingSteps = 100;
    }
    /*printf("[ %lf ] Remaining des: %zu (%zu - %llu) --> %lu\n",figures.desPerRun, remainingDes, model->ontheflyParams.desorptionLimit, figures.total_des,remainingSteps);
    std::cout << figures.desPerRun<< " ] Remaining des --> "<<remainingDes << " --> " << remainingSteps << std::endl;
    std::cout << figures.desPerRun_stop<< " ] Remaining des --> "<<remainingDes << " --> " << remainingSteps << std::endl;
*/
    return remainingSteps;
}

/**
 * Allow new particles, if new desorption limit (if any) is not yet reached
 */
void SimulationControllerGPU::AllowNewParticles() {
#ifdef WITHDESORPEXIT
    if (this->model->ontheflyParams.desorptionLimit > 0 &&
        figures.total_des >= this->model->ontheflyParams.desorptionLimit)
        return;
    optixHandle->downloadDataFromDevice(data.get()); //download tmp counters
    for (auto &particle: data->hitData) {
        particle.hasToTerminate = 0;
    }
    optixHandle->updateHostData(data.get());

    hasEnded = false;
#endif

    return;
}

/**
 * Stop new particles, e.g. if new desorption limit (if any) is reached
 */
void SimulationControllerGPU::StopNewParticles() {
#ifdef WITHDESORPEXIT
    if (this->model->ontheflyParams.desorptionLimit > 0 &&
        figures.total_des >= this->model->ontheflyParams.desorptionLimit)
        return;
    optixHandle->downloadDataFromDevice(data.get()); //download tmp counters
    for (auto &particle: data->hitData) {
        particle.hasToTerminate = 1;
    }
    optixHandle->updateHostData(data.get());

    hasEnded = false;
#endif

    return;
}

//! Check for desorption limit and block desorption of new particles
void SimulationControllerGPU::CheckAndBlockDesorption() {
#ifdef WITHDESORPEXIT
    if (this->model->ontheflyParams.desorptionLimit > 0) {
        if (figures.total_des >= this->model->ontheflyParams.desorptionLimit) {
            endCalled = false;
            size_t nbExit = 0;

            for (auto &particle: data->hitData) {
                if (particle.hasToTerminate > 0)
                    endCalled = true;
                else
                    particle.hasToTerminate = 1;
                if (endCalled && particle.hasToTerminate == 2)
                    nbExit++;
            }
            if (!endCalled) optixHandle->updateHostData(data.get());
            if (nbExit >= settings->kernelDimensions[0] * settings->kernelDimensions[1]) {
                std::cout << " READY TO EXIT! " << std::endl;
                hasEnded = true;
            }
        }
    }
#endif

    return;
}

static uint64_t prevExitCount = 0;

//! Check for desorption limit and block desorption of new particles, where a threshold (in%) can be specified to prevent long waiting times
// threshold == 1.0 : all particles have to be absorbed
void SimulationControllerGPU::CheckAndBlockDesorption_exact(double threshold) {
#ifdef WITHDESORPEXIT
    size_t nThreads = settings->kernelDimensions[0] * settings->kernelDimensions[1];
    if (this->model->ontheflyParams.desorptionLimit > 0) {
        if (globFigures.total_des + nThreads >= this->model->ontheflyParams.desorptionLimit) {
            size_t desToStop =
                    (model->ontheflyParams.desorptionLimit > globFigures.total_des) ?
                    (int64_t) model->ontheflyParams.desorptionLimit - globFigures.total_des : 0;
            //endCalled = false;
            size_t nbExit = 0;
            size_t nbTerm = 0;

            // 1. Set already terminated particles back to active
            // 2. set remaining active particles to terminate
            // that way there will never be inactive particles reaching full desorption limit
            int pInd = 0;
            auto desLim = desToStop;
            if (endCalled) {
                while (desLim > 0 && pInd < data->hitData.size()) {
                    auto &particle = data->hitData[pInd];
                    if (particle.hasToTerminate == 2) {
                        particle.hasToTerminate = 1;
                        --desLim;
                    } else if (particle.hasToTerminate == 1) {
                        ++nbTerm;
                    } else if (particle.hasToTerminate == 0) {
                        particle.hasToTerminate = 1;
                    }
                    ++pInd;
                }
            } else {
                for (auto &particle: data->hitData) {
                    particle.hasToTerminate = 1;
                }
                pInd = desToStop;
                endCalled = true;
            }

            // set remainders
            for (int p = pInd; p < data->hitData.size(); ++p) {
                auto &particle = data->hitData[p];
                if (particle.hasToTerminate == 0) {
                    particle.hasToTerminate = 1;
                    ++nbTerm;
                } else if (particle.hasToTerminate == 2) {
                    nbExit++;
                } else {
                    endCalled = true;
                }
            }

            if (nbExit) {
                globFigures.exitCount += nbExit - prevExitCount;
                prevExitCount = nbExit;
            }
            if (endCalled) optixHandle->updateHostData(data.get());
            if (nbExit >= nThreads * threshold) {
                prevExitCount = 0;
                std::cout << " READY TO EXIT! " << std::endl;
                hasEnded = true;
            }
            //printf("Block: %llu [%zu / %zu / %zu / %zu / %zu]\n", desToStop, nbTerm, nbExit, nThreads, data->hitData.size(), data->hitData.size() - desToStop);

        }
    }
#endif

    return;
}

/**
 * Calculates and returns transmission probability for a particular facet
 * @return transmission probability for facet with given index
 */
double SimulationControllerGPU::GetTransProb(size_t polyIndex) {
#if defined(WITHTRIANGLES)
    double sumAbs = 0;
    for (unsigned int i = 0; i < globalCounter->facetHitCounters.size(); i++) {
        unsigned int facIndex = i % this->model->nbFacets_total;
        unsigned int facParent = model->triangle_meshes[0]->poly[facIndex].parentIndex;
        if (facParent == polyIndex)
            sumAbs += this->globalCounter->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    return sumAbs / (double) figures.total_des;
#else
    double sumAbs = 0;
    for(unsigned int i = 0; i < globalCounter->facetHitCounters.size(); i++) {
        unsigned int facIndex = i%this->model->nbFacets_total;
        unsigned int facParent = model->poly_meshes[0]->poly[facIndex].parentIndex;
        if(facParent==polyIndex)
            sumAbs += this->globalCounter->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    return sumAbs / (double) figures.total_des;
#endif
}

//! handy function for quick compares that returns the transmission probability
double SimulationControllerGPU::GetTransProb() {
#if defined(WITHTRIANGLES)
    double sumAbs = 0;
    std::vector<double> sumAbs_all(this->model->nbFacets_total, 0.0);
    for (unsigned int i = 0; i < globalCounter->facetHitCounters.size(); i++) {
        unsigned int facIndex = i % this->model->nbFacets_total;
        if (model->triangle_meshes[0]->poly[facIndex].desProps.desorbType != 0)
            continue;
        unsigned int facParent = model->triangle_meshes[0]->poly[facIndex].parentIndex;
        sumAbs_all[facParent] += this->globalCounter->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    sumAbs = *std::max_element(sumAbs_all.begin(), sumAbs_all.end());
    return sumAbs / (double) figures.total_des;
#else
    double sumAbs = 0;
    for(unsigned int i = 0; i < globalCounter->facetHitCounters.size(); i++) {
        unsigned int facIndex = i%this->model->nbFacets_total;
        unsigned int facParent = model->poly_meshes[0]->poly[facIndex].parentIndex;
        if(facParent==i)
            sumAbs += this->globalCounter->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    return sumAbs / (double) figures.total_des;
#endif
}

//! Do various calculations for runtime statistics
void SimulationControllerGPU::CalcRuntimeFigures() {
    globFigures.runCount = figures.runCount;
    globFigures.runCountNoEnd = figures.runCountNoEnd;
    figures.desPerRun = (double) (figures.total_des - figures.ndes_stop) / figures.runCountNoEnd;
    globFigures.desPerRun = (double) (globFigures.total_des - globFigures.ndes_stop) / globFigures.runCountNoEnd;
    figures.desPerRun_stop = (double) (figures.exitCount) / (figures.runCount - figures.runCountNoEnd);
    /*printf(" DPR --> %lf [%llu / %u]\n", figures.desPerRun, figures.total_des - figures.ndes_stop, figures.runCountNoEnd);
    printf("gDPR --> %lf [%llu / %u]\n", globFigures.desPerRun, globFigures.total_des - globFigures.ndes_stop, globFigures.runCountNoEnd);
    printf("sDPR --> %lf [%llu / %u]\n", (double) figures.exitCount / (figures.runCount-figures.runCountNoEnd), figures.ndes_stop, figures.runCount-figures.runCountNoEnd);
*/
}

/**
 * Fetch simulation data from the device
 * @return 1=could not load GPU Sim, 0=successfully loaded
 */
unsigned long long int SimulationControllerGPU::GetSimulationData(bool silent) {

    bool writeData = false;
    bool printData = false & !silent;
    bool printDataParent = false & !silent;
    bool printCounters = false & !silent;
#ifdef WITHDESORPEXIT
    //printCounters = true;
#endif
    try {
        optixHandle->downloadDataFromDevice(data.get()); //download tmp counters
        IncreaseGlobalCounters(data.get()); //increase global counters
        UpdateGlobalFigures();

        if (printCounters) PrintTotalCounters();
        optixHandle->resetDeviceBuffers(); //reset tmp counters

        //CheckAndBlockDesorption();
        CheckAndBlockDesorption_exact(1.00);
        if (writeData) WriteDataToFile("hitcounters.txt");
        if (printData) PrintData();
        if (printDataParent) PrintDataForParent();
        CalcRuntimeFigures();

        return GetTotalHits();
    } catch (std::runtime_error &e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
}

//! Add up global GPU counters (on host) with downloaded data (temporary state)
void SimulationControllerGPU::IncreaseGlobalCounters(HostData *tempData) {
#ifdef DEBUGLEAKPOS
    {
        int nbPos = NBCOUNTS;
        const uint32_t nbLeaksMax = 1024;
        uint32_t curLeakPos = 0;
        const int hitPositionsPerMol = std::min(30, NBCOUNTS);
        for (int i = 0; i < tempData->leakPositions.size();) {
            bool begin = false;
            if (curLeakPos >= nbLeaksMax) break;
            for (int pos = 0; pos < hitPositionsPerMol; pos++) {
                size_t index = i / (NBCOUNTS) * NBCOUNTS + pos;
                if (tempData->leakPositions[index].x != 0.0f
                    || tempData->leakPositions[index].y != 0.0f
                    || tempData->leakPositions[index].z != 0.0f) {
                    if (curLeakPos < nbLeaksMax) {
                        this->globalCounter->leakPositions.emplace_back(tempData->leakPositions[index]);
                        this->globalCounter->leakDirections.emplace_back(tempData->leakDirections[index]);
                        curLeakPos++;
                    } else {
                        break;
                    }
                }
            }
            i += nbPos; // jump to next molecule/thread
        }
    }
#endif

#ifdef DEBUGPOS

    {
        uint32_t curPos = 0;
        for (auto &pos: tempData->positions) {
            if (tempData->positions[curPos].x != 0
                || tempData->positions[curPos].y != 0
                || tempData->positions[curPos].z != 0) {
                this->globalCounter->positions.emplace_back(tempData->positions[curPos]);
            }
            curPos++;
        }
    }
#endif

    //facet hit counters + miss
    for (unsigned int i = 0; i < data->facetHitCounters.size(); i++) {
        unsigned int facIndex = i % this->model->nbFacets_total;
        globalCounter->facetHitCounters[facIndex].nbMCHit += tempData->facetHitCounters[i].nbMCHit;
        globalCounter->facetHitCounters[facIndex].nbDesorbed += tempData->facetHitCounters[i].nbDesorbed;
        globalCounter->facetHitCounters[facIndex].nbAbsEquiv += tempData->facetHitCounters[i].nbAbsEquiv;

        globalCounter->facetHitCounters[facIndex].nbHitEquiv += tempData->facetHitCounters[i].nbHitEquiv;
        globalCounter->facetHitCounters[facIndex].sum_v_ort += tempData->facetHitCounters[i].sum_v_ort;
        globalCounter->facetHitCounters[facIndex].sum_1_per_velocity += tempData->facetHitCounters[i].sum_1_per_velocity;
        globalCounter->facetHitCounters[facIndex].sum_1_per_ort_velocity += tempData->facetHitCounters[i].sum_1_per_ort_velocity;
    }

    for (int i = 0; i < globalCounter->leakCounter.size(); ++i) {
        globalCounter->leakCounter[i] += tempData->leakCounter[i];
    }

    //textures
#ifdef WITH_TEX
    if (!tempData->texels.empty()) {
        for (auto &[id, texels]: globalCounter->textures) {
            // First check triangles
            for (auto &mesh: model->triangle_meshes) {
                int parentFacetId = -1;
                for (auto &facet: mesh->poly) {
                    if (parentFacetId == facet.parentIndex) break;
                    if ((facet.texProps.textureFlags) && (id == facet.parentIndex)) {
                        parentFacetId = id;
                        unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                        unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;
                        for (unsigned int h = 0; h < height; ++h) {
                            for (unsigned int w = 0; w < width; ++w) {
                                unsigned int index_glob =
                                        w + h * model->facetTex[facet.texProps.textureOffset].texWidth;
                                unsigned int index_tmp =
                                        index_glob + model->facetTex[facet.texProps.textureOffset].texelOffset;

                                texels[index_glob].countEquiv += tempData->texels[index_tmp].countEquiv;
                                texels[index_glob].sum_v_ort_per_area += tempData->texels[index_tmp].sum_v_ort_per_area;
                                texels[index_glob].sum_1_per_ort_velocity += tempData->texels[index_tmp].sum_1_per_ort_velocity;
                            }
                        }
                    }
                }
            }
            // Next check polygon
            for (auto &mesh: model->poly_meshes) {
                for (auto &facet: mesh->poly) {
                    if ((facet.texProps.textureFlags) && (id == facet.parentIndex)) {
                        unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                        unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;
                        for (unsigned int h = 0; h < height; ++h) {
                            for (unsigned int w = 0; w < width; ++w) {
                                unsigned int index_glob =
                                        w + h * model->facetTex[facet.texProps.textureOffset].texWidth;
                                unsigned int index_tmp =
                                        index_glob + model->facetTex[facet.texProps.textureOffset].texelOffset;

                                texels[index_glob].countEquiv += tempData->texels[index_tmp].countEquiv;
                                texels[index_glob].sum_v_ort_per_area += tempData->texels[index_tmp].sum_v_ort_per_area;
                                texels[index_glob].sum_1_per_ort_velocity += tempData->texels[index_tmp].sum_1_per_ort_velocity;
                            }
                        }
                    }
                }
            }
        }
    }
#endif // WITH_TEX
#ifdef WITH_PROF
    //profiles
    if (!tempData->profileSlices.empty()) {
        for (auto &[id, profiles]: globalCounter->profiles) {
            for (auto &mesh: model->triangle_meshes) {
                int parentFacetId = -1;
                for (auto &facet: mesh->poly) {
                    if (parentFacetId == facet.parentIndex) break;
                    if ((facet.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile) &&
                        (id == facet.parentIndex)) {
                        parentFacetId = id;
                        for (unsigned int s = 0; s < PROFILE_SIZE; ++s) {
                            unsigned int index_tmp = s + facet.profProps.profileOffset;

                            profiles[s].countEquiv += tempData->profileSlices[index_tmp].countEquiv;
                            profiles[s].sum_v_ort_per_area += tempData->profileSlices[index_tmp].sum_v_ort_per_area;
                            profiles[s].sum_1_per_ort_velocity += tempData->profileSlices[index_tmp].sum_1_per_ort_velocity;

                        }
                    }
                }
            }

            for (auto &mesh: model->poly_meshes) {
                for (auto &facet: mesh->poly) {
                    if ((facet.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile)) {
                        for (unsigned int s = 0; s < PROFILE_SIZE; ++s) {
                            unsigned int index_tmp = s + facet.profProps.profileOffset;

                            profiles[s].countEquiv += tempData->profileSlices[index_tmp].countEquiv;
                            profiles[s].sum_v_ort_per_area += tempData->profileSlices[index_tmp].sum_v_ort_per_area;
                            profiles[s].sum_1_per_ort_velocity += tempData->profileSlices[index_tmp].sum_1_per_ort_velocity;

                        }
                    }
                }
            }
        }
    }
#endif // WITH_PROF

}

//! Reset global GPU counters (on host)
int SimulationControllerGPU::ResetGlobalCounter() {
    globalCounter->facetHitCounters.clear();
    globalCounter->leakCounter.clear();
    globalCounter->textures.clear();
    globalCounter->profiles.clear();
#ifdef DEBUGPOS
    globalCounter->positions.clear();
    globalCounter->posOffset.clear();
#endif
    globalCounter->facetHitCounters.resize(model->nbFacets_total);
    globalCounter->leakCounter.resize(model->nbFacets_total);


    for (auto &mesh: model->triangle_meshes) {
        int lastTexture = -1;
        int lastProfile = -1;
        for (auto &facet: mesh->poly) {

            // has texture?
            if ((facet.texProps.textureFlags) &&
                (lastTexture < (int) facet.parentIndex)) { // prevent duplicates
                unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;
                std::vector<Texel64> texels(width * height);
                globalCounter->textures.insert(
                        std::pair<uint32_t, std::vector<Texel64>>(facet.parentIndex, std::move(texels)));
            }

            // has profile?
            if ((facet.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile) &&
                (lastProfile < (int) facet.parentIndex)) {
                std::vector<Texel64> texels(PROFILE_SIZE);
                globalCounter->profiles.insert(
                        std::pair<uint32_t, std::vector<Texel64>>(facet.parentIndex, std::move(texels)));
            }
        }
    }

    for (auto &mesh: model->poly_meshes) {
        int lastTexture = -1;
        int lastProfile = -1;
        for (auto &facet: mesh->poly) {

            // has texture?
            if ((facet.texProps.textureFlags) &&
                (lastTexture < (int) facet.parentIndex)) { // prevent duplicates
                unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;
                std::vector<Texel64> texels(width * height);
                globalCounter->textures.insert(
                        std::pair<uint32_t, std::vector<Texel64>>(facet.parentIndex, std::move(texels)));
            }

            // has profile?
            if ((facet.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile) &&
                (lastProfile < (int) facet.parentIndex)) {
                std::vector<Texel64> texels(PROFILE_SIZE);
                globalCounter->profiles.insert(
                        std::pair<uint32_t, std::vector<Texel64>>(facet.parentIndex, std::move(texels)));
            }
        }
    }

    return 0;
}
/**
 * Fetch simulation data from the device
 * @return 1=could not load GPU Sim, 0=successfully loaded
 */
unsigned long long int SimulationControllerGPU::ConvertSimulationData(GlobalSimuState &gState) {
    // global

    //facet hit counters + miss
    for (unsigned int facIndex = 0; facIndex < model->nbFacets_total; facIndex++) {
        unsigned int facParent = !model->triangle_meshes.empty() ? model->triangle_meshes[0]->poly[facIndex].parentIndex : model->poly_meshes[0]->poly[facIndex].parentIndex;
        auto &facetHits = gState.facetStates[facParent].momentResults[0].hits;
        auto &gCounter = globalCounter->facetHitCounters[facIndex];

        // increment global states
        gState.globalHits.globalHits.nbMCHit += gCounter.nbMCHit;
        gState.globalHits.globalHits.nbDesorbed += gCounter.nbDesorbed;
        gState.globalHits.globalHits.nbAbsEquiv += gCounter.nbAbsEquiv;
        gState.globalHits.globalHits.nbHitEquiv += gCounter.nbHitEquiv;

        facetHits.nbMCHit += gCounter.nbMCHit;
        facetHits.nbDesorbed += gCounter.nbDesorbed;
        facetHits.nbAbsEquiv += gCounter.nbAbsEquiv;
        facetHits.nbHitEquiv += gCounter.nbHitEquiv;

        facetHits.sum_v_ort += gCounter.sum_v_ort;
        facetHits.sum_1_per_velocity += gCounter.sum_1_per_velocity;
        facetHits.sum_1_per_ort_velocity += gCounter.sum_1_per_ort_velocity;
    }

    // Add up profiles
    if (!globalCounter->profiles.empty()) {
        double timeCorrection = model->wp.finalOutgassingRate;
        for (auto &[id, profiles]: globalCounter->profiles) {

            // triangles
            for (auto &mesh: model->triangle_meshes) {
                int previousId = 0;
                for (auto &facet: mesh->poly) {
                    if ((facet.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile) &&
                        (id == facet.parentIndex)) {
                        auto &profileHits = gState.facetStates[id].momentResults[0].profile;
                        assert(!profileHits.empty());
                        for (unsigned int s = 0; s < PROFILE_SIZE; ++s) {
                            profileHits[s].countEquiv += static_cast<double>(profiles[s].countEquiv);
                            profileHits[s].sum_v_ort += profiles[s].sum_v_ort_per_area;
                            profileHits[s].sum_1_per_ort_velocity += profiles[s].sum_1_per_ort_velocity;
                        }

                        break; //Only need 1 facet for texture position data
                    }
                }
            }
            //polygons
            for (auto &mesh: model->poly_meshes) {
                for (auto &facet: mesh->poly) {
                    if ((facet.profProps.profileType != flowgpu::PROFILE_FLAGS::noProfile)) {
                        auto &profileHits = gState.facetStates[facet.parentIndex].momentResults[0].profile;
                        assert(!profileHits.empty());
                        for (unsigned int s = 0; s < PROFILE_SIZE; ++s) {
                            profileHits[s].countEquiv += profiles[s].countEquiv;
                            profileHits[s].sum_v_ort += profiles[s].sum_v_ort_per_area;
                            profileHits[s].sum_1_per_ort_velocity += profiles[s].sum_1_per_ort_velocity;
                        }

                        break; //Only need 1 facet for texture position data
                    }
                }
            }
        }
    }

    // Add up textures
    if (!globalCounter->textures.empty()) {
        double timeCorrection = model->wp.finalOutgassingRate;
        for (auto &[id, textures]: globalCounter->textures) {

            // triangles
            for (auto &mesh: model->triangle_meshes) {
                int previousId = 0;
                for (auto &facet: mesh->poly) {
                    if ((facet.texProps.textureFlags) && (id == facet.parentIndex)) {
                        auto &textureHits = gState.facetStates[id].momentResults[0].texture;
                        assert(!textureHits.empty());
                        unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                        unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;
                        for (unsigned int h = 0; h < height; ++h) {
                            for (unsigned int w = 0; w < width; ++w) {
                                unsigned int index_glob =
                                        w + h * model->facetTex[facet.texProps.textureOffset].texWidth;
                                textureHits[index_glob].countEquiv += static_cast<double>(textures[index_glob].countEquiv);
                                textureHits[index_glob].sum_v_ort_per_area += textures[index_glob].sum_v_ort_per_area;
                                textureHits[index_glob].sum_1_per_ort_velocity += textures[index_glob].sum_1_per_ort_velocity;
                            }
                        }

                        break; //Only need 1 facet for texture position data
                    }
                }
            }
            //polygons
            for (auto &mesh: model->poly_meshes) {
                for (auto &facet: mesh->poly) {
                    if ((facet.texProps.textureFlags) && (id == facet.parentIndex)) {
                        auto &textureHits = gState.facetStates[id].momentResults[0].texture;
                        assert(!textureHits.empty());
                        unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                        unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;
                        for (unsigned int h = 0; h < height; ++h) {
                            for (unsigned int w = 0; w < width; ++w) {
                                unsigned int index_glob =
                                        w + h * model->facetTex[facet.texProps.textureOffset].texWidth;
                                textureHits[index_glob].countEquiv += static_cast<double>(textures[index_glob].countEquiv);
                                textureHits[index_glob].sum_v_ort_per_area += textures[index_glob].sum_v_ort_per_area;
                                textureHits[index_glob].sum_1_per_ort_velocity += textures[index_glob].sum_1_per_ort_velocity;
                            }
                        }

                        break; //Only need 1 facet for texture position data
                    }
                }
            }
        }
    }

    // Leak
    if (!globalCounter->leakCounter.empty()) {
        for (unsigned long leakCounter : globalCounter->leakCounter) {
            gState.globalHits.nbLeakTotal += leakCounter;
        }
        for (size_t i = 0; i < globalCounter->leakCounter.size(); ++i) {
            if (globalCounter->leakCounter[i] > 0)
                Log::console_msg_master(3, "{}[{}]  has {} / {} leaks\n",
                           i, !model->triangle_meshes.empty() ? model->triangle_meshes[0]->poly[i].parentIndex : model->poly_meshes[0]->poly[i].parentIndex, globalCounter->leakCounter[i],
                           gState.globalHits.nbLeakTotal);
        }
#ifdef DEBUGLEAKPOS
        for (size_t leakIndex = 0; leakIndex < globalCounter->leakPositions.size(); leakIndex++) {
            gState.globalHits.leakCache[(leakIndex + gState.globalHits.lastLeakIndex) %
                                        LEAKCACHESIZE].pos.x = globalCounter->leakPositions[leakIndex].x;
            gState.globalHits.leakCache[(leakIndex + gState.globalHits.lastLeakIndex) %
                                        LEAKCACHESIZE].pos.y = globalCounter->leakPositions[leakIndex].y;
            gState.globalHits.leakCache[(leakIndex + gState.globalHits.lastLeakIndex) %
                                        LEAKCACHESIZE].pos.z = globalCounter->leakPositions[leakIndex].z;
            gState.globalHits.leakCache[(leakIndex + gState.globalHits.lastLeakIndex) %
                                        LEAKCACHESIZE].dir.x = globalCounter->leakDirections[leakIndex].x;
            gState.globalHits.leakCache[(leakIndex + gState.globalHits.lastLeakIndex) %
                                        LEAKCACHESIZE].dir.y = globalCounter->leakDirections[leakIndex].y;
            gState.globalHits.leakCache[(leakIndex + gState.globalHits.lastLeakIndex) %
                                        LEAKCACHESIZE].dir.z = globalCounter->leakDirections[leakIndex].z;
        }
        gState.globalHits.lastLeakIndex =
                (gState.globalHits.lastLeakIndex + globalCounter->leakPositions.size()) % LEAKCACHESIZE;
        gState.globalHits.leakCacheSize = std::min(LEAKCACHESIZE, gState.globalHits.leakCacheSize +
                                                                  globalCounter->leakPositions.size());
#endif // DEBUGLEAKPOS
    }

    return gState.globalHits.nbLeakTotal;
}

//! Resize host buffers
void SimulationControllerGPU::Resize() {
#ifdef WITHDESORPEXIT
    data->hitData.resize(settings->kernelDimensions[0] * settings->kernelDimensions[1]);
#endif

    data->facetHitCounters.clear();
    data->texels.clear();
    data->profileSlices.clear();
    data->leakCounter.clear();

    //data->hit.resize(kernelDimensions[0]*kernelDimensions[1]);
    data->facetHitCounters.resize(model->nbFacets_total * EXTRAFACETCOUNTERS);
    data->texels.resize(model->textures.size());
    data->profileSlices.resize(model->profiles.size());
    data->leakCounter.resize(model->nbFacets_total);

    ResetGlobalCounter();

#ifdef DEBUGCOUNT
    data->detCounter.clear();
    data->uCounter.clear();
    data->vCounter.clear();
    data->detCounter.resize(NCOUNTBINS);
    data->uCounter.resize(NCOUNTBINS);
    data->vCounter.resize(NCOUNTBINS);
#endif

#ifdef DEBUGPOS
    data->positions.clear();
    data->posOffset.clear();
    data->posType.clear();
    data->positions.resize(NBPOSCOUNTS * 1);
    data->posOffset.resize(1);
    data->posType.resize(NBPOSCOUNTS * 1);
#endif

#ifdef DEBUGLEAKPOS
    data->leakPositions.clear();
    data->leakDirections.clear();
    data->leakPosOffset.clear();
    data->leakPositions.resize(NBCOUNTS * settings->kernelDimensions[0] * settings->kernelDimensions[1]);
    data->leakDirections.resize(NBCOUNTS * settings->kernelDimensions[0] * settings->kernelDimensions[1]);
    data->leakPosOffset.resize(settings->kernelDimensions[0] * settings->kernelDimensions[1]);
#endif
}

/*! print data in relation to a parent polygon (triangle data mapped/summed for parent polygon) */
void SimulationControllerGPU::PrintDataForParent() {
    // Find amount of Polygons, we don't have this information anymore
    unsigned int maxPoly = 0;
    for (auto &mesh: model->triangle_meshes) {
        for (auto &facet: mesh->poly) {
            maxPoly = std::max(maxPoly, facet.parentIndex);
        }
    }

    std::vector<unsigned long long int> counterMCHit(maxPoly + 1, 0);
    std::vector<unsigned long long int> counterDesorp(maxPoly + 1, 0);
    std::vector<double> counterAbsorp(maxPoly + 1, 0);

    for (unsigned int i = 0; i < data->facetHitCounters.size(); i++) {
        unsigned int facIndex = i % this->model->nbFacets_total;
        unsigned int facParent = model->triangle_meshes[0]->poly[facIndex].parentIndex;
        counterMCHit[facParent] += data->facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        counterDesorp[facParent] += data->facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        counterAbsorp[facParent] += data->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    for (unsigned int i = 0; i <= maxPoly; i++) {
        if (counterMCHit[i] > 0 || counterAbsorp[i] > 0 || counterDesorp[i] > 0)
            std::cout << i + 1 << " " << counterMCHit[i] << " " << counterDesorp[i] << " "
                      << static_cast<unsigned long long int>(counterAbsorp[i]) << std::endl;
    }

    for (auto &mesh: model->triangle_meshes) {
        int lastTexture = -1;
        for (auto &facet: mesh->poly) {
            if ((facet.texProps.textureFlags != flowgpu::TEXTURE_FLAGS::noTexture) &&
                (lastTexture < (int) facet.parentIndex)) {
                std::cout << "Texture for #" << facet.parentIndex << std::endl << " ";
                unsigned int total = 0;
                unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;

                for (unsigned int h = 0; h < height; ++h) {
                    for (unsigned int w = 0; w < width; ++w) {
                        unsigned int index = w + h * model->facetTex[facet.texProps.textureOffset].texWidth +
                                             model->facetTex[facet.texProps.textureOffset].texelOffset;
                        std::cout << data->texels[index].countEquiv << "  ";
                        total += data->texels[index].countEquiv;
                    }
                    std::cout << std::endl << " ";
                }
                std::cout << std::endl;
                std::cout << "  total: " << total << std::endl;

                lastTexture = facet.parentIndex;
            }
        }
    }
}

/*! Print various statistics from the downloaded data (debug + runtime totals) */
void SimulationControllerGPU::PrintData() {
#ifdef DEBUGCOUNT
    std::cout << "Determinant Distribution:"<<std::endl;
    for(int i=0;i<NBCOUNTS;i++)
        std::cout << "["<< ((float)i/NBCOUNTS)*(DETHIGH-DETLOW)+DETLOW << "] " << detCounter[i] << std::endl;
    std::cout << "U Distribution:"<<std::endl;
    for(int i=0;i<NBCOUNTS;i++)
        std::cout << "["<< ((float)i/NBCOUNTS)*(UHIGH-ULOW)+ULOW << "] " << uCounter[i] << std::endl;
    std::cout << "V Distribution:"<<std::endl;
    for(int i=0;i<NBCOUNTS;i++)
        std::cout << "["<< ((float)i/NBCOUNTS)*(VHIGH-VLOW)+VLOW << "] " << vCounter[i] << std::endl;

    /*for(int i=0;i<data->detCounter.size();i++) std::cout << "" << ((float)i/NBCOUNTS)*(DETHIGH - DETLOW) + DETLOW << " " << data->detCounter[i] << std::endl;
    for(int i=0;i<data->uCounter.size();i++) std::cout << "" << ((float)i/NBCOUNTS)*(UHIGH - ULOW) + ULOW << " " << data->uCounter[i] << std::endl;
    for(int i=0;i<data->vCounter.size();i++) std::cout << "" << ((float)i/NBCOUNTS)*(VHIGH - VLOW) + VLOW << " " << data->vCounter[i] << std::endl;*/

#endif

#ifdef DEBUGPOS

    int nbPos = NBPOSCOUNTS;

    const int hitPositionsPerMol = 30;
    for (int i = 0; i < data->positions.size();) {
        //std::cout << i/(NBCOUNTS) << " " << data->posOffset[i/(NBCOUNTS)] << " ";
        std::cout << "{";
        for (int pos = 0; pos < hitPositionsPerMol; pos++) {
            size_t index = i / (NBPOSCOUNTS) * NBPOSCOUNTS + pos;
            std::cout << "{" << data->positions[index].x << "," << data->positions[index].y << ","
                      << data->positions[index].z << "}";
            if (pos != hitPositionsPerMol - 1) std::cout << ",";
            //std::cout << data->positions[index].x << "," << data->positions[index].y << "," << data->positions[index].z <<"   ";
        }
        i += nbPos; // jump to next molecule/thread
        std::cout << "}," << std::endl;
    }
#endif

    std::vector<unsigned int> counterMCHit(this->model->nbFacets_total, 0);
    std::vector<unsigned int> counterDesorp(this->model->nbFacets_total, 0);
    std::vector<double> counterAbsorp(this->model->nbFacets_total, 0);

    for (unsigned int i = 0; i < data->facetHitCounters.size(); i++) {
        unsigned int facIndex = i % this->model->nbFacets_total;
        counterMCHit[facIndex] += data->facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        counterDesorp[facIndex] += data->facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        counterAbsorp[facIndex] += data->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)

        /*if(data->facetHitCounters[i].nbMCHit == 0*//* || data->facetHitCounters[i].nbAbsEquiv == 0*//*){
            std::cout << "["<<i/(this->model->nbFacets_total)<<"] on facet #"<<i%this->model->nbFacets_total<<" has total hits >>> "<< data->facetHitCounters[i].nbMCHit<< " / total abs >>> " << data->facetHitCounters[i].nbAbsEquiv<<" ---> "<< i<<std::endl;
        }*/
    }

    for (unsigned int i = 0; i < this->model->nbFacets_total; i++) {
        if (counterMCHit[i] > 0 || counterAbsorp[i] > 0 || counterDesorp[i] > 0)
            std::cout << i + 1 << " " << counterMCHit[i] << " " << counterDesorp[i] << " "
                      << static_cast<unsigned int>(counterAbsorp[i]) << std::endl;
    }

    /*unsigned long long int total_counter = 0;
    unsigned long long int total_abs = 0;
    unsigned long long int total_des = 0;
    for(unsigned int i = 0; i < this->model->nbFacets_total; i++){
        if(counter2[i]>0 || absorb[i]> 0 || desorb[i]>0) std::cout << i+1 << " " << counter2[i]<<" " << desorb[i]<<" " << absorb[i]<<std::endl;
        total_counter += counter2[i];
        total_abs += absorb[i];
        total_des += desorb[i];
    }
    std::cout << " total hits >>> "<< total_counter<<std::endl;
    std::cout << " total  abs >>> "<< total_abs<<std::endl;
    std::cout << " total  des >>> "<< total_des<<std::endl;
    std::cout << " total miss >>> "<< *data->leakCounter.data()<< " -- miss/hit ratio: "<<(double)(*data->leakCounter.data()) / total_counter <<std::endl;*/
}

/*! Update global runtime figures with downloaded data */
void SimulationControllerGPU::UpdateGlobalFigures() {
    uint64_t prevDes = globFigures.total_des;
    unsigned int num_threads = omp_get_max_threads();
    uint64_t total_counter = 0, total_des = 0, total_absd = 0;

#pragma omp parallel for reduction(+:total_counter,total_des,total_absd)
    for (unsigned int i = 0; i < globalCounter->facetHitCounters.size(); i++) {
        total_counter += globalCounter->facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        //if(endCalled) figures.ndes_stop += globalCounter->facetHitCounters[i].nbDesorbed;
        total_des += globalCounter->facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        total_absd += globalCounter->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    globFigures.total_counter += total_counter;
    globFigures.total_des += total_des;
    globFigures.total_absd += total_absd;
    if (endCalled)
        globFigures.ndes_stop += globFigures.total_des - prevDes;
}

/*! Add the downloaded data to total counters and print their values */
void SimulationControllerGPU::PrintTotalCounters() {
    uint64_t prevDes = figures.total_des;
    figures.total_counter = 0;
    figures.total_des = 0;
    //figures.ndes_stop = 0;
    figures.total_absd = 0.0;
    figures.total_leak = 0;

    for (unsigned int i = 0; i < globalCounter->facetHitCounters.size(); i++) {
        figures.total_counter += globalCounter->facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        //if(endCalled) figures.ndes_stop += globalCounter->facetHitCounters[i].nbDesorbed;
        figures.total_des += globalCounter->facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        figures.total_absd += globalCounter->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }
    for (unsigned int i = 0; i < globalCounter->leakCounter.size(); i++) {
        figures.total_leak += globalCounter->leakCounter[i]; // let misses count as 0 (-1+1)
    }
    if (endCalled)
        figures.ndes_stop += figures.total_des - prevDes;

    Log::console_msg(5, " Step: hits >>> {}", figures.total_counter);
    Log::console_msg(5, " __  des >>> {} ({})", figures.total_des, figures.ndes_stop);
    Log::console_msg(5, " __  abs >>> {}", static_cast<unsigned long long int>(figures.total_absd));
    Log::console_msg(5, " __  miss >>> {} -- miss/hit ratio: {}\n", figures.total_leak,
               static_cast<double>(figures.total_leak) /
               static_cast<double>(figures.total_counter));
}

/*! download the device data and write the data to a file */
void SimulationControllerGPU::WriteDataToFile(const std::string &fileName) {
    uint32_t nbFacets = this->model->nbFacets_total;

#ifdef DEBUGCOUNT
    std::ofstream detfile,ufile,vfile;
    detfile.open ("det_counter.txt");
    ufile.open ("u_counter.txt");
    vfile.open ("v_counter.txt");

    for(int i=0;i<data->detCounter.size();i++) detfile << "" << ((float)i/NBCOUNTS)*(DETHIGH - DETLOW) + DETLOW << " " << data->detCounter[i] << std::endl;
    for(int i=0;i<data->uCounter.size();i++) ufile << "" << ((float)i/NBCOUNTS)*(UHIGH - ULOW) + ULOW << " " << data->uCounter[i] << std::endl;
    for(int i=0;i<data->vCounter.size();i++) vfile << "" << ((float)i/NBCOUNTS)*(VHIGH - VLOW) + VLOW << " " << data->vCounter[i] << std::endl;

    detfile.close();
    ufile.close();
    vfile.close();
#endif

#ifdef DEBUGPOS
    std::ofstream posFile;
    posFile.open("debug_positions.txt");

    int nbPos = NBPOSCOUNTS;

    const int hitPositionsPerMol = 30;
    for (int i = 0; i < data->positions.size();) {
        //posFile << i/(NBCOUNTS) << " " << data->posOffset[i/(NBCOUNTS)] << " ";
        posFile << "{";
        for (int pos = 0; pos < hitPositionsPerMol; pos++) {
            size_t index = i / (nbPos) * nbPos + pos;
            posFile << "{" << data->positions[index].x << "," << data->positions[index].y << ","
                    << data->positions[index].z << "}";
            if (pos != hitPositionsPerMol - 1) posFile << ",";
        }
        i += nbPos; // jump to next molecule/thread
        posFile << "}," << std::endl;
    }
    posFile.close();
#endif

    std::vector<uint64_t> counterMCHit(nbFacets, 0);
    std::vector<uint64_t> counterDesorp(nbFacets, 0);
    std::vector<double> counterAbsorp(nbFacets, 0);


    //std::ofstream facetCounterEveryFile("every"+fileName);

    for (unsigned int i = 0; i < globalCounter->facetHitCounters.size(); i++) {
        unsigned int facIndex = i % nbFacets;
        counterMCHit[facIndex] += globalCounter->facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        counterDesorp[facIndex] += globalCounter->facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        counterAbsorp[facIndex] += globalCounter->facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
        //if(data->facetHitCounters[i].nbMCHit>0 || data->facetHitCounters[i].nbDesorbed> 0 || data->facetHitCounters[i].nbAbsEquiv>0)
        // facetCounterEveryFile << (i/this->model->nbFacets_total) << " " << (i%this->model->nbFacets_total)+1 << " " << data->facetHitCounters[i].nbMCHit << " " << data->facetHitCounters[i].nbDesorbed << " " << static_cast<unsigned int>(data->facetHitCounters[i].nbAbsEquiv) << std::endl;
    }
    //facetCounterEveryFile.close();

    std::ofstream facetCounterFile;
    facetCounterFile.open(fileName);
    for (unsigned int i = 0; i < nbFacets; i++) {
        //if(counter2[i]>0 || absorb[i]> 0 || desorb[i]>0)
        facetCounterFile << std::setprecision(12) << i + 1 << " " << counterMCHit[i] << " " << counterDesorp[i] << " "
                         << static_cast<unsigned int>(counterAbsorp[i]) << std::endl;
    }
    facetCounterFile.close();

    // Texture output
    for (auto &mesh: model->triangle_meshes) {
        int lastTexture = -1;
        for (auto &facet: mesh->poly) {
            if ((facet.texProps.textureFlags != flowgpu::TEXTURE_FLAGS::noTexture) &&
                (lastTexture < (int) facet.parentIndex)) {
                facetCounterFile.open("textures" + std::to_string(facet.parentIndex) + ".txt");

                unsigned long long int total0 = 0;
                double total1 = 0;
                double total2 = 0;

                unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;

                auto &texels = globalCounter->textures[facet.parentIndex];
                // textures
                for (unsigned int h = 0; h < height; ++h) {
                    for (unsigned int w = 0; w < width; ++w) {
                        unsigned int index = w + h * width;
                        facetCounterFile << w << " " << h << " " << texels[index].countEquiv << "  "
                                         << texels[index].sum_v_ort_per_area << "  "
                                         << texels[index].sum_1_per_ort_velocity << std::endl;
                        total0 += texels[index].countEquiv;
                        total1 += texels[index].sum_v_ort_per_area;
                        total2 += texels[index].sum_1_per_ort_velocity;
                    }
                }

                // profiles
                size_t binSize = 100;
                std::vector<unsigned long long int> bin_count(binSize);
                std::vector<double> bin_sumv(binSize);
                std::vector<double> bin_sum1(binSize);
                for (unsigned int h = 0; h < height; ++h) {
                    for (unsigned int w = 0; w < width; ++w) {
                        unsigned int index = w + h * width;
                        //if(h == (height/2)){
                        unsigned int bin_index = index / std::max((unsigned int) (width * height / binSize), 1u);
                        if (width * height < binSize) {
                            bin_count[bin_index] += texels[index].countEquiv;
                            bin_sumv[bin_index] += texels[index].sum_v_ort_per_area;
                            bin_sum1[bin_index] += texels[index].sum_1_per_ort_velocity;
                        }
                        // }

                        /*unsigned int index = w+h*model->facetTex[facet.texProps.textureOffset].texWidth;
                        facetCounterFile << data->texels[index].countEquiv << "  "<< data->texels[index].sum_v_ort_per_area << "  "<< data->texels[index].sum_1_per_ort_velocity<<std::endl;
                        total0 += data->texels[index].countEquiv;
                        total1 += data->texels[index].sum_v_ort_per_area;
                        total2 += data->texels[index].sum_1_per_ort_velocity;

                        bin_count[index/(width*height/binSize)] += data->texels[index].countEquiv;
                        bin_sumv[index/(width*height/binSize)] += data->texels[index].sum_v_ort_per_area;
                        bin_sum1[index/(width*height/binSize)] += data->texels[index].sum_1_per_ort_velocity;*/
                    }
                }
                facetCounterFile << std::endl;
                facetCounterFile.close();

                facetCounterFile.open("profiles" + std::to_string(facet.parentIndex) + ".txt");
                for (int i = 0; i < binSize; ++i) {
                    facetCounterFile << std::setprecision(12) << bin_count[i] << "  " << bin_sumv[i] << "  "
                                     << bin_sum1[i] << std::endl;
                }
                facetCounterFile.close();
                lastTexture = facet.parentIndex;
            }
        }
    }

}

//! Get nb of total hits
unsigned long long int SimulationControllerGPU::GetTotalHits() {

    unsigned long long int total_counter = 0;
    for (auto &facetHitCounter: data->facetHitCounters) {
        total_counter += facetHitCounter.nbMCHit; // let misses count as 0 (-1+1)
    }
    return total_counter;
}

/**
 *
 * @return 1=could not load GPU Sim, 0=successfully loaded
 */
int SimulationControllerGPU::CloseSimulation() {
    try {
        if (optixHandle) {
            //delete optixHandle;
            optixHandle = nullptr;
        }
    } catch (std::runtime_error &e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}

// Do a soft reset to keep active particles in memory
int SimulationControllerGPU::ResetSimulation(bool softReset) {
    if (!softReset && optixHandle) {
        CloseSimulation();
        //optixHandle->resetDeviceData(settings->kernelDimensions);
    }

    figures.total_des = 0;
    figures.total_abs = 0;
    figures.total_counter = 0;
    figures.total_absd = 0.0;
    hasEnded = false;

#if defined (WITHDESORPEXIT)
    if (!data->hitData.empty()) // only resize if already initialized once
        this->Resize();
#endif
    return 0;
}

//! Get pointer to HostData structure for GPU results
GlobalCounter *SimulationControllerGPU::GetGlobalCounter() {
    return globalCounter.get();
}

//! Change GPU parameters forwarded from GUI to Sim
int SimulationControllerGPU::ChangeParams(std::shared_ptr<flowgpu::MolflowGPUSettings> molflowGlobal) {
    if(!settings)
        settings = std::make_shared<flowgpu::MolflowGPUSettings>();
    *settings = *molflowGlobal;
    return 0;
}