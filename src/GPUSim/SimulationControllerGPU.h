//
// Created by pbahr on 04/10/2019.
//

#ifndef MOLFLOW_SIMULATIONOPTIX_H
#define MOLFLOW_SIMULATIONOPTIX_H


//#include "Simulation.h"
//#include "SimulationOptiX.h"
//#include "HostData.h"
#include "../Simulation/MolflowSimGeom.h"
#include "SimulationController.h"

class HostData;
class GlobalCounter;

namespace flowgpu {
    class SimulationOptiX;
    struct Model;
}
struct RuntimeFigures {
    uint32_t runCount {0};
    uint32_t runCountNoEnd {0};
    double desPerRun {0.0};
    double desPerRun_stop {0.0};

    unsigned long long int total_counter = 0;
    unsigned long long int total_abs = 0;
    double total_absd = 0;
    unsigned long long int total_des = 0;
    unsigned long long int total_leak = 0;
    uint64_t ndes_stop = 0;
    uint64_t exitCount = 0;
};

class SimulationControllerGPU : public SimulationController {
protected:
    //flowgpu::SampleWindow* window;
    std::shared_ptr<flowgpu::SimulationOptiX> optixHandle;
    std::shared_ptr<flowgpu::Model> model;
    unsigned int kernelDimensions[2]; // blocks and threads per block

    std::unique_ptr<HostData> data;
    std::unique_ptr<GlobalCounter> globalCounter;

    void Resize();
    unsigned long long int GetTotalHits();
private:
    void CheckAndBlockDesorption();
    void CheckAndBlockDesorption_exact(double threshold);
    void PrintData();
    void PrintDataForParent();
    void PrintTotalCounters();
    void UpdateGlobalFigures();
    void WriteDataToFile(const std::string& fileName);

    void CalcRuntimeFigures();
public:
    SimulationControllerGPU(size_t parentPID, size_t procIdx, size_t nbThreads, SimulationUnit *simUnit,
                            std::shared_ptr<ProcComm> pInfo);
    ~SimulationControllerGPU();

    int LoadSimulation(std::shared_ptr<flowgpu::Model> loaded_model, size_t launchSize);
    uint64_t RunSimulation();
    int CloseSimulation();
    int ResetSimulation(bool softReset);
    int ResetGlobalCounter();
    void AllowNewParticles();

    unsigned long long int GetSimulationData(bool silent = true);
    void IncreaseGlobalCounters(HostData* tempData);
    GlobalCounter* GetGlobalCounter() ;
    int RemainingStepsUntilStop();
    double GetTransProb(size_t polyIndex);
    double GetTransProb();
    bool runLoop();

    RuntimeFigures figures;
    RuntimeFigures globFigures;

    bool hasEnded{false};
    bool endCalled{false};

    unsigned long long int ConvertSimulationData(GlobalSimuState &gState);
    
    
public:
    int Start() override;
    bool Load() override;
    int RebuildAccel() override;
    int Reset() override;
    void EmergencyExit() override; // Killing threads
};


#endif //MOLFLOW_SIMULATIONOPTIX_H