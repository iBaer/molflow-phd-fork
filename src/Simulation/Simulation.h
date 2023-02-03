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
#pragma once

#include <tuple>
#include <vector>
#include <mutex>

#include "SimulationUnit.h"
#include "Buffer_shared.h" //Facetproperties
#include "SMP.h"
#include "Vector.h"
#include "Random.h"
#include "ProcessControl.h"
#include "SimulationController.h"
#include "MolflowSimGeom.h"
#include "Particle.h"
#include "RayTracing/RTHelper.h"

class Parameter;

// Local simulation structure
class Simulation : public SimulationUnit {
public:

    Simulation();
    Simulation(Simulation&& o) noexcept ;
    ~Simulation() override = default;

    std::pair<int, std::optional<std::string>> SanityCheckModel(bool strictCheck) override;
    void ClearSimulation() override;
    size_t LoadSimulation(char *loadStatus) override;
    int RebuildAccelStructure() override;

    void ResetSimulation() override;

    size_t GetHitsSize() override;

    int ReinitializeParticleLog() override;
    MFSim::Particle * GetParticle(size_t i) override;
    void SetNParticle(size_t n, bool fixedSeed) override;

	//size_t totalDesorbed;           // Total desorption number (for this process, not reset on UpdateMCHits)

	// Geometry

	//double stepPerSec=0.0;  // Avg number of step per sec
	//bool loadOK;        // Load OK flag
	//bool lastHitUpdateOK;  // Last hit update timeout
	bool lastLogUpdateOK; // Last log update timeout
	bool hasVolatile;   // Contains volatile facet

    // Particle coordinates (MC)//std::vector<FacetHistogramBuffer> tmpGlobalHistograms; //Recorded histogram since last UpdateMCHits, 1+nbMoment copies
    //ParticleLog tmpParticleLog; //Recorded particle log since last UpdateMCHits
    std::vector<MFSim::Particle> particles;
    mutable std::timed_mutex tMutex;

    void FindBestADS();

    bool RunParallel(size_t nSteps);
};
// -- Methods ---------------------------------------------------