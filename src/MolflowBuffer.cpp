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

#include "MolflowBuffer.h"
#include <limits>

WorkerParams::WorkerParams(){
    accel_type = KD;
    timeWindowSize = 1E-10; //Dirac-delta desorption pulse at t=0
    useMaxwellDistribution = true;
    calcConstantFlow = true;
    gasMass = 28.0;
    enableDecay = false;
    halfLife = 1;
    finalOutgassingRate = finalOutgassingRate_Pa_m3_sec = totalDesorbedMolecules = 0.0;
    motionType = 0;
}

OntheflySimulationParams::OntheflySimulationParams(){
    nbProcess = 0;
    enableLogging = false;
    desorptionLimit = 0;
    lowFluxCutoff = 1.0E-7;
    lowFluxMode = false;
    timeLimit = 0.0;
    logFacetId = std::numeric_limits<size_t>::max();
    logLimit = 0;

    benchmarkADS = false;
    raySampling = false;
}

void FacetHistogramBuffer::Resize(const HistogramParams& params){
    this->nbHitsHistogram.resize(params.recordBounce ? params.GetBounceHistogramSize() : 0); 
    this->nbHitsHistogram.shrink_to_fit();
    this->distanceHistogram.resize(params.recordDistance ? params.GetDistanceHistogramSize() : 0); 
    this->distanceHistogram.shrink_to_fit();
    this->timeHistogram.resize(params.recordTime ? params.GetTimeHistogramSize() : 0); 
    this->timeHistogram.shrink_to_fit();
}

void FacetHistogramBuffer::Reset(){
    ZEROVECTOR(nbHitsHistogram);
    ZEROVECTOR(distanceHistogram);
    ZEROVECTOR(timeHistogram);
}

FacetProperties::FacetProperties(size_t nbIndices) {
    nbIndex = nbIndices;

    sticking = 0.0;
    opacity = 1.0;

    profileType = PROFILE_NONE;

    texWidth = 0;
    texHeight = 0;
    texWidth_precise = 0.0;
    texHeight_precise = 0.0;
    center.x = 0.0;
    center.y = 0.0;
    center.z = 0.0;

    is2sided = false;
    isProfile = false;
    //wp.isOpaque = true;
    isTextured = false;
    countAbs = false;
    countRefl = false;
    countTrans = false;
    countDirection = false;

    superIdx = 0;
    superDest = 0;
    teleportDest = 0;
    isVolatile = false;
    
#if defined(MOLFLOW)
    temperature = 293.15; // 20degC
    outgassing = 0.0;           // 1 unit*l/s //will be outgasssing
    desorbType = DES_NONE;
    desorbTypeN = 0.0;

    reflection.diffusePart = 1.0; //totally diffuse reflection
    reflection.specularPart = 0.0;
    reflection.cosineExponent = 0.0; //Cos^0 = uniform

    countDes = false;
    countACD = false;
    useOutgassingFile = false;
    accomodationFactor = 1.0;

    enableSojournTime = false;
    sojournFreq = 1E13;
    sojournE = 100;

    outgassing_paramId = -1;
    opacity_paramId = -1;
    sticking_paramId = -1;

    isMoving = false;
    

    anglemapParams.record = false;

    anglemapParams.phiWidth = anglemapParams.thetaLowerRes = anglemapParams.thetaHigherRes = 0;
    anglemapParams.thetaLimit = 1.570796326; //slightly lower than PI/2

    //facetHistogramParams.record = false;

    totalOutgassing = 0.0;
#endif

#if defined(SYNRAD)
    doScattering = false;
	rmsRoughness = 100.0E-9; //100nm
	autoCorrLength = 100 * 100E-9; //tau=autoCorr/RMS=100

	reflectType = REFLECTION_SPECULAR;
	recordSpectrum = false;
#endif
}
