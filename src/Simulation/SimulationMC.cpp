/*
Program:     MolFlow+ / Synrad+
Description: Monte Carlo simulator for ultra-high vacuum and synchrotron radiation
Authors:     Jean-Luc PONS / Roberto KERSEVAN / Marton ADY
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
#include <cmath>
#include <cstdio>
#include <tuple> //std::tie
#include <cstring>
#include <sstream>
#include <cereal/archives/binary.hpp>
#include "Simulation.h"
#include "IntersectAABB_shared.h"
#include "Random.h"
#include "Helper/MathTools.h"

#include "Parameter.h"
#include <omp.h>

class AnglemapGeneration {
public:
    static double GetTheta(const double &thetaIndex, const AnglemapParams &anglemapParams);

    static double GetPhi(const double &phiIndex, const AnglemapParams &anglemapParams);

    static double GetPhipdfValue(const double &thetaIndex, const int &phiLowerIndex,
                                 const AnglemapParams &anglemapParams, const std::vector<size_t> &angleMapPDF);

    static double GetPhiCDFValue(const double &thetaIndex, const int &phiLowerIndex,
                                 const AnglemapParams &anglemapParams, const Anglemap &anglemap);

    static double GetPhiCDFSum(const double &thetaIndex, const AnglemapParams &anglemapParams,
                               const Anglemap &anglemap);

    static std::tuple<double, int, double>
    GenerateThetaFromAngleMap(const AnglemapParams &anglemapParams, const Anglemap &anglemap,
                              const double lookupValue);

    static double GeneratePhiFromAngleMap(const int &thetaLowerIndex, const double &thetaOvershoot,
                                          const AnglemapParams &anglemapParams, Anglemap &anglemap,
                                          const std::vector<size_t> &angleMapPDF,
                                          double lookupValue);
};

bool CurrentParticleStatus::UpdateMCHits(GlobalSimuState &globSimuState, size_t nbMoments, DWORD timeout) {
    TEXTURE_MIN_MAX texture_limits_old[3];
    int i, j, s, x, y;
//#if defined(DEBUG)
    double t0, t1;
    t0 = omp_get_wtime();
    if (!globSimuState.tMutex.try_lock_for(std::chrono::milliseconds(timeout))) {
        return false;
    }
//#endif
    //SetState(PROCESS_STARTING, "Waiting for 'hits' dataport access...", false, true);

    //bool lastHitUpdateOK = LockMutex(worker->results.mutex, timeout);
    //if (!lastHitUpdateOK) return false; //Timeout, will try again later
    //SetState(PROCESS_STARTING, "Updating MC hits...", false, true);

    //buffer = (BYTE *) dpHit->buff;
    //gHits = (GlobalHitBuffer *) buffer;

    // Global hits and leaks: adding local hits to shared memory
    /*gHits->globalHits += tmpGlobalResult.globalHits;
    gHits->distTraveled_total += tmpGlobalResult.distTraveled_total;
    gHits->distTraveledTotal_fullHitsOnly += tmpGlobalResult.distTraveledTotal_fullHitsOnly;*/

    {
        //auto& tmpState = currentParticles[prIdx].tmpState;

        //for (auto &tmpState : tmpGlobalResults) {
        //for(auto& tmpState : tmpGlobalResults) {
/*#if defined(DEBUG)
        std::cout << "[" << 0 << "] "
                  << globSimuState.globalHits.globalHits.hit.nbMCHit + tmpState.globalHits.globalHits.hit.nbMCHit
                  << " : " << globSimuState.globalHits.globalHits.hit.nbMCHit << " += "
                  << tmpState.globalHits.globalHits.hit.nbMCHit << std::endl;
#endif*/
        //printf("UP![%zu] %lu + %zu = %zu\n", particleId, tmpState.globalHits.globalHits.hit.nbDesorbed, globSimuState.globalHits.globalHits.hit.nbDesorbed, tmpState.globalHits.globalHits.hit.nbDesorbed + globSimuState.globalHits.globalHits.hit.nbDesorbed);
        globSimuState.globalHits.globalHits += tmpState.globalHits.globalHits;
        globSimuState.globalHits.distTraveled_total += tmpState.globalHits.distTraveled_total;
        globSimuState.globalHits.distTraveledTotal_fullHitsOnly += tmpState.globalHits.distTraveledTotal_fullHitsOnly;

        // Update too late
        //totalDesorbed += tmpState.globalHits.globalHits.hit.nbDesorbed;


        /*gHits->globalHits.hit.nbMCHit += tmpGlobalResult.globalHits.hit.nbMCHit;
        gHits->globalHits.hit.nbHitEquiv += tmpGlobalResult.globalHits.hit.nbHitEquiv;
        gHits->globalHits.hit.nbAbsEquiv += tmpGlobalResult.globalHits.hit.nbAbsEquiv;
        gHits->globalHits.hit.nbDesorbed += tmpGlobalResult.globalHits.hit.nbDesorbed;*/

        //model->wp.sMode = MC_MODE;
        //for(i=0;i<BOUNCEMAX;i++) globState.globalHits.wallHits[i] += wallHits[i];

        // Leak
        for (size_t leakIndex = 0; leakIndex < tmpState.globalHits.leakCacheSize; leakIndex++)
            globSimuState.globalHits.leakCache[(leakIndex + globSimuState.globalHits.lastLeakIndex) %
                                               LEAKCACHESIZE] = tmpState.globalHits.leakCache[leakIndex];
        globSimuState.globalHits.nbLeakTotal += tmpState.globalHits.nbLeakTotal;
        globSimuState.globalHits.lastLeakIndex =
                (globSimuState.globalHits.lastLeakIndex + tmpState.globalHits.leakCacheSize) % LEAKCACHESIZE;
        globSimuState.globalHits.leakCacheSize = Min(LEAKCACHESIZE, globSimuState.globalHits.leakCacheSize +
                                                                    tmpState.globalHits.leakCacheSize);

        // HHit (Only prIdx 0)
        if (particleId == 0) {
            for (size_t hitIndex = 0; hitIndex < tmpState.globalHits.hitCacheSize; hitIndex++)
                globSimuState.globalHits.hitCache[(hitIndex + globSimuState.globalHits.lastHitIndex) %
                                                  HITCACHESIZE] = tmpState.globalHits.hitCache[hitIndex];

            if (tmpState.globalHits.hitCacheSize > 0) {
                globSimuState.globalHits.lastHitIndex =
                        (globSimuState.globalHits.lastHitIndex + tmpState.globalHits.hitCacheSize) % HITCACHESIZE;
                globSimuState.globalHits.hitCache[globSimuState.globalHits.lastHitIndex].type = HIT_LAST; //Penup (border between blocks of consecutive hits in the hit cache)
                globSimuState.globalHits.hitCacheSize = Min(HITCACHESIZE, globSimuState.globalHits.hitCacheSize +
                                                                          tmpState.globalHits.hitCacheSize);
            }
        }

        //Global histograms
        globSimuState.globalHistograms += tmpState.globalHistograms;

        // Facets
        globSimuState.facetStates += tmpState.facetStates;
    }

    if (particleId == 0) {
        // Another loop for a comlete global min/max texture search

        // first get tmp limit
        TEXTURE_MIN_MAX limits[3];
        for(auto& lim : limits){
            lim.max.all = lim.max.moments_only = 0;
            lim.min.all = lim.min.moments_only = HITMAX;
        }

        for (s = 0; s < model->sh.nbSuper; s++) {
            for (const SubprocessFacet &f : model->structures[s].facets) {
                if (f.sh.isTextured) {
                    for (int m = 0; m < (1 + nbMoments); m++) {
                        {
                            // go on if the facet was never hit before
                            auto &facetHitBuffer = globSimuState.facetStates[f.globalId].momentResults[m].hits;
                            if (facetHitBuffer.hit.nbMCHit == 0 && facetHitBuffer.hit.nbDesorbed == 0) continue;
                        }

                        //double dCoef = globState.globalHits.globalHits.hit.nbDesorbed * 1E4 * model->wp.gasMass / 1000 / 6E23 * MAGIC_CORRECTION_FACTOR;  //1E4 is conversion from m2 to cm2
                        const double timeCorrection =
                                m == 0 ? model->wp.finalOutgassingRate : (model->wp.totalDesorbedMolecules) /
                                                                        model->tdParams.moments[m - 1].second;
                        //model->wp.timeWindowSize;
                        //Timecorrection is required to compare constant flow texture values with moment values (for autoscaling)
                        const auto &texture = globSimuState.facetStates[f.globalId].momentResults[m].texture;
                        const size_t textureSize = texture.size();
                        for (size_t t = 0; t < textureSize; t++) {
                            //Add temporary hit counts

                            if (f.largeEnough[t]) {
                                double val[3];  //pre-calculated autoscaling values (Pressure, imp.rate, density)

                                val[0] = texture[t].sum_v_ort_per_area *
                                         timeCorrection; //pressure without dCoef_pressure
                                val[1] = texture[t].countEquiv * f.textureCellIncrements[t] *
                                         timeCorrection; //imp.rate without dCoef
                                val[2] = f.textureCellIncrements[t] * texture[t].sum_1_per_ort_velocity *
                                         timeCorrection; //particle density without dCoef

                                //Global autoscale
                                for (int v = 0; v < 3; v++) {
                                    limits[v].max.all = std::max(val[v],limits[v].max.all);

                                    if (val[v] > 0.0) {
                                        limits[v].min.all = std::min(val[v],limits[v].min.all);
                                    }
                                    //Autoscale ignoring constant flow (moments only)
                                    if (m != 0) {
                                        limits[v].max.moments_only = std::max(val[v],limits[v].max.moments_only);;

                                        if (val[v] > 0.0)
                                            limits[v].min.moments_only = std::min(val[v],limits[v].min.moments_only);;
                                    }
                                }
                            } // if largeenough
                        }
                    }
                }
            }
        }
        // Last put temp limits into global struct
        for(int v = 0; v < 3; ++v) {
            globSimuState.globalHits.texture_limits[v] = limits[v];
        }

    }
    //ReleaseDataport(dpHit);
    //TODO: //ReleaseMutex(worker->results.mutex);
    globSimuState.tMutex.unlock();

    //extern char *GetSimuStatus();
    //SetState(PROCESS_STARTING, GetSimuStatus(), false, true);

//#if defined(DEBUG)
    t1 = omp_get_wtime();
    //printf("Update hits (glob): %lf s [%zu]\n", (t1 - t0) * 1.0, particleId);
    //#endif

    return true;
}

// Compute particle teleport
void CurrentParticleStatus::PerformTeleport(SubprocessFacet *iFacet) {

    //Search destination
    SubprocessFacet *destination;
    bool found = false;
    bool revert = false;
    int destIndex;
    if (iFacet->sh.teleportDest == -1) {
        destIndex = teleportedFrom;
        if (destIndex == -1) {
            /*char err[128];
            sprintf(err, "Facet %d tried to teleport to the facet where the particle came from, but there is no such facet.", iFacet->globalId + 1);
            SetErrorSub(err);*/
            if (particleId == 0)RecordHit(HIT_REF);
            lastHitFacet = iFacet;
            return; //LEAK
        }
    } else destIndex = iFacet->sh.teleportDest - 1;

    //Look in which superstructure is the destination facet:
    for (size_t i = 0; i < model->sh.nbSuper && (!found); i++) {
        for (size_t j = 0; j < model->structures[i].facets.size() && (!found); j++) {
            if (destIndex == model->structures[i].facets[j].globalId) {
                destination = &(model->structures[i].facets[j]);
                if (destination->sh.superIdx != -1) {
                    structureId = destination->sh.superIdx; //change current superstructure, unless the target is a universal facet
                }
                teleportedFrom = (int) iFacet->globalId; //memorize where the particle came from
                found = true;
            }
        }
    }
    if (!found) {
        /*char err[128];
        sprintf(err, "Teleport destination of facet %d not found (facet %d does not exist)", iFacet->globalId + 1, iFacet->sh.teleportDest);
        SetErrorSub(err);*/
        if (particleId == 0)RecordHit(HIT_REF);
        lastHitFacet = iFacet;
        return; //LEAK
    }
    // Count this hit as a transparent pass
    if (particleId == 0)RecordHit(HIT_TELEPORTSOURCE);
    if (/*iFacet->texture && */iFacet->sh.countTrans)
        RecordHitOnTexture(iFacet, particleTime, true, 2.0, 2.0);
    if (/*iFacet->direction && */iFacet->sh.countDirection)
        RecordDirectionVector(iFacet, particleTime);
    ProfileFacet(iFacet, particleTime, true, 2.0, 2.0);
    //TODO: if(particleId==0)LogHit(iFacet, model, tmpParticleLog);
    if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);

    // Relaunch particle from new facet
    auto[inTheta, inPhi] = CartesianToPolar(direction, iFacet->sh.nU, iFacet->sh.nV,
                                            iFacet->sh.N);
    direction = PolarToCartesian(destination, inTheta, inPhi, false);
    // Move particle to teleport destination point
    double u = tmpFacetVars[iFacet->globalId].colU;
    double v = tmpFacetVars[iFacet->globalId].colV;
    position = destination->sh.O + u * destination->sh.U + v * destination->sh.V;
    if (particleId == 0)RecordHit(HIT_TELEPORTDEST);
    int nbTry = 0;
    if (!IsInFacet(*destination, u, v)) { //source and destination facets not the same shape, would generate leak
        // Choose a new starting point
        if (particleId == 0)RecordHit(HIT_ABS);
        found = false;
        while (!found && nbTry < 1000) {
            u = randomGenerator.rnd();
            v = randomGenerator.rnd();
            if (IsInFacet(*destination, u, v)) {
                found = true;
                position = destination->sh.O + u * destination->sh.U + v * destination->sh.V;
                if (particleId == 0)RecordHit(HIT_DES);
            }
        }
        nbTry++;
    }

    lastHitFacet = destination;

    //Count hits on teleport facets
    /*iFacet->sh.tmpCounter.hit.nbAbsEquiv++;
    destination->sh.tmpCounter.hit.nbDesorbed++;*/

    double ortVelocity =
            velocity * std::abs(Dot(direction, iFacet->sh.N));
    //We count a teleport as a local hit, but not as a global one since that would affect the MFP calculation
    /*iFacet->sh.tmpCounter.hit.nbMCHit++;
    iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / ortVelocity;
    iFacet->sh.tmpCounter.hit.sum_v_ort += 2.0*(model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/
    IncreaseFacetCounter(iFacet, particleTime, 1, 0, 0, 2.0 / ortVelocity,
                         2.0 * (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    tmpFacetVars[iFacet->globalId].isHit = true;
    /*destination->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / velocity;
    destination->sh.tmpCounter.hit.sum_v_ort += velocity*abs(DOT3(
    direction.x, direction.y, direction.z,
    destination->sh.N.x, destination->sh.N.y, destination->sh.N.z));*/
}

// Perform nbStep simulation steps (a step is a bounce)

bool CurrentParticleStatus::SimulationMCStep(size_t nbStep, size_t threadNum, size_t remainingDes) {

    // Check end of simulation
    /*if (model->otfParams.desorptionLimit > 0) {
        if (model->wp.totalDesorbedMolecules >=
            model->otfParams.desorptionLimit / model->otfParams.nbProcess) {
            //lastHitFacet = nullptr; // reset full particle status or go on from where we left
            return false;
        }
    }*/

    // Perform simulation steps
    int returnVal = true;
    //int allQuit = 0;

//#pragma omp parallel num_threads(nbThreads) default(none) firstprivate(nbStep) shared( returnVal, allQuit)
    {

        const int ompIndex = threadNum;//omp_get_thread_num();
        //printf("Simthread %i of %i (%i).\n", ompIndex, omp_get_num_threads(), omp_get_max_threads());

        particleId = ompIndex;
        size_t i;

        // start new particle when no previous hit facet was saved
        bool insertNewParticle = !lastHitFacet;
        for (i = 0; i < nbStep /*&& allQuit <= 0*/; i++) {
            if (insertNewParticle) {
                // quit on desorp error or limit reached
                if(!StartFromSource() || remainingDes-1==0){
                    returnVal = false; // desorp limit reached
                    break;
                }
                insertNewParticle = false;
                --remainingDes;
            }

            //return (lastHitFacet != nullptr);

            //Prepare output values
            auto[found, collidedFacet, d] = Intersect(*this, position,
                                                      direction);

            if (found) {
                // Second pass for transparent hits
                for (const auto &tpFacet : transparentHitBuffer) {
                    if (tpFacet) {
                        RegisterTransparentPass(tpFacet);
                    }
                }
                // Move particle to intersection point
                position =
                        position + d * direction;
                //distanceTraveled += d;

                const double lastParticleTime = particleTime; //memorize for partial hits
                particleTime +=
                        d / 100.0 / velocity; //conversion from cm to m

                if ((!model->wp.calcConstantFlow && (particleTime > model->wp.latestMoment))
                    || (model->wp.enableDecay &&
                        (expectedDecayMoment < particleTime))) {
                    //hit time over the measured period - we create a new particle
                    //OR particle has decayed
                    const double remainderFlightPath = velocity * 100.0 *
                                                       Min(model->wp.latestMoment - lastParticleTime,
                                                           expectedDecayMoment -
                                                           lastParticleTime); //distance until the point in space where the particle decayed
                    tmpState.globalHits.distTraveled_total += remainderFlightPath * oriRatio;
                    if (particleId == 0)RecordHit(HIT_LAST);
                    //distTraveledSinceUpdate += distanceTraveled;
                    insertNewParticle = true;
                    lastHitFacet=nullptr;
                } else { //hit within measured time, particle still alive
                    if (collidedFacet->sh.teleportDest != 0) { //Teleport
                        IncreaseDistanceCounters(d * oriRatio);
                        PerformTeleport(collidedFacet);
                    }
                        /*else if ((GetOpacityAt(collidedFacet, particleTime) < 1.0) && (randomGenerator.rnd() > GetOpacityAt(collidedFacet, particleTime))) {
                            //Transparent pass
                            tmpState.globalHits.distTraveled_total += d;
                            PerformTransparentPass(collidedFacet);
                        }*/
                    else { //Not teleport
                        IncreaseDistanceCounters(d * oriRatio);
                        const double stickingProbability = model->GetStickingAt(collidedFacet, particleTime);
                        if (!model->otfParams.lowFluxMode) { //Regular stick or bounce
                            if (stickingProbability == 1.0 ||
                                ((stickingProbability > 0.0) && (randomGenerator.rnd() < (stickingProbability)))) {
                                //Absorbed
                                RecordAbsorb(collidedFacet);
                                //currentParticle.lastHitFacet = nullptr; // null facet in case we reached des limit and want to go on, prevents leak
                            //distTraveledSinceUpdate += distanceTraveled;
                                insertNewParticle = true;
                                lastHitFacet=nullptr;
                            } else {
                                //Reflected
                                PerformBounce(collidedFacet);
                            }
                        } else { //Low flux mode
                            if (stickingProbability > 0.0) {
                                const double oriRatioBeforeCollision = oriRatio; //Local copy
                                oriRatio *= (stickingProbability); //Sticking part
                                RecordAbsorb(collidedFacet);
                                oriRatio =
                                        oriRatioBeforeCollision * (1.0 - stickingProbability); //Reflected part
                            } else
                                oriRatio *= (1.0 - stickingProbability);
                            if (oriRatio > model->otfParams.lowFluxCutoff) {
                                PerformBounce(collidedFacet);
                            } else { //eliminate remainder and create new particle
                                insertNewParticle = true;
                                lastHitFacet=nullptr;
                            }
                        }
                    }
                } //end hit within measured time
            } //end intersection found
            else {
                // No intersection found: Leak
                tmpState.globalHits.nbLeakTotal++;
                if (particleId == 0)RecordLeakPos();
                insertNewParticle = true;
                lastHitFacet=nullptr;
            }
        }

/*#pragma omp critical
            ++allQuit;*/
    } // omp parallel

    return returnVal;
}

void CurrentParticleStatus::IncreaseDistanceCounters(double distanceIncrement) {
    tmpState.globalHits.distTraveled_total += distanceIncrement;
    tmpState.globalHits.distTraveledTotal_fullHitsOnly += distanceIncrement;
    distanceTraveled += distanceIncrement;
}

// Launch a ray from a source facet. The ray 
// direction is chosen according to the desorption type.

bool CurrentParticleStatus::StartFromSource() {
    bool found = false;
    bool foundInMap = false;
    bool reverse;
    size_t mapPositionW, mapPositionH;
    SubprocessFacet *src = nullptr;
    double srcRnd;
    double sumA = 0.0;
    int i = 0, j = 0;
    int nbTry = 0;

    // Check end of simulation
    /*if (model->otfParams.desorptionLimit > 0) {
        if (tmpState.globalHits.globalHits.hit.nbDesorbed >=
            model->otfParams.desorptionLimit / model->otfParams.nbProcess) {
            //lastHitFacet = nullptr; // reset full particle status or go on from where we left
            return false;
        }
    }*/

    // Select source
    srcRnd = randomGenerator.rnd() * model->wp.totalDesorbedMolecules;

    while (!found && j < model->sh.nbSuper) { //Go through superstructures
        i = 0;
        while (!found && i < model->structures[j].facets.size()) { //Go through facets in a structure
            const SubprocessFacet &f = model->structures[j].facets[i];
            if (f.sh.desorbType != DES_NONE) { //there is some kind of outgassing
                if (f.sh.useOutgassingFile) { //Using SynRad-generated outgassing map
                    if (f.sh.totalOutgassing > 0.0) {
                        found = (srcRnd >= sumA) && (srcRnd < (sumA + model->wp.latestMoment * f.sh.totalOutgassing /
                                                                      (1.38E-23 * f.sh.temperature)));
                        if (found) {
                            //look for exact position in map
                            double rndRemainder = (srcRnd - sumA) / model->wp.latestMoment * (1.38E-23 *
                                                                                             f.sh.temperature); //remainder, should be less than f.sh.totalOutgassing
                            /*double sumB = 0.0;
                            for (w = 0; w < f.sh.outgassingMapWidth && !foundInMap; w++) {
                                for (h = 0; h < f.sh.outgassingMapHeight && !foundInMap; h++) {
                                    double cellOutgassing = f.outgassingMap[h*f.sh.outgassingMapWidth + w];
                                    if (cellOutgassing > 0.0) {
                                        foundInMap = (rndRemainder >= sumB) && (rndRemainder < (sumB + cellOutgassing));
                                        if (foundInMap) mapPositionW = w; mapPositionH = h;
                                        sumB += cellOutgassing;
                                    }
                                }
                            }*/
                            double lookupValue = rndRemainder;
                            int outgLowerIndex = my_lower_bound(lookupValue,
                                                                f.outgassingMap); //returns line number AFTER WHICH LINE lookup value resides in ( -1 .. size-2 )
                            outgLowerIndex++;
                            mapPositionH = (size_t) ((double) outgLowerIndex / (double) f.sh.outgassingMapWidth);
                            mapPositionW = (size_t) outgLowerIndex - mapPositionH * f.sh.outgassingMapWidth;
                            foundInMap = true;
                            /*if (!foundInMap) {
                                SetErrorSub("Starting point not found in imported desorption map");
                                return false;
                            }*/
                        }
                        sumA += model->wp.latestMoment * f.sh.totalOutgassing / (1.38E-23 * f.sh.temperature);
                    }
                } //end outgassing file block
                else { //constant or time-dependent outgassing
                    double facetOutgassing =
                            ((f.sh.outgassing_paramId >= 0)
                             ? model->tdParams.IDs[f.sh.IDid].back().second
                             : model->wp.latestMoment * f.sh.outgassing) / (1.38E-23 * f.sh.temperature);
                    found = (srcRnd >= sumA) && (srcRnd < (sumA + facetOutgassing));
                    sumA += facetOutgassing;
                } //end constant or time-dependent outgassing block
            } //end 'there is some kind of outgassing'
            if (!found) i++;
            if (f.sh.is2sided) reverse = randomGenerator.rnd() > 0.5;
            else reverse = false;
        }
        if (!found) j++;
    }
    if (!found) {
        std::cerr << "No starting point, aborting" << std::endl;
        //SetErrorSub("No starting point, aborting");
        return false;
    }
    src = &(model->structures[j].facets[i]);

    lastHitFacet = src;
    //distanceTraveled = 0.0;  //for mean free path calculations
    //particleTime = desorptionStartTime + (desorptionStopTime - desorptionStartTime)*randomGenerator.rnd();
    particleTime = generationTime = GenerateDesorptionTime(src, randomGenerator.rnd());
    lastMomentIndex = 0;
    if (model->wp.useMaxwellDistribution) velocity = GenerateRandomVelocity(src->sh.CDFid, randomGenerator.rnd());
    else
        velocity =
                145.469 * std::sqrt(src->sh.temperature / model->wp.gasMass);  //sqrt(8*R/PI/1000)=145.47
    oriRatio = 1.0;
    if (model->wp.enableDecay) { //decaying gas
        expectedDecayMoment =
                particleTime + model->wp.halfLife * 1.44269 * -log(randomGenerator.rnd()); //1.44269=1/ln2
        //Exponential distribution PDF: probability of 't' life = 1/TAU*exp(-t/TAU) where TAU = half_life/ln2
        //Exponential distribution CDF: probability of life shorter than 't" = 1-exp(-t/TAU)
        //Equation: randomGenerator.rnd()=1-exp(-t/TAU)
        //Solution: t=TAU*-log(1-randomGenerator.rnd()) and 1-randomGenerator.rnd()=randomGenerator.rnd() therefore t=half_life/ln2*-log(randomGenerator.rnd())
    } else {
        expectedDecayMoment = 1e100; //never decay
    }
    //temperature = src->sh.temperature; //Thermalize particle
    nbBounces = 0;
    distanceTraveled = 0;

    found = false; //Starting point within facet

    // Choose a starting point
    while (!found && nbTry < 1000) {
        double u, v;

        if (foundInMap) {
            if (mapPositionW < (src->sh.outgassingMapWidth - 1)) {
                //Somewhere in the middle of the facet
                u = ((double) mapPositionW + randomGenerator.rnd()) / src->outgassingMapWidthD;
            } else {
                //Last element, prevent from going out of facet
                u = ((double) mapPositionW +
                     randomGenerator.rnd() * (src->outgassingMapWidthD - (src->sh.outgassingMapWidth - 1))) /
                    src->outgassingMapWidthD;
            }
            if (mapPositionH < (src->sh.outgassingMapHeight - 1)) {
                //Somewhere in the middle of the facet
                v = ((double) mapPositionH + randomGenerator.rnd()) / src->outgassingMapHeightD;
            } else {
                //Last element, prevent from going out of facet
                v = ((double) mapPositionH +
                     randomGenerator.rnd() * (src->outgassingMapHeightD - (src->sh.outgassingMapHeight - 1))) /
                    src->outgassingMapHeightD;
            }
        } else {
            u = randomGenerator.rnd();
            v = randomGenerator.rnd();
        }
        if (IsInFacet(*src, u, v)) {

            // (U,V) -> (x,y,z)
            position = src->sh.O + u * src->sh.U + v * src->sh.V;
            tmpFacetVars[src->globalId].colU = u;
            tmpFacetVars[src->globalId].colV = v;
            found = true;

        }
        nbTry++;
    }

    if (!found) {
        // Get the center, if the center is not included in the facet, a leak is generated.
        if (foundInMap) {
            //double uLength = sqrt(pow(src->sh.U.x, 2) + pow(src->sh.U.y, 2) + pow(src->sh.U.z, 2));
            //double vLength = sqrt(pow(src->sh.V.x, 2) + pow(src->sh.V.y, 2) + pow(src->sh.V.z, 2));
            double u = ((double) mapPositionW + 0.5) / src->outgassingMapWidthD;
            double v = ((double) mapPositionH + 0.5) / src->outgassingMapHeightD;
            position = src->sh.O + u * src->sh.U + v * src->sh.V;
            tmpFacetVars[src->globalId].colU = u;
            tmpFacetVars[src->globalId].colV = v;
        } else {
            tmpFacetVars[src->globalId].colU = 0.5;
            tmpFacetVars[src->globalId].colV = 0.5;
            position = model->structures[j].facets[i].sh.center;
        }

    }

    if (src->sh.isMoving && model->wp.motionType)
        if (particleId == 0)RecordHit(HIT_MOVING);
        else if (particleId == 0)RecordHit(HIT_DES); //create blue hit point for created particle

    //See docs/theta_gen.png for further details on angular distribution generation
    switch (src->sh.desorbType) {
        case DES_UNIFORM:
            direction = PolarToCartesian(src, std::acos(randomGenerator.rnd()), randomGenerator.rnd() * 2.0 * PI,
                                         reverse);
            break;
        case DES_NONE: //for file-based
        case DES_COSINE:
            direction = PolarToCartesian(src, std::acos(std::sqrt(randomGenerator.rnd())),
                                         randomGenerator.rnd() * 2.0 * PI,
                                         reverse);
            break;
        case DES_COSINE_N:
            direction = PolarToCartesian(src, std::acos(
                    std::pow(randomGenerator.rnd(), 1.0 / (src->sh.desorbTypeN + 1.0))),
                                         randomGenerator.rnd() * 2.0 * PI, reverse);
            break;
        case DES_ANGLEMAP: {
            auto[theta, thetaLowerIndex, thetaOvershoot] = AnglemapGeneration::GenerateThetaFromAngleMap(
                    src->sh.anglemapParams, src->angleMap, randomGenerator.rnd());
            auto phi = AnglemapGeneration::GeneratePhiFromAngleMap(thetaLowerIndex, thetaOvershoot,
                                                                   src->sh.anglemapParams, src->angleMap,
                                                                   tmpState.facetStates[src->globalId].recordedAngleMapPdf,
                                                                   randomGenerator.rnd());
            direction = PolarToCartesian(src, PI - theta, phi,
                                         false); //angle map contains incident angle (between N and source dir) and theta is dir (between N and dest dir)

        }
    }

    // Current structure
    if (src->sh.superIdx == -1) {
        std::ostringstream out;
        out << "Facet " << (src->globalId + 1) << " is in all structures, it shouldn't desorb.";
        //SetErrorSub(out.str().c_str());
        std::cerr << out.str() << std::endl;

        return false;
    }
    structureId = src->sh.superIdx;
    teleportedFrom = -1;

    // Count

    tmpFacetVars[src->globalId].isHit = true;
/*#pragma omp critical
    {
        totalDesorbed++;
    }*/
    tmpState.globalHits.globalHits.hit.nbDesorbed++;
    //nbPHit = 0;

    if (src->sh.isMoving) {
        TreatMovingFacet();
    }

    double ortVelocity =
            velocity * std::abs(Dot(direction, src->sh.N));
    /*src->sh.tmpCounter.hit.nbDesorbed++;
    src->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / ortVelocity; //was 2.0 / ortV
    src->sh.tmpCounter.hit.sum_v_ort += (model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/
    IncreaseFacetCounter(src, particleTime, 0, 1, 0, 2.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    //Desorption doesn't contribute to angular profiles, nor to angle maps
    ProfileFacet(src, particleTime, false, 2.0, 1.0); //was 2.0, 1.0
    //TODO:: if(particleId==0) LogHit(src, model, tmpParticleLog);
    if (/*src->texture && */src->sh.countDes)
        RecordHitOnTexture(src, particleTime, true, 2.0, 1.0); //was 2.0, 1.0
    //if (src->direction && src->sh.countDirection) RecordDirectionVector(src, particleTime);

    // Reset volatile state
    /*if (hasVolatile) {
        for (auto &s : model->structures) {
            for (auto &f : s.facets) {
                f.isReady = true;
            }
        }
    }*/

    found = false;
    return true;
}

std::tuple<double, int, double>
AnglemapGeneration::GenerateThetaFromAngleMap(const AnglemapParams &anglemapParams, const Anglemap &anglemap,
                                              const double lookupValue) {
    //double lookupValue = randomGenerator.rnd();
    int thetaLowerIndex = my_lower_bound(lookupValue,
                                         anglemap.theta_CDF); //returns line number AFTER WHICH LINE lookup value resides in ( -1 .. size-2 )
    double theta, thetaOvershoot;

    if (thetaLowerIndex == -1) { //first half section
        thetaOvershoot = 0.5 + 0.5 * lookupValue / anglemap.theta_CDF[0]; //between 0.5 and 1
        theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot,
                         anglemapParams); //between 0 and the first section end
        return {theta, thetaLowerIndex, thetaOvershoot};
    } else if (thetaLowerIndex == (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes -
                                   1)) { //last half section //can this happen?
        thetaOvershoot = 0.5 * (lookupValue - anglemap.theta_CDF[thetaLowerIndex])
                         / (1.0 - anglemap.theta_CDF[thetaLowerIndex]); //between 0 and 0.5
        theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot,
                         anglemapParams); //between 0 and the first section end
        return {theta, thetaLowerIndex, thetaOvershoot};
    } else { //regular section
        if (/*true || */anglemap.phi_CDFsums[thetaLowerIndex] == anglemap.phi_CDFsums[thetaLowerIndex + 1]) {
            //The pdf's slope is 0, linear interpolation
            thetaOvershoot = (lookupValue - anglemap.theta_CDF[thetaLowerIndex]) /
                             (anglemap.theta_CDF[thetaLowerIndex + 1] - anglemap.theta_CDF[thetaLowerIndex]);
            theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot, anglemapParams);
        } else {
            //2nd degree interpolation
            // y(x) = ax^2 + bx + c
            // c: CDF value at lower index
            // b: pdf value at lower index
            // a: pdf slope at lower index / 2
            // dy := y - c
            // dx := x - [x at lower index]
            // dy = ax^2 + bx
            // dx = ( -b + sqrt(b^2 +4*a*dy) ) / (2a)
            double thetaStep = GetTheta((double) thetaLowerIndex + 1.5, anglemapParams) -
                               GetTheta((double) thetaLowerIndex + 0.5, anglemapParams);
            double c = anglemap.theta_CDF[thetaLowerIndex]; //CDF value at lower index
            double b = (double) anglemap.phi_CDFsums[thetaLowerIndex] / (double) anglemap.theta_CDFsum /
                       thetaStep; //pdf value at lower index
            double a = 0.5 * ((double) (anglemap.phi_CDFsums[thetaLowerIndex + 1]) -
                              (double) anglemap.phi_CDFsums[thetaLowerIndex]) /
                       (double) anglemap.theta_CDFsum / Sqr(thetaStep); //pdf slope at lower index
            double dy = lookupValue - c;

            double dx = (-b + sqrt(Sqr(b) + 4 * a * dy)) /
                        (2 * a); //Since b>=0 it's the + branch of the +- that is valid for us

            thetaOvershoot = dx / thetaStep;
            theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot, anglemapParams);
        }
    }
    return {theta, thetaLowerIndex, thetaOvershoot};
}

/**
* \brief Generates phi angle (azimuth) from angle map
* \param thetaLowerIndex lower index of theta angle of bin in CDF (cummulative distribution function)
* \param thetaOvershoot corresponding to a weight of the previous and next lines
* \param anglemapParams parameters of the angle map
* \param randomGenerator reference to the random number generator (Mersenne Twister)
* \return phi angle
*/

double AnglemapGeneration::GeneratePhiFromAngleMap(const int &thetaLowerIndex, const double &thetaOvershoot,
                                                   const AnglemapParams &anglemapParams, Anglemap &anglemap,
                                                   const std::vector<size_t> &angleMapPDF,
                                                   double lookupValue) {
    //double lookupValue = randomGenerator.rnd();
    if (anglemapParams.phiWidth == 1) return -PI + 2.0 * PI * lookupValue; //special case, uniform phi distribution
    int phiLowerIndex;
    double weigh; //0: take previous theta line, 1: take next theta line, 0..1: interpolate in-between
    if (thetaLowerIndex == -1) { //first theta half section
        lookupValue += anglemap.phi_CDFs[0]; //periodic BCs over -PI...PI, can be larger than 1
        phiLowerIndex = my_lower_bound(lookupValue, &anglemap.phi_CDFs[0],
                                       anglemapParams.phiWidth); //take entirely the phi ditro belonging to first theta
        weigh = thetaOvershoot; // [0.5 - 1], will subtract 0.5 when evaluating thetaIndex
    } else if (thetaLowerIndex ==
               (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1)) { //last theta half section
        lookupValue += anglemap.phi_CDFs[thetaLowerIndex *
                                         anglemapParams.phiWidth]; //periodic BCs over -PI...PI, can be larger than 1
        phiLowerIndex = my_lower_bound(lookupValue, &anglemap.phi_CDFs[thetaLowerIndex * anglemapParams.phiWidth],
                                       anglemapParams.phiWidth); //take entirely the phi ditro belonging to latest theta
        weigh = thetaOvershoot; // [0 - 0.5], will add 0.5 when evaluating thetaIndex
    } else {
        //Here we do a weighing both by the hit sum of the previous and next lines (w1 and w2) and also the weighs of the two lines based on thetaOvershoot (w3 and w4)
        // w1: sum of hits in previous line
        // w2: sum of hits in next line
        // w3: weigh of previous line (1 - thetaOvershoot)
        // w4: weigh of next line     (thetaOvershoot)
        // result: previous value weight: w1*w3 / (w1*w3 + w2*w4)
        //         next     value weight: w2*w4 / (w1*w3 + w2*w4) <- this will be the input for weighed_lower_bound

        double div;
        div = ((double) anglemap.phi_CDFsums[thetaLowerIndex] * (1.0 - thetaOvershoot) +
               (double) anglemap.phi_CDFsums[thetaLowerIndex + 1] * thetaOvershoot); // (w1*w3 + w2*w4)
        if (div > 0.0) {
            weigh = (thetaOvershoot * (double) anglemap.phi_CDFsums[thetaLowerIndex + 1]) /
                    div;    //      w2*w4 / (w1*w3 + w2*w4)
        } else {
            weigh = thetaOvershoot;
        }
        lookupValue += Weigh((double) anglemap.phi_CDFs[thetaLowerIndex * anglemapParams.phiWidth],
                             (double) anglemap.phi_CDFs[(thetaLowerIndex + 1) * anglemapParams.phiWidth], weigh);
        phiLowerIndex = weighed_lower_bound_X(lookupValue, weigh,
                                              &anglemap.phi_CDFs[thetaLowerIndex * anglemapParams.phiWidth],
                                              &anglemap.phi_CDFs[(thetaLowerIndex + 1) * anglemapParams.phiWidth],
                                              anglemapParams.phiWidth);
    }

    double phi, phiOvershoot;
    double thetaIndex = (double) thetaLowerIndex + 0.5 + weigh;
    if (phiLowerIndex == -1) { //first half section
        DEBUG_BREAK; //should not happen since we shifted the lookup value with first value
        phiOvershoot =
                0.5 + 0.5 * lookupValue / GetPhiCDFValue(thetaIndex, 0, anglemapParams, anglemap); //between 0.5 and 1
        phi = GetPhi((double) phiLowerIndex + 0.5 + phiOvershoot, anglemapParams); //between 0 and the first section end
    }
        /*else if (phiLowerIndex == (anglemapParams.phiWidth - 1)) { //last half section
            phiOvershoot = 0.5 * (lookupValue - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams) )
                / (1.0 - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams)); //between 0 and 0.5
            phi = GetPhi((double)phiLowerIndex + 0.5 + phiOvershoot, anglemapParams); //between 0 and the first section end
        }*/
    else { //regular or last section
        if (/*true ||*/ GetPhipdfValue(thetaIndex, phiLowerIndex, anglemapParams, angleMapPDF) ==
                        GetPhipdfValue(thetaIndex, phiLowerIndex + 1, anglemapParams, angleMapPDF)) {
            //The pdf's slope is 0, linear interpolation
            phiOvershoot = (lookupValue - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap))
                           / (GetPhiCDFValue(thetaIndex, phiLowerIndex + 1, anglemapParams, anglemap) -
                              GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap));
            phi = GetPhi((double) phiLowerIndex + 0.5 + phiOvershoot, anglemapParams);
        } else {

            //2nd degree interpolation
            // y(x) = ax^2 + bx + c
            // c: CDF value at lower index
            // b: pdf value at lower index
            // a: pdf slope at lower index / 2
            // dy := y - c
            // dx := x - [x at lower index]
            // dy = ax^2 + bx
            // dx = ( -b + sqrt(b^2 +4*a*dy) ) / (2a)
            double phiStep = 2.0 * PI / (double) anglemapParams.phiWidth;
            double c = GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap); //CDF value at lower index
            double b = GetPhipdfValue(thetaIndex, phiLowerIndex, anglemapParams, angleMapPDF) /
                       GetPhiCDFSum(thetaIndex, anglemapParams, anglemap) / phiStep; //pdf value at lower index
            double a = 0.5 * (GetPhipdfValue(thetaIndex, phiLowerIndex + 1, anglemapParams, angleMapPDF) -
                              GetPhipdfValue(thetaIndex, phiLowerIndex, anglemapParams, angleMapPDF)) /
                       GetPhiCDFSum(thetaIndex, anglemapParams, anglemap) / Sqr(phiStep); //pdf slope at lower index
            double dy = lookupValue - c;

            double D = Sqr(b) + 4 * a *
                                dy; //Discriminant. In rare cases it might be slightly negative, then fall back to linear interpolation:
            if (D < 0) {
                phiOvershoot = (lookupValue - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams,
                                                             anglemap))
                               / (GetPhiCDFValue(thetaIndex, (int) IDX(phiLowerIndex + 1, anglemapParams.phiWidth),
                                                 anglemapParams, anglemap) -
                                  GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap));
            } else {
                double dx = (-b + sqrt(Sqr(b) + 4 * a * dy)) /
                            (2 * a); //Since b>=0 it's the + branch of the +- that is valid for us
                phiOvershoot = dx / phiStep;
            }
            phi = GetPhi((double) phiLowerIndex + 0.5 + phiOvershoot, anglemapParams);
        }
    }
    assert(phi > -PI && phi < PI);
    return phi;
}

/**
* \brief Converts from index to theta value
* \param thetaIndex theta index 
* \param anglemapParams parameters of the angle map
* \return theta angle
*/

double AnglemapGeneration::GetTheta(const double &thetaIndex, const AnglemapParams &anglemapParams) {
    if ((size_t) (thetaIndex) < anglemapParams.thetaLowerRes) { // 0 < theta < limit
        return anglemapParams.thetaLimit * (thetaIndex) / (double) anglemapParams.thetaLowerRes;
    } else { // limit < theta < PI/2
        return anglemapParams.thetaLimit +
               (PI / 2.0 - anglemapParams.thetaLimit) * (thetaIndex - (double) anglemapParams.thetaLowerRes) /
               (double) anglemapParams.thetaHigherRes;
    }
}

/**
* \brief makes phiIndex circular and converts from index to -pi...pi
* \param phiIndex phi index
* \param anglemapParams parameters of the angle map
* \return phi angle
*/

double AnglemapGeneration::GetPhi(const double &phiIndex, const AnglemapParams &anglemapParams)
//makes phiIndex circular and converts from index to -pi...pi
{
    double width = (double) anglemapParams.phiWidth;
    double correctedIndex = (phiIndex < width) ? phiIndex : phiIndex - width;
    return -PI + 2.0 * PI * correctedIndex / width;
}

double
AnglemapGeneration::GetPhipdfValue(const double &thetaIndex, const int &phiLowerIndex,
                                   const AnglemapParams &anglemapParams, const std::vector<size_t> &angleMapPDF)
//phiLowerIndex is circularized
{
    if (thetaIndex < 0.5) {
        return (double) angleMapPDF[IDX(phiLowerIndex, anglemapParams.phiWidth)];
    } else if (thetaIndex > (double) (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes) - 0.5) {
        return (double) angleMapPDF[
                anglemapParams.phiWidth * (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1) +
                IDX(phiLowerIndex, anglemapParams.phiWidth)];
    } else {
        size_t thetaLowerIndex = (size_t) (thetaIndex - 0.5);
        double thetaOvershoot = thetaIndex - 0.5 - (double) thetaLowerIndex;
        double valueFromLowerpdf = (double) angleMapPDF[anglemapParams.phiWidth * thetaLowerIndex +
                                                        IDX(phiLowerIndex, anglemapParams.phiWidth)];
        double valueFromHigherpdf = (double) angleMapPDF[anglemapParams.phiWidth * (thetaLowerIndex + 1) +
                                                         IDX(phiLowerIndex, anglemapParams.phiWidth)];
        return Weigh(valueFromLowerpdf, valueFromHigherpdf, thetaOvershoot);
    }
}

/**
* \brief Get phi value from cummulative density function
* \param thetaIndex theta index
* \param phiLowerIndex lower index of the bin
* \param anglemapParams parameters of the angle map
* \return phi cdf value
*/

double
AnglemapGeneration::GetPhiCDFValue(const double &thetaIndex, const int &phiLowerIndex,
                                   const AnglemapParams &anglemapParams, const Anglemap &anglemap) {
    if (thetaIndex < 0.5) {
        return (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[phiLowerIndex] : 1.0 +
                                                                                              anglemap.phi_CDFs[0];
    } else if (thetaIndex > (double) (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes) - 0.5) {
        return (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[
                anglemapParams.phiWidth * (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1) +
                phiLowerIndex] : 1.0 + anglemap.phi_CDFs[anglemapParams.phiWidth *
                                                         (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes -
                                                          1)];
    } else {
        size_t thetaLowerIndex = (size_t) (thetaIndex - 0.5);
        double thetaOvershoot = thetaIndex - 0.5 - (double) thetaLowerIndex;
        double valueFromLowerCDF = (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[
                anglemapParams.phiWidth * thetaLowerIndex + phiLowerIndex] : 1.0 +
                                                                             anglemap.phi_CDFs[anglemapParams.phiWidth *
                                                                                               (thetaLowerIndex)];
        double valueFromHigherCDF = (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[
                anglemapParams.phiWidth * (thetaLowerIndex + 1) + phiLowerIndex] : 1.0 +
                                                                                   anglemap.phi_CDFs[
                                                                                           anglemapParams.phiWidth *
                                                                                           (thetaLowerIndex + 1)];
        return Weigh(valueFromLowerCDF, valueFromHigherCDF, thetaOvershoot);
    }

}

/**
* \brief Get phi value from cummulative density function
* \param thetaIndex theta index
* \param anglemapParams parameters of the angle map
* \return phi cdf summed value
*/

double AnglemapGeneration::GetPhiCDFSum(const double &thetaIndex, const AnglemapParams &anglemapParams,
                                        const Anglemap &anglemap) {
    if (thetaIndex < 0.5) {
        return (double) anglemap.phi_CDFsums[0];
    } else if (thetaIndex > (double) (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes) - 0.5) {
        return (double) anglemap.phi_CDFsums[anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1];
    } else {
        size_t thetaLowerIndex = (size_t) (thetaIndex - 0.5);
        double thetaOvershoot = thetaIndex - 0.5 - (double) thetaLowerIndex;
        double valueFromLowerSum = (double) anglemap.phi_CDFsums[thetaLowerIndex];
        double valueFromHigherSum = (double) anglemap.phi_CDFsums[thetaLowerIndex + 1];
        return Weigh(valueFromLowerSum, valueFromHigherSum, thetaOvershoot);
    }
}

/**
* \brief Perform a bounce from a facet by logging the hit and sometimes relaunching it
* \param iFacet facet corresponding to the bounce event
*/

void CurrentParticleStatus::PerformBounce(SubprocessFacet *iFacet) {

    bool revert = false;
    tmpState.globalHits.globalHits.hit.nbMCHit++; //global
    tmpState.globalHits.globalHits.hit.nbHitEquiv += oriRatio;

    // Handle super structure link facet. Can be
    if (iFacet->sh.superDest) {
        IncreaseFacetCounter(iFacet, particleTime, 1, 0, 0, 0, 0);
        structureId = iFacet->sh.superDest - 1;
        if (iFacet->sh.isMoving) { //A very special case where link facets can be used as transparent but moving facets
            if (particleId == 0)RecordHit(HIT_MOVING);
            TreatMovingFacet();
        } else {
            // Count this hit as a transparent pass
            if (particleId == 0)RecordHit(HIT_TRANS);
        }
        //TODO: if(particleId==0)LogHit(iFacet, model, tmpParticleLog);
        ProfileFacet(iFacet, particleTime, true, 2.0, 2.0);
        if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);
        if (/*iFacet->texture &&*/ iFacet->sh.countTrans)
            RecordHitOnTexture(iFacet, particleTime, true, 2.0, 2.0);
        if (/*iFacet->direction &&*/ iFacet->sh.countDirection)
            RecordDirectionVector(iFacet, particleTime);

        return;

    }

    // Handle volatile facet
    if (iFacet->sh.isVolatile) {

        if (iFacet->isReady) {
            IncreaseFacetCounter(iFacet, particleTime, 0, 0, 1, 0, 0);
            iFacet->isReady = false;
            //TODO::if(particleId==0)LogHit(iFacet, model, tmpParticleLog);
            ProfileFacet(iFacet, particleTime, true, 2.0, 1.0);
            if (/*iFacet->texture && */iFacet->sh.countAbs)
                RecordHitOnTexture(iFacet, particleTime, true, 2.0, 1.0);
            if (/*iFacet->direction && */iFacet->sh.countDirection)
                RecordDirectionVector(iFacet, particleTime);
        }
        return;

    }

    if (iFacet->sh.is2sided) {
        // We may need to revert normal in case of 2 sided hit
        revert = Dot(direction, iFacet->sh.N) > 0.0;
    }

    //Texture/Profile incoming hit


    //Register (orthogonal) velocity
    double ortVelocity =
            velocity * std::abs(Dot(direction, iFacet->sh.N));

    /*iFacet->sh.tmpCounter.hit.nbMCHit++; //hit facet
    iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 1.0 / ortVelocity;
    iFacet->sh.tmpCounter.hit.sum_v_ort += (model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/

    IncreaseFacetCounter(iFacet, particleTime, 1, 0, 0, 1.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    nbBounces++;
    if (/*iFacet->texture &&*/ iFacet->sh.countRefl)
        RecordHitOnTexture(iFacet, particleTime, true, 1.0, 1.0);
    if (/*iFacet->direction &&*/ iFacet->sh.countDirection)
        RecordDirectionVector(iFacet, particleTime);
    //TODO::if(particleId==0)LogHit(iFacet, model, tmpParticleLog);
    ProfileFacet(iFacet, particleTime, true, 1.0, 1.0);
    if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);

    // Relaunch particle
    UpdateVelocity(iFacet);
    //Sojourn time
    if (iFacet->sh.enableSojournTime) {
        double A = exp(-iFacet->sh.sojournE / (8.31 * iFacet->sh.temperature));
        particleTime += -log(randomGenerator.rnd()) / (A * iFacet->sh.sojournFreq);
    }

    if (iFacet->sh.reflection.diffusePart > 0.999999) { //Speedup branch for most common, diffuse case
        direction = PolarToCartesian(iFacet, std::acos(std::sqrt(randomGenerator.rnd())),
                                     randomGenerator.rnd() * 2.0 * PI,
                                     revert);
    } else {
        double reflTypeRnd = randomGenerator.rnd();
        if (reflTypeRnd < iFacet->sh.reflection.diffusePart) {
            //diffuse reflection
            //See docs/theta_gen.png for further details on angular distribution generation
            direction = PolarToCartesian(iFacet, std::acos(std::sqrt(randomGenerator.rnd())),
                                         randomGenerator.rnd() * 2.0 * PI,
                                         revert);
        } else if (reflTypeRnd < (iFacet->sh.reflection.diffusePart + iFacet->sh.reflection.specularPart)) {
            //specular reflection
            auto[inTheta, inPhi] = CartesianToPolar(direction, iFacet->sh.nU, iFacet->sh.nV,
                                                    iFacet->sh.N);
            direction = PolarToCartesian(iFacet, PI - inTheta, inPhi, false);

        } else {
            //Cos^N reflection
            direction = PolarToCartesian(iFacet, std::acos(
                    std::pow(randomGenerator.rnd(), 1.0 / (iFacet->sh.reflection.cosineExponent + 1.0))),
                                         randomGenerator.rnd() * 2.0 * PI, revert);
        }
    }

    if (iFacet->sh.isMoving) {
        TreatMovingFacet();
    }

    //Texture/Profile outgoing particle
    //Register outgoing velocity
    ortVelocity = velocity * std::abs(Dot(direction, iFacet->sh.N));

    /*iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 1.0 / ortVelocity;
    iFacet->sh.tmpCounter.hit.sum_v_ort += (model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/
    IncreaseFacetCounter(iFacet, particleTime, 0, 0, 0, 1.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    if (/*iFacet->texture &&*/ iFacet->sh.countRefl)
        RecordHitOnTexture(iFacet, particleTime, false, 1.0,
                           1.0); //count again for outward velocity
    ProfileFacet(iFacet, particleTime, false, 1.0, 1.0);
    //no direction count on outgoing, neither angle map

    if (iFacet->sh.isMoving && model->wp.motionType) {
        if (particleId == 0)
            RecordHit(HIT_MOVING);
    } else if (particleId == 0)RecordHit(HIT_REF);
    lastHitFacet = iFacet;
    //nbPHit++;
}

/*void Simulation::PerformTransparentPass(SubprocessFacet *iFacet) { //disabled, caused finding hits with the same facet
    *//*double directionFactor = abs(DOT3(
        direction.x, direction.y, direction.z,
        iFacet->sh.N.x, iFacet->sh.N.y, iFacet->sh.N.z));
    iFacet->sh.tmpCounter.hit.nbMCHit++;
    iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / (velocity*directionFactor);
    iFacet->sh.tmpCounter.hit.sum_v_ort += 2.0*(model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*velocity*directionFactor;
    iFacet->isHit = true;
    if (iFacet->texture && iFacet->sh.countTrans) RecordHitOnTexture(iFacet, particleTime + iFacet->colDist / 100.0 / velocity,
        true, 2.0, 2.0);
    if (iFacet->direction && iFacet->sh.countDirection) RecordDirectionVector(iFacet, particleTime + iFacet->colDist / 100.0 / velocity);
    ProfileFacet(iFacet, particleTime + iFacet->colDist / 100.0 / velocity,
        true, 2.0, 2.0);
    RecordHit(HIT_TRANS);
    lastHit = iFacet;*//*
}*/

void CurrentParticleStatus::RecordAbsorb(SubprocessFacet *iFacet) {
    tmpState.globalHits.globalHits.hit.nbMCHit++; //global
    tmpState.globalHits.globalHits.hit.nbHitEquiv += oriRatio;
    tmpState.globalHits.globalHits.hit.nbAbsEquiv += oriRatio;

    RecordHistograms(iFacet);

    if (particleId == 0)RecordHit(HIT_ABS);
    double ortVelocity =
            velocity * std::abs(Dot(direction, iFacet->sh.N));
    IncreaseFacetCounter(iFacet, particleTime, 1, 0, 1, 2.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    //TODO: if(particleId==0)LogHit(iFacet, model, tmpParticleLog);
    ProfileFacet(iFacet, particleTime, true, 2.0, 1.0); //was 2.0, 1.0
    if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);
    if (/*iFacet->texture &&*/ iFacet->sh.countAbs)
        RecordHitOnTexture(iFacet, particleTime, true, 2.0, 1.0); //was 2.0, 1.0
    if (/*iFacet->direction &&*/ iFacet->sh.countDirection)
        RecordDirectionVector(iFacet, particleTime);
}

void CurrentParticleStatus::RecordHistograms(SubprocessFacet *iFacet) {
    //Record in global and facet histograms
    size_t binIndex;

    auto &tmpGlobalHistograms = tmpState.globalHistograms;
    auto &facetHistogram = tmpState.facetStates[iFacet->globalId].momentResults;
    auto &globHistParams = model->wp.globalHistogramParams;
    auto &facHistParams = iFacet->sh.facetHistogramParams;

    int moments[2] = {0, -1};
    moments[1] = LookupMomentIndex(particleTime, model->tdParams.moments,
                                   lastMomentIndex);
    for (auto &m : moments) {
        if (m <= 0) return;

        if (globHistParams.recordBounce) {
            binIndex = Min(nbBounces / globHistParams.nbBounceBinsize,
                           globHistParams.GetBounceHistogramSize() - 1);
            tmpGlobalHistograms[m].nbHitsHistogram[binIndex] += oriRatio;
        }
        if (globHistParams.recordDistance) {
            binIndex = Min(static_cast<size_t>(distanceTraveled /
                                               globHistParams.distanceBinsize),
                           globHistParams.GetDistanceHistogramSize() - 1);
            tmpGlobalHistograms[m].distanceHistogram[binIndex] += oriRatio;
        }
        if (globHistParams.recordTime) {
            binIndex = Min(static_cast<size_t>((particleTime - generationTime) /
                                               globHistParams.timeBinsize),
                           globHistParams.GetTimeHistogramSize() - 1);
            tmpGlobalHistograms[m].timeHistogram[binIndex] += oriRatio;
        }
        if (facHistParams.recordBounce) {
            binIndex = Min(nbBounces / facHistParams.nbBounceBinsize,
                           facHistParams.GetBounceHistogramSize() - 1);
            facetHistogram[m].histogram.nbHitsHistogram[binIndex] += oriRatio;
        }
        if (facHistParams.recordDistance) {
            binIndex = Min(static_cast<size_t>(distanceTraveled /
                                               facHistParams.distanceBinsize),
                           facHistParams.GetDistanceHistogramSize() - 1);
            facetHistogram[m].histogram.distanceHistogram[binIndex] += oriRatio;
        }
        if (facHistParams.recordTime) {
            binIndex = Min(static_cast<size_t>((particleTime - generationTime) /
                                               facHistParams.timeBinsize),
                           facHistParams.GetTimeHistogramSize() - 1);
            facetHistogram[m].histogram.timeHistogram[binIndex] += oriRatio;
        }
    }
}

void
CurrentParticleStatus::RecordHitOnTexture(const SubprocessFacet *f, double time, bool countHit, double velocity_factor,
                                          double ortSpeedFactor) {

    size_t tu = (size_t) (tmpFacetVars[f->globalId].colU * f->sh.texWidthD);
    size_t tv = (size_t) (tmpFacetVars[f->globalId].colV * f->sh.texHeightD);
    size_t add = tu + tv * (f->sh.texWidth);
    double ortVelocity = (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * velocity *
                         std::abs(Dot(direction,
                                      f->sh.N)); //surface-orthogonal velocity component

    {
        TextureCell &texture = tmpState.facetStates[f->globalId].momentResults[0].texture[add];
        if (countHit) texture.countEquiv += oriRatio;
        texture.sum_1_per_ort_velocity +=
                oriRatio * velocity_factor / ortVelocity;
        texture.sum_v_ort_per_area += oriRatio * ortSpeedFactor * ortVelocity *
                                      f->textureCellIncrements[add]; // sum ortho_velocity[m/s] / cell_area[cm2]
    }
    int m = -1;
    if ((m = LookupMomentIndex(time, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = m - 1;
        TextureCell &texture = tmpState.facetStates[f->globalId].momentResults[m].texture[add];
        if (countHit) texture.countEquiv += oriRatio;
        texture.sum_1_per_ort_velocity +=
                oriRatio * velocity_factor / ortVelocity;
        texture.sum_v_ort_per_area += oriRatio * ortSpeedFactor * ortVelocity *
                                      f->textureCellIncrements[add]; // sum ortho_velocity[m/s] / cell_area[cm2]
    }
}

void CurrentParticleStatus::RecordDirectionVector(const SubprocessFacet *f, double time) {
    size_t tu = (size_t) (tmpFacetVars[f->globalId].colU * f->sh.texWidthD);
    size_t tv = (size_t) (tmpFacetVars[f->globalId].colV * f->sh.texHeightD);
    size_t add = tu + tv * (f->sh.texWidth);

    {
        DirectionCell &dirCell = tmpState.facetStates[f->globalId].momentResults[0].direction[add];
        dirCell.dir += oriRatio * direction * velocity;
        dirCell.count++;
    }
    int m = -1;
    if ((m = LookupMomentIndex(time, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = m - 1;
        DirectionCell &dirCell = tmpState.facetStates[f->globalId].momentResults[m].direction[add];
        dirCell.dir += oriRatio * direction * velocity;
        dirCell.count++;
    }

}

void
CurrentParticleStatus::ProfileFacet(const SubprocessFacet *f, double time, bool countHit, double velocity_factor,
                                    double ortSpeedFactor) {

    size_t nbMoments = model->tdParams.moments.size();
    int m = LookupMomentIndex(time, model->tdParams.moments, lastMomentIndex);
    if (countHit && f->sh.profileType == PROFILE_ANGULAR) {
        double dot = Dot(f->sh.N, direction);
        double theta = std::acos(std::abs(dot));     // Angle to normal (PI/2 => PI)
        size_t pos = (size_t) (theta / (PI / 2) * ((double) PROFILE_SIZE)); // To Grad
        Saturate(pos, 0, PROFILE_SIZE - 1);

        tmpState.facetStates[f->globalId].momentResults[0].profile[pos].countEquiv += oriRatio;
        if (m > 0) {
            lastMomentIndex = m - 1;
            tmpState.facetStates[f->globalId].momentResults[m].profile[pos].countEquiv += oriRatio;
        }
    } else if (f->sh.profileType == PROFILE_U || f->sh.profileType == PROFILE_V) {
        size_t pos = (size_t) (
                (f->sh.profileType == PROFILE_U ? tmpFacetVars[f->globalId].colU : tmpFacetVars[f->globalId].colV) *
                (double) PROFILE_SIZE);
        if (pos >= 0 && pos < PROFILE_SIZE) {
            {
                ProfileSlice &profile = tmpState.facetStates[f->globalId].momentResults[0].profile[pos];
                if (countHit) profile.countEquiv += oriRatio;
                double ortVelocity = velocity *
                                     std::abs(Dot(f->sh.N, direction));
                profile.sum_1_per_ort_velocity +=
                        oriRatio * velocity_factor / ortVelocity;
                profile.sum_v_ort += oriRatio * ortSpeedFactor *
                                     (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity;
            }
            if (m > 0) {
                lastMomentIndex = m - 1;
                ProfileSlice &profile = tmpState.facetStates[f->globalId].momentResults[m].profile[pos];
                if (countHit) profile.countEquiv += oriRatio;
                double ortVelocity = velocity *
                                     std::abs(Dot(f->sh.N, direction));
                profile.sum_1_per_ort_velocity +=
                        oriRatio * velocity_factor / ortVelocity;
                profile.sum_v_ort += oriRatio * ortSpeedFactor *
                                     (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity;
            }
        }
    } else if (countHit && (f->sh.profileType == PROFILE_VELOCITY || f->sh.profileType == PROFILE_ORT_VELOCITY ||
                            f->sh.profileType == PROFILE_TAN_VELOCITY)) {
        double dot;
        if (f->sh.profileType == PROFILE_VELOCITY) {
            dot = 1.0;
        } else if (f->sh.profileType == PROFILE_ORT_VELOCITY) {
            dot = std::abs(Dot(f->sh.N, direction));  //cos(theta) as "dot" value
        } else { //Tangential
            dot = std::sqrt(1 - Sqr(std::abs(Dot(f->sh.N, direction))));  //tangential
        }
        size_t pos = (size_t) (dot * velocity / f->sh.maxSpeed *
                               (double) PROFILE_SIZE); //"dot" default value is 1.0
        if (pos >= 0 && pos < PROFILE_SIZE) {
            tmpState.facetStates[f->globalId].momentResults[0].profile[pos].countEquiv += oriRatio;
            if (m > 0) {
                lastMomentIndex = m - 1;
                tmpState.facetStates[f->globalId].momentResults[m].profile[pos].countEquiv += oriRatio;
            }
        }
    }
}

void
CurrentParticleStatus::LogHit(SubprocessFacet *f, std::vector<ParticleLoggerItem> &tmpParticleLog) {
    //if(omp_get_thread_num() != 0) return; // only let 1 thread update
    if (model->otfParams.enableLogging &&
        model->otfParams.logFacetId == f->globalId &&
        tmpParticleLog.size() < (model->otfParams.logLimit / model->otfParams.nbProcess)) {
        ParticleLoggerItem log;
        log.facetHitPosition = Vector2d(tmpFacetVars[f->globalId].colU, tmpFacetVars[f->globalId].colV);
        std::tie(log.hitTheta, log.hitPhi) = CartesianToPolar(direction, f->sh.nU, f->sh.nV,
                                                              f->sh.N);
        log.oriRatio = oriRatio;
        log.particleDecayMoment = expectedDecayMoment;
        log.time = particleTime;
        log.velocity = velocity;
        tmpParticleLog.push_back(log);
    }
}

void CurrentParticleStatus::RecordAngleMap(const SubprocessFacet *collidedFacet) {
    auto[inTheta, inPhi] = CartesianToPolar(direction, collidedFacet->sh.nU,
                                            collidedFacet->sh.nV, collidedFacet->sh.N);
    if (inTheta > PI / 2.0)
        inTheta = std::abs(
                PI - inTheta); //theta is originally respective to N, but we'd like the angle between 0 and PI/2
    bool countTheta = true;
    size_t thetaIndex;
    if (inTheta < collidedFacet->sh.anglemapParams.thetaLimit) {
        if (collidedFacet->sh.anglemapParams.thetaLowerRes > 0) {
            thetaIndex = (size_t) (inTheta / collidedFacet->sh.anglemapParams.thetaLimit *
                                   (double) collidedFacet->sh.anglemapParams.thetaLowerRes);
        } else {
            countTheta = false;
        }
    } else {
        if (collidedFacet->sh.anglemapParams.thetaHigherRes > 0) {
            thetaIndex = collidedFacet->sh.anglemapParams.thetaLowerRes +
                         (size_t) ((inTheta - collidedFacet->sh.anglemapParams.thetaLimit)
                                   / (PI / 2.0 - collidedFacet->sh.anglemapParams.thetaLimit) *
                                   (double) collidedFacet->sh.anglemapParams.thetaHigherRes);
        } else {
            countTheta = false;
        }
    }
    if (countTheta) {
        size_t phiIndex = (size_t) ((inPhi + 3.1415926) / (2.0 * PI) *
                                    (double) collidedFacet->sh.anglemapParams.phiWidth); //Phi: -PI..PI , and shifting by a number slightly smaller than PI to store on interval [0,2PI[

        auto &angleMap = tmpState.facetStates[collidedFacet->globalId].recordedAngleMapPdf;
        angleMap[thetaIndex * collidedFacet->sh.anglemapParams.phiWidth + phiIndex]++;
    }
}

void CurrentParticleStatus::UpdateVelocity(const SubprocessFacet *collidedFacet) {
    if (collidedFacet->sh.accomodationFactor > 0.9999) { //speedup for the most common case: perfect thermalization
        if (model->wp.useMaxwellDistribution)
            velocity = GenerateRandomVelocity(collidedFacet->sh.CDFid, randomGenerator.rnd());
        else
            velocity =
                    145.469 * std::sqrt(collidedFacet->sh.temperature / model->wp.gasMass);
    } else {
        double oldSpeed2 = pow(velocity, 2);
        double newSpeed2;
        if (model->wp.useMaxwellDistribution)
            newSpeed2 = pow(GenerateRandomVelocity(collidedFacet->sh.CDFid,
                                                   randomGenerator.rnd()), 2);
        else newSpeed2 = /*145.469*/ 29369.939 * (collidedFacet->sh.temperature / model->wp.gasMass);
        //sqrt(29369)=171.3766= sqrt(8*R*1000/PI)*3PI/8, that is, the constant part of the v_avg=sqrt(8RT/PI/m/0.001)) found in literature, multiplied by
        //the corrective factor of 3PI/8 that accounts for moving from volumetric speed distribution to wall collision speed distribution
        velocity = std::sqrt(
                oldSpeed2 + (newSpeed2 - oldSpeed2) * collidedFacet->sh.accomodationFactor);
    }
}

double CurrentParticleStatus::GenerateRandomVelocity(int CDFId, const double rndVal) {
    //return FastLookupY(randomGenerator.rnd(),CDFs[CDFId],false);
    //double r = randomGenerator.rnd();
    double v = InterpolateX(rndVal, model->tdParams.CDFs[CDFId], false, false, true); //Allow extrapolate
    return v;
}

double CurrentParticleStatus::GenerateDesorptionTime(const SubprocessFacet *src, const double rndVal) {
    if (src->sh.outgassing_paramId >= 0) { //time-dependent desorption
        return InterpolateX(rndVal * model->tdParams.IDs[src->sh.IDid].back().second, model->tdParams.IDs[src->sh.IDid],
                            false, false, true); //allow extrapolate
    } else {
        return rndVal * model->wp.latestMoment; //continous desorption between 0 and latestMoment
    }
}

double SimulationModel::GetStickingAt(SubprocessFacet *f, double time) const {
    if (f->sh.sticking_paramId == -1) //constant sticking
        return f->sh.sticking;
    else {
        auto &par = tdParams.parameters[f->sh.sticking_paramId];
        return InterpolateY(time, par.GetValues(), par.logXinterp, par.logYinterp, false);
    }
    //else return model->tdParams.parameters[f->sh.sticking_paramId].InterpolateY(time, false);
}

double SimulationModel::GetOpacityAt(SubprocessFacet *f, double time) const {
    if (f->sh.opacity_paramId == -1) //constant sticking
        return f->sh.opacity;
    else {
        auto &par = tdParams.parameters[f->sh.opacity_paramId];
        return InterpolateY(time, par.GetValues(), par.logXinterp, par.logYinterp, false);
    }
    //else return this->tdParams.parameters[f->sh.opacity_paramId].InterpolateY(time, false);
}

/**
* \brief Updates particle direction and velocity if we are dealing with a moving facet (translated or rotated)
*/


void CurrentParticleStatus::TreatMovingFacet() {
    Vector3d localVelocityToAdd;
    if (model->wp.motionType == 1) { //Translation
        localVelocityToAdd = model->wp.motionVector2; //Fixed translational vector
    } else if (model->wp.motionType == 2) { //Rotation
        Vector3d distanceVector = 0.01 * (position -
                                          model->wp.motionVector1); //distance from base, with cm->m conversion, motionVector1 is rotation base point
        localVelocityToAdd = CrossProduct(model->wp.motionVector2, distanceVector); //motionVector2 is rotation axis
    }
    Vector3d oldVelocity, newVelocity;
    oldVelocity = direction * velocity;
    newVelocity = oldVelocity + localVelocityToAdd;
    direction = newVelocity.Normalized();
    velocity = newVelocity.Norme();
}

/**
* \brief Increase facet counter on a hit, pass etc.
* \param f source facet
* \param time simulation time
* \param hit amount of hits to add
* \param desorb amount of desorptions to add
* \param absorb amount of absorptions to add
* \param sum_1_per_v reciprocals of orthogonal speed components to add
* \param sum_v_ort orthogonal momentum change to add
*/
void
CurrentParticleStatus::IncreaseFacetCounter(const SubprocessFacet *f, double time, size_t hit, size_t desorb,
                                            size_t absorb,
                                            double sum_1_per_v, double sum_v_ort) {
    const double hitEquiv = static_cast<double>(hit) * oriRatio;
    {
        FacetHitBuffer &hits = tmpState.facetStates[f->globalId].momentResults[0].hits;
        hits.hit.nbMCHit += hit;
        hits.hit.nbHitEquiv += hitEquiv;
        hits.hit.nbDesorbed += desorb;
        hits.hit.nbAbsEquiv += static_cast<double>(absorb) * oriRatio;
        hits.hit.sum_1_per_ort_velocity += oriRatio * sum_1_per_v;
        hits.hit.sum_v_ort += oriRatio * sum_v_ort;
        hits.hit.sum_1_per_velocity += (hitEquiv + static_cast<double>(desorb)) / velocity;
    }
    int m = -1;
    if ((m = LookupMomentIndex(time, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = m - 1;
        FacetHitBuffer &hits = tmpState.facetStates[f->globalId].momentResults[m].hits;
        hits.hit.nbMCHit += hit;
        hits.hit.nbHitEquiv += hitEquiv;
        hits.hit.nbDesorbed += desorb;
        hits.hit.nbAbsEquiv += static_cast<double>(absorb) * oriRatio;
        hits.hit.sum_1_per_ort_velocity += oriRatio * sum_1_per_v;
        hits.hit.sum_v_ort += oriRatio * sum_v_ort;
        hits.hit.sum_1_per_velocity += (hitEquiv + static_cast<double>(desorb)) / velocity;
    }
}

void CurrentParticleStatus::RegisterTransparentPass(SubprocessFacet *facet) {
    double directionFactor = std::abs(Dot(direction, facet->sh.N));
    IncreaseFacetCounter(facet, particleTime +
                                tmpFacetVars[facet->globalId].colDistTranspPass / 100.0 / velocity, 1, 0, 0,
                         2.0 / (velocity * directionFactor),
                         2.0 * (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * velocity *
                         directionFactor);

    tmpFacetVars[facet->globalId].isHit = true;
    if (/*facet->texture &&*/ facet->sh.countTrans) {
        RecordHitOnTexture(facet, particleTime +
                                  tmpFacetVars[facet->globalId].colDistTranspPass / 100.0 / velocity,
                           true, 2.0, 2.0);
    }
    if (/*facet->direction &&*/ facet->sh.countDirection) {
        RecordDirectionVector(facet, particleTime +
                                     tmpFacetVars[facet->globalId].colDistTranspPass / 100.0 / velocity);
    }
    //TODO::if(particleId==0)LogHit(facet, model, tmpParticleLog);
    ProfileFacet(facet, particleTime +
                        tmpFacetVars[facet->globalId].colDistTranspPass / 100.0 /
                        velocity,
                 true, 2.0, 2.0);
    if (facet->sh.anglemapParams.record) RecordAngleMap(facet);
}