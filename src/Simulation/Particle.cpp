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


#include <IntersectAABB_shared.h>
#include "Particle.h"
#include "AnglemapGeneration.h"
#include "Physics.h"
#include "RayTracing/RTHelper.h"
#include "MolflowSimFacet.h"

#include <Helper/Chronometer.h>
#include <Helper/MathTools.h>

#include <sstream>
#include <cmath>
#include <set>

using namespace MFSim;

bool Particle::UpdateMCHits(GlobalSimuState &globSimuState, size_t nbMoments, DWORD timeout) {
    int i, j, x, y;

    Chronometer timer(false);
    timer.Start();

    if (!globSimuState.tMutex.try_lock_for(std::chrono::milliseconds(timeout))) {
        return false;
    }

    //SetState(PROCESS_STARTING, "Waiting for 'hits' dataport access...", false, true);

    //SetState(PROCESS_STARTING, "Updating MC hits...", false, true);

    // Global hits and leaks: adding local hits to shared memory
    {
        globSimuState.globalHits.globalHits += tmpState.globalHits.globalHits;
        globSimuState.globalHits.distTraveled_total += tmpState.globalHits.distTraveled_total;
        globSimuState.globalHits.distTraveledTotal_fullHitsOnly += tmpState.globalHits.distTraveledTotal_fullHitsOnly;

        // Update too late
        //totalDesorbed += tmpState.globalHits.globalHits.hit.nbDesorbed;
        totalDesorbed += tmpState.globalHits.globalHits.nbDesorbed;
        //tmpState.globalHits.globalHits.hit.nbDesorbed = 0;

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
            
            if(model->otfParams.raySampling) {
                auto &hitBattery = globSimuState.hitBattery;
                if (hitBattery.initialized && hitBattery.size() > 0) {
                    int hit_n = 0;
                    if(hitBattery.buffer_per_facet && tmpState.hitBattery.size() == hitBattery.size()) {
                        for (auto &bat: hitBattery.rays) {
                            auto &tmp = tmpState.hitBattery.rays[hit_n];
                            if (!tmp.data.empty())
                                bat.Add_Batch(tmp.data);
                            ++hit_n;
                        }
                    }
                    else {
                        auto &tmp = tmpState.hitBattery.grays;
                        if (!tmp.data.empty())
                            hitBattery.grays.Add_Batch(tmp.data);
                    }
                } else {
                    globSimuState.UpdateBatteryFrequencies();
                }
            }
        }

        //Global histograms
        globSimuState.globalHistograms += tmpState.globalHistograms;

        // Facets
        globSimuState.facetStates += tmpState.facetStates;
    }

    globSimuState.stateChanged = true;
    globSimuState.tMutex.unlock();

    //extern char *GetSimuStatus();
    //SetState(PROCESS_STARTING, GetSimuStatus(), false, true);

    //#if defined(DEBUG)
    timer.Stop();

    //printf("Update hits (glob): %lf s [%zu]\n", (t1 - t0) * 1.0, particleId);
    //#endif

    return true;
}

// Compute particle teleport
void Particle::PerformTeleport(SimulationFacet *iFacet) {

    //Search destination
    SimulationFacet *destination;
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
    size_t facId = 0;
    for(auto& fac : model->facets){
        auto& sFac = *fac;
        if (destIndex == static_cast<int>(sFac.globalId)) {
            destination = &(sFac);
            if (destination->sh.superIdx != -1) {
                particle.structure = destination->sh.superIdx; //change current superstructure, unless the target is a universal facet
            }
            teleportedFrom = static_cast<int>(iFacet->globalId); //memorize where the particle came from
            found = true;
            break;
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

    int momentIndex = -1;
    if ((momentIndex = LookupMomentIndex(particle.time, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = momentIndex - 1;
    }
    // Count this hit as a transparent pass
    if (particleId == 0) RecordHit(HIT_TELEPORTSOURCE);
    if (/*iFacet->texture && */iFacet->sh.countTrans)
        RecordHitOnTexture(iFacet, momentIndex, true, 2.0, 2.0);
    if (/*iFacet->direction && */iFacet->sh.countDirection)
        RecordDirectionVector(iFacet, momentIndex);
    ProfileFacet(iFacet, momentIndex, true, 2.0, 2.0);
    LogHit(iFacet);
    if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);

    // Relaunch particle from new facet
    auto[inTheta, inPhi] = CartesianToPolar(particle.direction, iFacet->sh.nU, iFacet->sh.nV,
                                            iFacet->sh.N);
    particle.direction = PolarToCartesian(destination->sh.nU, destination->sh.nV, destination->sh.N, inTheta, inPhi, false);
    // Move particle to teleport destination point
    double u = tmpFacetVars[iFacet->globalId].colU;
    double v = tmpFacetVars[iFacet->globalId].colV;
    particle.origin = destination->sh.O + u * destination->sh.U + v * destination->sh.V;
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
                particle.origin = destination->sh.O + u * destination->sh.U + v * destination->sh.V;
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
            velocity * std::abs(Dot(particle.direction, iFacet->sh.N));
    //We count a teleport as a local hit, but not as a global one since that would affect the MFP calculation
    /*iFacet->sh.tmpCounter.hit.nbMCHit++;
    iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / ortVelocity;
    iFacet->sh.tmpCounter.hit.sum_v_ort += 2.0*(model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/
    IncreaseFacetCounter(iFacet, momentIndex, 1, 0, 0, 2.0 / ortVelocity,
                         2.0 * (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    tmpFacetVars[iFacet->globalId].isHit = true;
    /*destination->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / velocity;
    destination->sh.tmpCounter.hit.sum_v_ort += velocity*abs(DOT3(
    particle.direction.x, particle.direction.y, particle.direction.z,
    destination->sh.N.x, destination->sh.N.y, destination->sh.N.z));*/
}

void DeleteChain (HitChain** head_ref){
    /* deref head_ref to get the real head */
    HitChain* current = *head_ref;
    HitChain* next = nullptr;

    while (current != nullptr)
    {
        next = current->next;
        delete current->hit;
        delete current;
        current = next;
    }

    /* deref head_ref to affect the real head back
        in the caller. */

    *head_ref = nullptr;
}

// Perform nbStep simulation steps (a step is a bounce) or remainingDes desorptions
bool Particle::SimulationMCStep(size_t nbStep, size_t threadNum, size_t remainingDes) {

    // Perform simulation steps
    int returnVal = true;
    //int allQuit = 0;

//#pragma omp parallel num_threads(nbThreads) default(none) firstprivate(nbStep) shared( returnVal, allQuit)
    {
        const int ompIndex = threadNum;//omp_get_thread_num();

        particleId = ompIndex;
        size_t i;

#if defined(USE_OLD_BVH)
        //std::vector<HitLink> hits;
        //hits.reserve(4);
        particle.pay = nullptr;
        particle.tMax = 1.0e99;
        if(lastHitFacet)
            particle.lastIntersected = lastHitFacet->globalId;
        else
            particle.lastIntersected = -1;
        auto copy_rng = randomGenerator;
        particle.rng = &randomGenerator;

        //particle.hits = &hits;
#else
//std::vector<HitLink> hits;
        //hits.reserve(4);
        //particle.pay = nullptr;
        particle.tMax = 1.0e99;
        if(lastHitFacet)
            particle.lastIntersected = lastHitFacet->globalId;
        else
            particle.lastIntersected = -1;

        if(particle.pay) ((RopePayload*)particle.pay)->lastNode = nullptr;
        particle.rng = &randomGenerator;

        //particle.hits = &hits;
#endif
        // start new particle when no previous hit facet was saved
        bool insertNewParticle = !lastHitFacet;
        for (i = 0; i < nbStep && !allQuit; i++) {
            if (insertNewParticle) {
                // quit on desorp error or limit reached
                if((/*model->otfParams.desorptionLimit > 0 && */remainingDes==0) || !StartFromSource(particle)){
                    returnVal = false; // desorp limit reached
                    break;
                }
                if(particle.pay) ((RopePayload*)particle.pay)->lastNode = nullptr;

                //DEBUG
                /*particle.origin = {-6.4573631052006828,0.32848390990035026,17.554289139097389};
                particle.direction = {-0.43593250084036084,-0.60556971638170987,-0.66576885877340675};
                particle.lastIntersected = 860;
                lastHitFacet = model->facets.at(860).get();
                if(!particle.pay)
                    new RopePayload;
                if(dynamic_cast<KdTreeAccel*>(model->accel.at(particle.structure).get())){
                    if(particle.pay)((RopePayload*)particle.pay)->lastNode = &dynamic_cast<KdTreeAccel*>(model->accel.at(particle.structure).get())->nodes[14386];
                }*/
                //ENDDEBUG

                insertNewParticle = false;
                --remainingDes;
            }

            // Todo: Only use one method, ID or Ptr
            if(lastHitFacet)
                particle.lastIntersected = lastHitFacet->globalId;
            else
                particle.lastIntersected = -1;
            //return (lastHitFacet != nullptr);

            //Prepare output values
#if defined(USE_OLD_BVH)
            auto* tmprng = particle.rng;
            particle.rng = &copy_rng;

            auto[oldfound, oldcollidedFacet, oldd] = Intersect(*this, particle.origin,
                                                      particle.direction, model->structures[particle.structure].aabbTree.get());
            //printf("%lf ms time spend in old BVH\n", tmpTime.ElapsedMs());
            //TODO: compare results
            std::set<int> oldtrans;
            if(oldfound){
                for(auto& trans : transparentHitBuffer) {
                    if(trans != nullptr) {
                        oldtrans.emplace(trans->globalId);
                        //RegisterTransparentPass(trans);
                    }
                }
            }

            // reset rng for equal compare
            particle.rng = tmprng;
#endif
            //transparentHitBuffer.clear();
            bool found;
            SimulationFacet* collidedFacet;
            double d;
            {
                //particle = Ray(position, particle.direction, nullptr, 1.0e99, this->particle.time);
                //particle.hits = &hits;
                //particle.rng = &randomGenerator;
                particle.tMax = 1.0e99;
                if(lastHitFacet)
                    particle.lastIntersected = lastHitFacet->globalId;
                else
                    particle.lastIntersected = -1;

#if defined(DEBUG)
                // WIP: Compare
                Ray ray;
                ray.direction = particle.direction;
                ray.origin = particle.origin;
                ray.lastIntersected = particle.lastIntersected;
                ray.structure = particle.structure;
                ray.tMax = particle.tMax;
                ray.rng = new MersenneTwister;
                *ray.rng = *particle.rng;

                RayStat ray2;
                ray2.direction = particle.direction;
                ray2.origin = particle.origin;
                ray2.lastIntersected = particle.lastIntersected;
                ray2.structure = particle.structure;
                ray2.tMax = particle.tMax;
                ray2.rng = new MersenneTwister;
                *ray2.rng = *particle.rng;
                if(model->wp.kd_restart_ropes)
                    ray2.pay = new RopePayload;


                bool testFound = false;
                if(dynamic_cast<KdTreeAccel*>(model->accel.at(particle.structure).get())){
                    testFound = dynamic_cast<KdTreeAccel*>(model->accel.at(particle.structure).get())->IntersectTrav(ray);
                    //assert(found == foundRope);
                }
                delete ray.rng;
                //found = model->accel.at(particle.structure)->Intersect(particle);
                /*if(dynamic_cast<KdTreeAccel*>(model->accel.at(particle.structure).get())){
                    found = dynamic_cast<KdTreeAccel*>(model->accel.at(particle.structure).get())->IntersectRope(particle);
                    //assert(found == foundRope);
                }
                else{*/
                const KdAccelNode* tmp = nullptr;
                if(particle.pay) {
                    if (dynamic_cast<KdTreeAccel *>(model->accel.at(particle.structure).get())) {
                        if (dynamic_cast<KdTreeAccel *>(model->accel.at(particle.structure).get())->restartFromNode)
                            tmp = ((RopePayload *) particle.pay)->lastNode;
                    }
                    ((RopePayload *) ray2.pay)->lastNode = tmp;
                    if (particle.pay)((RopePayload *) ray2.pay)->lastRay = ((RopePayload *) particle.pay)->lastRay;
                }
#endif

#if defined(PROFILING)
                RayStat testParticle(particle);
                found = model->accel.at(testParticle.structure)->IntersectStat(testParticle);
                testParticle.pay = nullptr; // unreference as delete is managed by original particle
#endif
                found = model->accel.at(particle.structure)->Intersect(particle);
                //}

#if defined(DEBUG)
                // particle is selected (e.g rope) algorithm, ray is always regular traversal
                if(found != testFound
                   || particle.hardHit.hitId != ray.hardHit.hitId
                || particle.hardHit.hit.colDistTranspPass != ray.hardHit.hit.colDistTranspPass){
                    std::cerr << "Verification error\n";
                    found = model->accel.at(particle.structure)->IntersectStat(ray2);
                }
                delete ray2.rng;
#endif


                if(found){

#if defined (OLD_HITLINK)

                    // first pass
                    std::set<size_t> alreadyHit; // account for duplicate hits on kdtree

                    for(auto& hit : particle.hits){
                        if(particle.tMax <= hit.hit.colDistTranspPass) {
                            continue;
                        }

                        if(!hit.hit.isHit) { // not hard hit
                            //transparentHitBuffer.push_back(model->facets[hit.hitId].get());

                            // Second pass for transparent hits
                            auto tpFacet = model->facets[hit.hitId].get();
                            tmpFacetVars[hit.hitId] = hit.hit;

                            if(model->wp.accel_type==1) { // account for duplicate hits on kdtree
                                if (alreadyHit.find(tpFacet->globalId) == alreadyHit.end()) {
                                    tmpFacetVars[hit.hitId] = hit.hit;
                                    RegisterTransparentPass(tpFacet);
                                    alreadyHit.insert(tpFacet->globalId);
                                }
                            }
                            else {
                                tmpFacetVars[hit.hitId] = hit.hit;
                                RegisterTransparentPass(tpFacet);
                            }
                        }
                    }
#else
#ifdef DEBUG
                    if(!particle.transparentHits.empty()){
                        for(auto h = 0; h < ray.transparentHits.size(); ++h){
                            auto& hit = ray.transparentHits[h];
                            for(auto hh = h; hh < ray.transparentHits.size(); ++hh){
                                auto& comp_hit = ray.transparentHits[hh];
                                if(hit.hitId == comp_hit.hitId && hit.hit.colU != comp_hit.hit.colU){
                                    std::cout << "Multi hit"<< comp_hit.hit.colU << " vs " << hit.hit.colU << "\n";
                                }
                            }
                        }
                    }
#endif

                    std::set<int> newtrans;
                    std::set<size_t> alreadyHit; // account for duplicate hits on kdtree
                    for(auto& hit : particle.transparentHits){
                        if(particle.tMax <= hit.hit.colDistTranspPass) {
                            continue;
                        }

                        // Second pass for transparent hits
                        auto tpFacet = model->facets[hit.hitId].get();
                        if(model->wp.accel_type==1) { // account for duplicate hits on kdtree
                            if (alreadyHit.find(tpFacet->globalId) == alreadyHit.end()) {
                                tmpFacetVars[hit.hitId] = hit.hit;
                                RegisterTransparentPass(tpFacet);
                                alreadyHit.insert(tpFacet->globalId);
                            }
                            // TODO: Test for ropes
                            else if(hit.hit.colDistTranspPass < tmpFacetVars[hit.hitId].colDistTranspPass) {
                                tmpFacetVars[hit.hitId] = hit.hit;
                            }
                        }
                        else {
                            newtrans.emplace(tpFacet->globalId);
                            assert(hit.hitId == tpFacet->globalId);
#if defined(USE_OLD_BVH)
                            assert(tmpFacetVars[hit.hitId].colU == hit.hit.colU);
                            assert(tmpFacetVars[hit.hitId].colV == hit.hit.colV);
                            //assert(tmpFacetVars[hit.hitId].isHit == hit.hit.isHit);
                            assert(tmpFacetVars[hit.hitId].colDistTranspPass == hit.hit.colDistTranspPass);
#endif
                            tmpFacetVars[hit.hitId] = hit.hit;
                            RegisterTransparentPass(tpFacet);
                        }
                        assert(tmpFacetVars[hit.hitId].colU == hit.hit.colU);
                        assert(tmpFacetVars[hit.hitId].colV == hit.hit.colV);
                        //assert(tmpFacetVars[hit.hitId].isHit == hit.hit.isHit);
                        assert(tmpFacetVars[hit.hitId].colDistTranspPass == hit.hit.colDistTranspPass);
                        tmpFacetVars[hit.hitId] = hit.hit;
                    }

#if defined(USE_OLD_BVH)
                    assert(particle.transparentHits.size() == transparentHitBuffer.size());
                    assert(oldtrans.size() == newtrans.size());
#endif

                    particle.transparentHits.clear();

#endif
                }

                particle.hits.clear();
                transparentHitBuffer.clear();
            }

            // hard hit
            if(found){
                auto& hit = particle.hardHit;
                collidedFacet = model->facets[hit.hitId].get();
                tmpFacetVars[hit.hitId] = hit.hit;
                d = hit.hit.colDistTranspPass;
            }

#if defined(USE_OLD_BVH)
            assert(found == oldfound);
            if(found) {
                assert(collidedFacet->globalId == oldcollidedFacet->globalId);
                assert(d == oldd);
            }
#endif
            //found = oldfound;collidedFacet = oldcollidedFacet;d = oldd;
            /*{
                Ray tmpRay(position, particle.direction, nullptr, 1.0e99, this->particle.time);
                if(lastHitFacet)
                    tmpRay.lastIntersected = lastHitFacet->globalId;
                else
                    tmpRay.lastIntersected = -1;
                tmpRay.rng = &randomGenerator;

                HitChain* hitChain = new HitChain();
                tmpRay.hitChain = hitChain;
#if defined(USE_KDTREE)
                found = model->kdtree[particle.structure].Intersect(tmpRay);
#else
                found = model->bvhs[particle.structure].Intersect(tmpRay);
#endif
                if(found){
                    HitChain* currHit = hitChain;
                    while(currHit){
                        if(!currHit->hit->isHit) {
                            transparentHitBuffer.push_back(model->facets[currHit->hitId].get());
                        }
                        else {
                            collidedFacet = model->facets[currHit->hitId].get();
                            d = currHit->hit->colDistTranspPass;
                        }
                        tmpFacetVars[currHit->hitId] = *currHit->hit;
                        currHit = currHit->next;
                    }
                }

                DeleteChain(&hitChain);
            }*/
//#endif //use old bvh

            if (found) {

                // Move particle to intersection point
                particle.origin =
                        particle.origin + d * particle.direction;
                //distanceTraveled += d;

                const double lastParticleTime = particle.time; //memorize for partial hits
                particle.time +=
                        d / 100.0 / velocity; //conversion from cm to m

                if ((!model->wp.calcConstantFlow && (particle.time > model->wp.latestMoment))
                    || (model->wp.enableDecay &&
                        (expectedDecayMoment < particle.time))) {
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
                    particle.lastIntersected = -1;
                } else { //hit within measured time, particle still alive
                    if (collidedFacet->sh.teleportDest != 0) { //Teleport
                        IncreaseDistanceCounters(d * oriRatio);
                        PerformTeleport(collidedFacet);
                    }
                        /*else if ((GetOpacityAt(collidedFacet, particle.time) < 1.0) && (randomGenerator.rnd() > GetOpacityAt(collidedFacet, particle.time))) {
                            //Transparent pass
                            tmpState.globalHits.distTraveled_total += d;
                            PerformTransparentPass(collidedFacet);
                        }*/
                    else { //Not teleport
                        IncreaseDistanceCounters(d * oriRatio);
                        const double stickingProbability = model->GetStickingAt(collidedFacet, particle.time);
                        if (!model->otfParams.lowFluxMode) { //Regular stick or bounce
                            double stickingRand = randomGenerator.rnd();
                            if (stickingProbability == 1.0 ||
                                ((stickingProbability > 0.0) && (stickingRand < (stickingProbability)))) {
                                //Absorbed
                                RecordAbsorb(collidedFacet);
                                //currentParticle.lastHitFacet = nullptr; // null facet in case we reached des limit and want to go on, prevents leak
                                //distTraveledSinceUpdate += distanceTraveled;
                                insertNewParticle = true;
                                lastHitFacet=nullptr;
                                particle.lastIntersected = -1;
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
                                particle.lastIntersected = -1;
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
                particle.lastIntersected = -1;
                lastHitFacet=nullptr;
                particle.lastIntersected = -1;
            }
        }

/*#pragma omp critical
            ++allQuit;*/
    } // omp parallel

    return returnVal;
}

void Particle::IncreaseDistanceCounters(double distanceIncrement) {
    tmpState.globalHits.distTraveled_total += distanceIncrement;
    tmpState.globalHits.distTraveledTotal_fullHitsOnly += distanceIncrement;
    distanceTraveled += distanceIncrement;
}

// Launch a ray from a source facet. The ray
// particle.direction is chosen according to the desorption type.
bool Particle::StartFromSource(Ray& ray) {
    bool found = false;
    bool foundInMap = false;
    bool reverse;
    size_t mapPositionW, mapPositionH;
    double srcRnd;
    double sumA = 0.0;
    size_t i = 0, j = 0;
    int nbTry = 0;
    if(particle.pay) ((RopePayload*)particle.pay)->lastNode = nullptr;

    // Check end of simulation
    /*if (model->otfParams.desorptionLimit > 0) {
        if (tmpState.globalHits.globalHits.hit.nbDesorbed >=
            model->otfParams.desorptionLimit / model->otfParams.nbProcess) {
            //lastHitFacet = nullptr; // reset full particle status or go on from where we left
            return false;
        }
    }*/

    // Select source
    srcRnd = ray.rng->rnd() * model->wp.totalDesorbedMolecules;

    i = 0;
    for(auto& fac : model->facets) { //Go through facets in a structure
        auto f = std::dynamic_pointer_cast<MolflowSimFacet>(fac);
        if (f->sh.desorbType != DES_NONE) { //there is some kind of outgassing
            if (f->sh.useOutgassingFile) {
                //Using SynRad-generated outgassing map
                if (f->sh.totalOutgassing > 0.0) {
                    found = (srcRnd >= sumA) && (srcRnd < (sumA + model->wp.latestMoment * f->sh.totalOutgassing /
                                                                  (1.38E-23 * f->sh.temperature)));
                    if (found) {
                        //look for exact position in map
                        double rndRemainder = (srcRnd - sumA) / model->wp.latestMoment * (1.38E-23 *
                                                                                          f->sh.temperature); //remainder, should be less than f->sh.totalOutgassing
                        /*double sumB = 0.0;
                        for (w = 0; w < f->sh.outgassingMapWidth && !foundInMap; w++) {
                            for (h = 0; h < f->sh.outgassingMapHeight && !foundInMap; h++) {
                                double cellOutgassing = f->outgassingMap[h*f->sh.outgassingMapWidth + w];
                                if (cellOutgassing > 0.0) {
                                    foundInMap = (rndRemainder >= sumB) && (rndRemainder < (sumB + cellOutgassing));
                                    if (foundInMap) mapPositionW = w; mapPositionH = h;
                                    sumB += cellOutgassing;
                                }
                            }
                        }*/
                        double lookupValue = rndRemainder;
                        int outgLowerIndex = my_lower_bound(lookupValue,
                                                            f->ogMap.outgassingMap_cdf); //returns line number AFTER WHICH LINE lookup value resides in ( -1 .. size-2 )
                        outgLowerIndex++;
                        mapPositionH = (size_t) ((double) outgLowerIndex / (double) f->ogMap.outgassingMapWidth);
                        mapPositionW = (size_t) outgLowerIndex - mapPositionH * f->ogMap.outgassingMapWidth;
                        foundInMap = true;
                        /*if (!foundInMap) {
                            SetErrorSub("Starting point not found in imported desorption map");
                            return false;
                        }*/
                    }
                    sumA += model->wp.latestMoment * f->sh.totalOutgassing / (1.38E-23 * f->sh.temperature);
                }
            } //end outgassing file block
            else { //constant or time-dependent outgassing
                double facetOutgassing =
                        ((f->sh.outgassing_paramId >= 0)
                         ? model->tdParams.IDs[f->sh.IDid].back().second
                         : model->wp.latestMoment * f->sh.outgassing) / (1.38E-23 * f->sh.temperature);
                found = (srcRnd >= sumA) && (srcRnd < (sumA + facetOutgassing));
                sumA += facetOutgassing;
            } //end constant or time-dependent outgassing block
        } //end 'there is some kind of outgassing'
        if (!found) i++;
        if (f->sh.is2sided) reverse = ray.rng->rnd() > 0.5;
        else reverse = false;

        if(found) break;
    } // facet loop

    if (!found) {
        fmt::print(stderr,  "No starting point, aborting\n");
        return false;
    }

    auto src = model->facets[i].get();
    lastHitFacet = src;
    ray.lastIntersected = lastHitFacet->globalId;
    //distanceTraveled = 0.0;  //for mean free path calculations
    //particle.time = desorptionStartTime + (desorptionStopTime - desorptionStartTime)*randomGenerator.rnd();
    ray.time = generationTime = Physics::GenerateDesorptionTime(model->tdParams.IDs, src, randomGenerator.rnd(), model->wp.latestMoment);
    lastMomentIndex = 0;
    if (model->wp.useMaxwellDistribution) velocity = Physics::GenerateRandomVelocity(model->tdParams.CDFs, src->sh.CDFid, randomGenerator.rnd());
    else
        velocity =
                145.469 * std::sqrt(src->sh.temperature / model->wp.gasMass);  //sqrt(8*R/PI/1000)=145.47

    oriRatio = 1.0;
    if (model->wp.enableDecay) { //decaying gas
        expectedDecayMoment =
                ray.time + model->wp.halfLife * 1.44269 * -log(randomGenerator.rnd()); //1.44269=1/ln2
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
            auto& outgMap = ((MolflowSimFacet*)(src))->ogMap;
            if (mapPositionW < (outgMap.outgassingMapWidth - 1)) {
                //Somewhere in the middle of the facet
                u = ((double) mapPositionW + randomGenerator.rnd()) / outgMap.outgassingMapWidth_precise;
            } else {
                //Last element, prevent from going out of facet
                u = ((double) mapPositionW +
                     randomGenerator.rnd() * (outgMap.outgassingMapWidth_precise - (outgMap.outgassingMapWidth - 1.0))) /
                    outgMap.outgassingMapWidth_precise;
            }
            if (mapPositionH < (outgMap.outgassingMapHeight - 1)) {
                //Somewhere in the middle of the facet
                v = ((double) mapPositionH + randomGenerator.rnd()) / outgMap.outgassingMapHeight_precise;
            } else {
                //Last element, prevent from going out of facet
                v = ((double) mapPositionH +
                     randomGenerator.rnd() * (outgMap.outgassingMapHeight_precise - (outgMap.outgassingMapHeight - 1.0))) /
                    outgMap.outgassingMapHeight_precise;
            }
        } else {
            u = randomGenerator.rnd();
            v = randomGenerator.rnd();
        }
        if (IsInFacet(*src, u, v)) {

            // (U,V) -> (x,y,z)
            ray.origin = src->sh.O + u * src->sh.U + v * src->sh.V;
            tmpFacetVars[src->globalId].colU = u;
            tmpFacetVars[src->globalId].colV = v;
            found = true;

        }
        nbTry++;
    }

    if (!found) {
        // Get the center, if the center is not included in the facet, a leak is generated.
        if (foundInMap) {
            auto& outgMap = ((MolflowSimFacet*)(src))->ogMap;
            //double uLength = sqrt(pow(src->sh.U.x, 2) + pow(src->sh.U.y, 2) + pow(src->sh.U.z, 2));
            //double vLength = sqrt(pow(src->sh.V.x, 2) + pow(src->sh.V.y, 2) + pow(src->sh.V.z, 2));
            double u = ((double) mapPositionW + 0.5) / outgMap.outgassingMapWidth_precise;
            double v = ((double) mapPositionH + 0.5) / outgMap.outgassingMapHeight_precise;
            ray.origin = src->sh.O + u * src->sh.U + v * src->sh.V;
            tmpFacetVars[src->globalId].colU = u;
            tmpFacetVars[src->globalId].colV = v;
        } else {
            tmpFacetVars[src->globalId].colU = 0.5;
            tmpFacetVars[src->globalId].colV = 0.5;
            ray.origin = src->sh.center;
        }

    }



    if (particleId == 0) {
        if (src->sh.isMoving && model->wp.motionType)
            RecordHit(HIT_MOVING);
        else
            RecordHit(HIT_DES); //create blue hit point for created particle
    }

    //See docs/theta_gen.png for further details on angular distribution generation
    switch (src->sh.desorbType) {
        case DES_UNIFORM:
            ray.direction = PolarToCartesian(src->sh.nU, src->sh.nV, src->sh.N, std::acos(randomGenerator.rnd()),
                                         randomGenerator.rnd() * 2.0 * PI,
                                         reverse);
            break;
        case DES_NONE: //for file-based
        case DES_COSINE:
            ray.direction = PolarToCartesian(src->sh.nU, src->sh.nV, src->sh.N, std::acos(std::sqrt(randomGenerator.rnd())),
                                         randomGenerator.rnd() * 2.0 * PI,
                                         reverse);
            break;
        case DES_COSINE_N:
            ray.direction = PolarToCartesian(src->sh.nU, src->sh.nV, src->sh.N, std::acos(
                    std::pow(randomGenerator.rnd(), 1.0 / (src->sh.desorbTypeN + 1.0))),
                                         randomGenerator.rnd() * 2.0 * PI, reverse);
            break;
        case DES_ANGLEMAP: {
            auto[theta, thetaLowerIndex, thetaOvershoot] = AnglemapGeneration::GenerateThetaFromAngleMap(
                    src->sh.anglemapParams, ((MolflowSimFacet*)(src))->angleMap, randomGenerator.rnd());

            auto phi = AnglemapGeneration::GeneratePhiFromAngleMap(thetaLowerIndex, thetaOvershoot,
                                                                   src->sh.anglemapParams, ((MolflowSimFacet*)(src))->angleMap, randomGenerator.rnd());
                            
            /*                                                      
            //Debug
            double phi;
            thetaLowerIndex = 0;
            thetaOvershoot = 0;
            std::vector<double> phis;
            for (double r = 0.0; r < 1.0; r += 0.001) {
                 phi = AnglemapGeneration::GeneratePhiFromAngleMap(thetaLowerIndex, thetaOvershoot,
                    src->sh.anglemapParams, src->angleMap,
                    r);
                phis.push_back(phi);
            }
            */

            ray.direction = PolarToCartesian(src->sh.nU, src->sh.nV, src->sh.N, PI - theta, phi,
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
    ray.structure = src->sh.superIdx;

    teleportedFrom = -1;

    // Count

    tmpFacetVars[src->globalId].isHit = true;
/*#pragma omp critical
    {
        totalDesorbed++;
    }*/
    tmpState.globalHits.globalHits.nbDesorbed++;
    //nbPHit = 0;

    if (src->sh.isMoving) {
        Physics::TreatMovingFacet(model, ray.origin, ray.direction, velocity);
    }

    double ortVelocity =
            velocity * std::abs(Dot(ray.direction, src->sh.N));
    /*src->sh.tmpCounter.hit.nbDesorbed++;
    src->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / ortVelocity; //was 2.0 / ortV
    src->sh.tmpCounter.hit.sum_v_ort += (model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/
    int momentIndex = -1;
    if ((momentIndex = LookupMomentIndex(ray.time, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = momentIndex - 1;
    }

    IncreaseFacetCounter(src, momentIndex, 0, 1, 0, 2.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    //Desorption doesn't contribute to angular profiles, nor to angle maps
    ProfileFacet(src, momentIndex, false, 2.0, 1.0); //was 2.0, 1.0
    LogHit(src);
    if (/*src->texture && */src->sh.countDes)
        RecordHitOnTexture(src, momentIndex, true, 2.0, 1.0); //was 2.0, 1.0
    //if (src->direction && src->sh.countDirection) RecordDirectionVector(src, particle.time);

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

/**
* \brief Perform a bounce from a facet by logging the hit and sometimes relaunching it
* \param iFacet facet corresponding to the bounce event
*/
void Particle::PerformBounce(SimulationFacet *iFacet) {

    bool revert = false;
    tmpState.globalHits.globalHits.nbMCHit++; //global
    tmpState.globalHits.globalHits.nbHitEquiv += oriRatio;

    // Handle super structure link facet. Can be
    if (iFacet->sh.superDest) {
        int momentIndex = -1;
        if ((momentIndex = LookupMomentIndex(particle.time, model->tdParams.moments, lastMomentIndex)) > 0) {
            lastMomentIndex = momentIndex - 1;
        }

        IncreaseFacetCounter(iFacet, momentIndex, 1, 0, 0, 0, 0);
        particle.structure = iFacet->sh.superDest - 1;
        if (iFacet->sh.isMoving) { //A very special case where link facets can be used as transparent but moving facets
            if (particleId == 0)RecordHit(HIT_MOVING);
            Physics::TreatMovingFacet(model, particle.origin, particle.direction, velocity);
        } else {
            // Count this hit as a transparent pass
            if (particleId == 0)RecordHit(HIT_TRANS);
        }
        LogHit(iFacet);

        ProfileFacet(iFacet, momentIndex, true, 2.0, 2.0);
        if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);
        if (/*iFacet->texture &&*/ iFacet->sh.countTrans)
            RecordHitOnTexture(iFacet, momentIndex, true, 2.0, 2.0);
        if (/*iFacet->direction &&*/ iFacet->sh.countDirection)
            RecordDirectionVector(iFacet, momentIndex);

        return;

    }

    // Handle volatile facet
    if (iFacet->sh.isVolatile) {
        if (iFacet->isReady) {
            int momentIndex = -1;
            if ((momentIndex = LookupMomentIndex(particle.time, model->tdParams.moments, lastMomentIndex)) > 0) {
                lastMomentIndex = momentIndex - 1;
            }

            IncreaseFacetCounter(iFacet, momentIndex, 0, 0, 1, 0, 0);
            iFacet->isReady = false;
            LogHit(iFacet);
            ProfileFacet(iFacet, momentIndex, true, 2.0, 1.0);
            if (/*iFacet->texture && */iFacet->sh.countAbs)
                RecordHitOnTexture(iFacet, momentIndex, true, 2.0, 1.0);
            if (/*iFacet->direction && */iFacet->sh.countDirection)
                RecordDirectionVector(iFacet, momentIndex);
        }
        return;

    }

    if (iFacet->sh.is2sided) {
        // We may need to revert normal in case of 2 sided hit
        revert = Dot(particle.direction, iFacet->sh.N) > 0.0;
    }

    //Texture/Profile incoming hit


    //Register (orthogonal) velocity
    double ortVelocity =
            velocity * std::abs(Dot(particle.direction, iFacet->sh.N));

    /*iFacet->sh.tmpCounter.hit.nbMCHit++; //hit facet
    iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 1.0 / ortVelocity;
    iFacet->sh.tmpCounter.hit.sum_v_ort += (model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/

    int momentIndex = -1;
    if ((momentIndex = LookupMomentIndex(particle.time, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = momentIndex - 1;
    }

    IncreaseFacetCounter(iFacet, momentIndex, 1, 0, 0, 1.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    nbBounces++;
    if (/*iFacet->texture &&*/ iFacet->sh.countRefl)
        RecordHitOnTexture(iFacet, momentIndex, true, 1.0, 1.0);
    if (/*iFacet->direction &&*/ iFacet->sh.countDirection)
        RecordDirectionVector(iFacet, momentIndex);
    LogHit(iFacet);
    ProfileFacet(iFacet, momentIndex, true, 1.0, 1.0);
    if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);

    // Relaunch particle
    UpdateVelocity(iFacet);
    //Sojourn time
    if (iFacet->sh.enableSojournTime) {
        double A = exp(-iFacet->sh.sojournE / (8.31 * iFacet->sh.temperature));
        particle.time += -log(randomGenerator.rnd()) / (A * iFacet->sh.sojournFreq);
        momentIndex = LookupMomentIndex(particle.time, model->tdParams.moments, lastMomentIndex); //reflection might happen in another moment
    }

    if (iFacet->sh.reflection.diffusePart > 0.999999) { //Speedup branch for most common, diffuse case
        particle.direction = PolarToCartesian(iFacet->sh.nU, iFacet->sh.nV, iFacet->sh.N, std::acos(std::sqrt(randomGenerator.rnd())),
                                     randomGenerator.rnd() * 2.0 * PI,
                                     revert);
    } else {
        double reflTypeRnd = randomGenerator.rnd();
        if (reflTypeRnd < iFacet->sh.reflection.diffusePart) {
            //diffuse reflection
            //See docs/theta_gen.png for further details on angular distribution generation
            particle.direction = PolarToCartesian(iFacet->sh.nU, iFacet->sh.nV, iFacet->sh.N, std::acos(std::sqrt(randomGenerator.rnd())),
                                         randomGenerator.rnd() * 2.0 * PI,
                                         revert);
        } else if (reflTypeRnd < (iFacet->sh.reflection.diffusePart + iFacet->sh.reflection.specularPart)) {
            //specular reflection
            auto[inTheta, inPhi] = CartesianToPolar(particle.direction, iFacet->sh.nU, iFacet->sh.nV,
                                                    iFacet->sh.N);
            particle.direction = PolarToCartesian(iFacet->sh.nU, iFacet->sh.nV, iFacet->sh.N, PI - inTheta, inPhi, false);

        } else {
            //Cos^N reflection
            particle.direction = PolarToCartesian(iFacet->sh.nU, iFacet->sh.nV, iFacet->sh.N, std::acos(
                            std::pow(randomGenerator.rnd(), 1.0 / (iFacet->sh.reflection.cosineExponent + 1.0))),
                                         randomGenerator.rnd() * 2.0 * PI, revert);
        }
    }

    if (iFacet->sh.isMoving) {
        Physics::TreatMovingFacet(model, particle.origin, particle.direction, velocity);
    }

    //Texture/Profile outgoing particle
    //Register outgoing velocity
    ortVelocity = velocity * std::abs(Dot(particle.direction, iFacet->sh.N));

    /*iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 1.0 / ortVelocity;
    iFacet->sh.tmpCounter.hit.sum_v_ort += (model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*ortVelocity;*/
    IncreaseFacetCounter(iFacet, momentIndex, 0, 0, 0, 1.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    if (/*iFacet->texture &&*/ iFacet->sh.countRefl)
        RecordHitOnTexture(iFacet, momentIndex, false, 1.0,
                           1.0); //count again for outward velocity
    ProfileFacet(iFacet, momentIndex, false, 1.0, 1.0);
    //no particle.direction count on outgoing, neither angle map

    if (iFacet->sh.isMoving && model->wp.motionType) {
        if (particleId == 0)
            RecordHit(HIT_MOVING);
    } else if (particleId == 0)RecordHit(HIT_REF);
    lastHitFacet = iFacet;
    //nbPHit++;
}

/*void Simulation::PerformTransparentPass(SimulationFacet *iFacet) { //disabled, caused finding hits with the same facet
    *//*double particle.directionFactor = abs(DOT3(
        particle.direction.x, particle.direction.y, particle.direction.z,
        iFacet->sh.N.x, iFacet->sh.N.y, iFacet->sh.N.z));
    iFacet->sh.tmpCounter.hit.nbMCHit++;
    iFacet->sh.tmpCounter.hit.sum_1_per_ort_velocity += 2.0 / (velocity*directionFactor);
    iFacet->sh.tmpCounter.hit.sum_v_ort += 2.0*(model->wp.useMaxwellDistribution ? 1.0 : 1.1781)*velocity*directionFactor;
    iFacet->isHit = true;
    if (iFacet->texture && iFacet->sh.countTrans) RecordHitOnTexture(iFacet, particle.time + iFacet->colDist / 100.0 / velocity,
        true, 2.0, 2.0);
    if (iFacet->direction && iFacet->sh.countDirection) RecordDirectionVector(iFacet, particle.time + iFacet->colDist / 100.0 / velocity);
    ProfileFacet(iFacet, particle.time + iFacet->colDist / 100.0 / velocity,
        true, 2.0, 2.0);
    RecordHit(HIT_TRANS);
    lastHit = iFacet;*//*
}*/

void Particle::RecordAbsorb(SimulationFacet *iFacet) {
    tmpState.globalHits.globalHits.nbMCHit++; //global
    tmpState.globalHits.globalHits.nbHitEquiv += oriRatio;
    tmpState.globalHits.globalHits.nbAbsEquiv += oriRatio;

    int momentIndex = -1;
    if ((momentIndex = LookupMomentIndex(particle.time, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = momentIndex - 1;
    }

    RecordHistograms(iFacet, momentIndex);

    if (particleId == 0) RecordHit(HIT_ABS);
    double ortVelocity =
            velocity * std::abs(Dot(particle.direction, iFacet->sh.N));
    IncreaseFacetCounter(iFacet, momentIndex, 1, 0, 1, 2.0 / ortVelocity,
                         (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity);
    LogHit(iFacet);
    ProfileFacet(iFacet, momentIndex, true, 2.0, 1.0); //was 2.0, 1.0
    if (iFacet->sh.anglemapParams.record) RecordAngleMap(iFacet);
    if (/*iFacet->texture &&*/ iFacet->sh.countAbs)
        RecordHitOnTexture(iFacet, momentIndex, true, 2.0, 1.0); //was 2.0, 1.0
    if (/*iFacet->direction &&*/ iFacet->sh.countDirection)
        RecordDirectionVector(iFacet, momentIndex);
}

void Particle::RecordHistograms(SimulationFacet *iFacet, int m) {
    //Record in global and facet histograms
    size_t binIndex;

    auto &tmpGlobalHistograms = tmpState.globalHistograms;
    auto &facetHistogram = tmpState.facetStates[iFacet->globalId].momentResults;
    auto &globHistParams = model->wp.globalHistogramParams;
    auto &facHistParams = iFacet->sh.facetHistogramParams;

    for (const int moment : {0, m}) {
        if (moment < 0) return;

        if (globHistParams.recordBounce) {
            binIndex = Min(nbBounces / globHistParams.nbBounceBinsize,
                           globHistParams.GetBounceHistogramSize() - 1);
            tmpGlobalHistograms[moment].nbHitsHistogram[binIndex] += oriRatio;
        }
        if (globHistParams.recordDistance) {
            binIndex = Min(static_cast<size_t>(distanceTraveled /
                                               globHistParams.distanceBinsize),
                           globHistParams.GetDistanceHistogramSize() - 1);
            tmpGlobalHistograms[moment].distanceHistogram[binIndex] += oriRatio;
        }
        if (globHistParams.recordTime) {
            binIndex = Min(static_cast<size_t>((particle.time - generationTime) /
                                               globHistParams.timeBinsize),
                           globHistParams.GetTimeHistogramSize() - 1);
            tmpGlobalHistograms[moment].timeHistogram[binIndex] += oriRatio;
        }
        if (facHistParams.recordBounce) {
            binIndex = Min(nbBounces / facHistParams.nbBounceBinsize,
                           facHistParams.GetBounceHistogramSize() - 1);
            facetHistogram[moment].histogram.nbHitsHistogram[binIndex] += oriRatio;
        }
        if (facHistParams.recordDistance) {
            binIndex = Min(static_cast<size_t>(distanceTraveled /
                                               facHistParams.distanceBinsize),
                           facHistParams.GetDistanceHistogramSize() - 1);
            facetHistogram[moment].histogram.distanceHistogram[binIndex] += oriRatio;
        }
        if (facHistParams.recordTime) {
            binIndex = Min(static_cast<size_t>((particle.time - generationTime) /
                                               facHistParams.timeBinsize),
                           facHistParams.GetTimeHistogramSize() - 1);
            facetHistogram[moment].histogram.timeHistogram[binIndex] += oriRatio;
        }
    }
}

void
Particle::RecordHitOnTexture(const SimulationFacet *f, int m, bool countHit, double velocity_factor,
                             double ortSpeedFactor) {

    size_t tu = (size_t) (tmpFacetVars[f->globalId].colU * f->sh.texWidth_precise);
    size_t tv = (size_t) (tmpFacetVars[f->globalId].colV * f->sh.texHeight_precise);
    size_t add = tu + tv * (f->sh.texWidth);
    double ortVelocity = (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * velocity *
                         std::abs(Dot(particle.direction,
                                      f->sh.N)); //surface-orthogonal velocity component

    {
        TextureCell &texture = tmpState.facetStates[f->globalId].momentResults[0].texture[add];
        if (countHit) texture.countEquiv += oriRatio;
        texture.sum_1_per_ort_velocity +=
                oriRatio * velocity_factor / ortVelocity;
        texture.sum_v_ort_per_area += oriRatio * ortSpeedFactor * ortVelocity *
                                      f->textureCellIncrements[add]; // sum ortho_velocity[m/s] / cell_area[cm2]
    }
    if (m > 0) {
        TextureCell &texture = tmpState.facetStates[f->globalId].momentResults[m].texture[add];
        if (countHit) texture.countEquiv += oriRatio;
        texture.sum_1_per_ort_velocity +=
                oriRatio * velocity_factor / ortVelocity;
        texture.sum_v_ort_per_area += oriRatio * ortSpeedFactor * ortVelocity *
                                      f->textureCellIncrements[add]; // sum ortho_velocity[m/s] / cell_area[cm2]
    }
}

void Particle::RecordDirectionVector(const SimulationFacet *f, int m) {
    size_t tu = (size_t) (tmpFacetVars[f->globalId].colU * f->sh.texWidth_precise);
    size_t tv = (size_t) (tmpFacetVars[f->globalId].colV * f->sh.texHeight_precise);
    size_t add = tu + tv * (f->sh.texWidth);

    {
        DirectionCell &dirCell = tmpState.facetStates[f->globalId].momentResults[0].direction[add];
        dirCell.dir += oriRatio * particle.direction * velocity;
        dirCell.count++;
    }
    if (m > 0) {
        lastMomentIndex = m - 1;
        DirectionCell &dirCell = tmpState.facetStates[f->globalId].momentResults[m].direction[add];
        dirCell.dir += oriRatio * particle.direction * velocity;
        dirCell.count++;
    }

}

void
Particle::ProfileFacet(const SimulationFacet *f, int m, bool countHit, double velocity_factor,
                       double ortSpeedFactor) {

    size_t nbMoments = model->tdParams.moments.size();
    if (countHit && f->sh.profileType == PROFILE_ANGULAR) {
        double dot = Dot(f->sh.N, particle.direction);
        double theta = std::acos(std::abs(dot));     // Angle to normal (PI/2 => PI)
        size_t pos = (size_t) (theta / (PI / 2) * ((double) PROFILE_SIZE)); // To Grad
        Saturate(pos, 0, PROFILE_SIZE - 1);

        tmpState.facetStates[f->globalId].momentResults[0].profile[pos].countEquiv += oriRatio;
        if (m > 0) {
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
                                     std::abs(Dot(f->sh.N, particle.direction));
                profile.sum_1_per_ort_velocity +=
                        oriRatio * velocity_factor / ortVelocity;
                profile.sum_v_ort += oriRatio * ortSpeedFactor *
                                     (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * ortVelocity;
            }
            if (m > 0) {
                ProfileSlice &profile = tmpState.facetStates[f->globalId].momentResults[m].profile[pos];
                if (countHit) profile.countEquiv += oriRatio;
                double ortVelocity = velocity *
                                     std::abs(Dot(f->sh.N, particle.direction));
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
            dot = std::abs(Dot(f->sh.N, particle.direction));  //cos(theta) as "dot" value
        } else { //Tangential
            dot = std::sqrt(1 - Sqr(std::abs(Dot(f->sh.N, particle.direction))));  //tangential
        }
        size_t pos = (size_t) (dot * velocity / f->sh.maxSpeed *
                               (double) PROFILE_SIZE); //"dot" default value is 1.0
        if (pos >= 0 && pos < PROFILE_SIZE) {
            tmpState.facetStates[f->globalId].momentResults[0].profile[pos].countEquiv += oriRatio;
            if (m > 0) {
                tmpState.facetStates[f->globalId].momentResults[m].profile[pos].countEquiv += oriRatio;
            }
        }
    }
}

void
Particle::LogHit(SimulationFacet *f) {
    //if(omp_get_thread_num() != 0) return; // only let 1 thread update
    if (model->otfParams.enableLogging &&
        model->otfParams.logFacetId == f->globalId &&
        tmpParticleLog.pLog.size() < tmpParticleLog.pLog.capacity()) {
        ParticleLoggerItem log;
        log.facetHitPosition = Vector2d(tmpFacetVars[f->globalId].colU, tmpFacetVars[f->globalId].colV);
        std::tie(log.hitTheta, log.hitPhi) = CartesianToPolar(particle.direction, f->sh.nU, f->sh.nV, f->sh.N);
        log.oriRatio = oriRatio;
        log.particleDecayMoment = expectedDecayMoment;
        log.time = particle.time;
        log.velocity = velocity;
        tmpParticleLog.pLog.push_back(log);
    }
}

void Particle::RecordAngleMap(const SimulationFacet *collidedFacet) {
    auto[inTheta, inPhi] = CartesianToPolar(particle.direction, collidedFacet->sh.nU,
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

void Particle::UpdateVelocity(const SimulationFacet *collidedFacet) {
    if (collidedFacet->sh.accomodationFactor > 0.9999) { //speedup for the most common case: perfect thermalization
        if (model->wp.useMaxwellDistribution)
            velocity = Physics::GenerateRandomVelocity(model->tdParams.CDFs, collidedFacet->sh.CDFid, randomGenerator.rnd());
        else
            velocity =
                    145.469 * std::sqrt(collidedFacet->sh.temperature / model->wp.gasMass);
    } else {
        double oldSpeed2 = pow(velocity, 2);
        double newSpeed2;
        if (model->wp.useMaxwellDistribution)
            newSpeed2 = pow(Physics::GenerateRandomVelocity(model->tdParams.CDFs,collidedFacet->sh.CDFid,
                                                   randomGenerator.rnd()), 2);
        else newSpeed2 = /*145.469*/ 29369.939 * (collidedFacet->sh.temperature / model->wp.gasMass);
        //sqrt(29369)=171.3766= sqrt(8*R*1000/PI)*3PI/8, that is, the constant part of the v_avg=sqrt(8RT/PI/m/0.001)) found in literature, multiplied by
        //the corrective factor of 3PI/8 that accounts for moving from volumetric speed distribution to wall collision speed distribution
        velocity = std::sqrt(
                oldSpeed2 + (newSpeed2 - oldSpeed2) * collidedFacet->sh.accomodationFactor);
    }
}

/*double Particle::GenerateRandomVelocity(int CDFId, const double rndVal) {
    //return FastLookupY(randomGenerator.rnd(),CDFs[CDFId],false);
    //double r = randomGenerator.rnd();
    double v = InterpolateX(rndVal, model->tdParams.CDFs[CDFId], false, false, true); //Allow extrapolate
    return v;
}

double Particle::GenerateDesorptionTime(const SimulationFacet *src, const double rndVal) {
    if (src->sh.outgassing_paramId >= 0) { //time-dependent desorption
        return InterpolateX(rndVal * model->tdParams.IDs[src->sh.IDid].back().second, model->tdParams.IDs[src->sh.IDid],
                            false, false, true); //allow extrapolate
    } else {
        return rndVal * model->wp.latestMoment; //continous desorption between 0 and latestMoment
    }
}*/



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
Particle::IncreaseFacetCounter(const SimulationFacet *f, int m, size_t hit, size_t desorb,
                               size_t absorb,
                               double sum_1_per_v, double sum_v_ort) {
    const double hitEquiv = static_cast<double>(hit) * oriRatio;
    {
        FacetHitBuffer &hits = tmpState.facetStates[f->globalId].momentResults[0].hits;
        hits.nbMCHit += hit;
        hits.nbHitEquiv += hitEquiv;
        hits.nbDesorbed += desorb;
        hits.nbAbsEquiv += static_cast<double>(absorb) * oriRatio;
        hits.sum_1_per_ort_velocity += oriRatio * sum_1_per_v;
        hits.sum_v_ort += oriRatio * sum_v_ort;
        hits.sum_1_per_velocity += (hitEquiv + static_cast<double>(desorb)) / velocity;
    }
    if (m > 0) {
        FacetHitBuffer &hits = tmpState.facetStates[f->globalId].momentResults[m].hits;
        hits.nbMCHit += hit;
        hits.nbHitEquiv += hitEquiv;
        hits.nbDesorbed += desorb;
        hits.nbAbsEquiv += static_cast<double>(absorb) * oriRatio;
        hits.sum_1_per_ort_velocity += oriRatio * sum_1_per_v;
        hits.sum_v_ort += oriRatio * sum_v_ort;
        hits.sum_1_per_velocity += (hitEquiv + static_cast<double>(desorb)) / velocity;
    }
}

void Particle::RegisterTransparentPass(SimulationFacet *facet) {
    double directionFactor = std::abs(Dot(particle.direction, facet->sh.N));

    int momentIndex = -1;
    if ((momentIndex = LookupMomentIndex(particle.time +
                                         tmpFacetVars[facet->globalId].colDistTranspPass / 100.0 / velocity, model->tdParams.moments, lastMomentIndex)) > 0) {
        lastMomentIndex = momentIndex - 1;
    }

    IncreaseFacetCounter(facet, momentIndex, 1, 0, 0,
                         2.0 / (velocity * directionFactor),
                         2.0 * (model->wp.useMaxwellDistribution ? 1.0 : 1.1781) * velocity *
                         directionFactor);

    tmpFacetVars[facet->globalId].isHit = true;
    if (/*facet->texture &&*/ facet->sh.countTrans) {
        RecordHitOnTexture(facet, momentIndex,
                           true, 2.0, 2.0);
    }
    if (/*facet->direction &&*/ facet->sh.countDirection) {
        RecordDirectionVector(facet, momentIndex);
    }
    LogHit(facet);
    ProfileFacet(facet, momentIndex,
                 true, 2.0, 2.0);
    if (facet->sh.anglemapParams.record) RecordAngleMap(facet);
}

void Particle::Reset() {
    particle.origin = Vector3d();
    particle.direction = Vector3d();
    particle.time = 0;
    particle.structure = -1;

    oriRatio = 0.0;

    nbBounces = 0;
    lastMomentIndex = 0;
    //particleId = 0;
    distanceTraveled = 0;
    generationTime = 0;
    teleportedFrom = -1;

    velocity = 0.0;
    expectedDecayMoment = 0.0;

    tmpState.Reset();
    lastHitFacet = nullptr;
    particle.lastIntersected = -1;
    //randomGenerator.SetSeed(randomGenerator.GetSeed());
    model = nullptr;
    transparentHitBuffer.clear();
    tmpFacetVars.clear();
}

bool Particle::UpdateHits(GlobalSimuState *globState, ParticleLog *particleLog, size_t timeout) {
    /*if(!globState) return false; // checked before simulation start */

    bool lastHitUpdateOK = UpdateMCHits(*globState, model->tdParams.moments.size(), timeout);
    /*totalDesorbed += tmpState.globalHits.globalHits.hit.nbDesorbed;
    tmpState.globalHits.globalHits.hit.nbDesorbed = 0;*/

    // only 1 , so no reduce necessary
    if (particleLog) UpdateLog(particleLog, timeout);

    // At last delete tmpCache
    if(lastHitUpdateOK) tmpState.Reset();

    //ResetTmpCounters();
    // only reset buffers 1..N-1
    // 0 = global buffer for reduce
    /*for(auto & tmpGlobalResult : tmpGlobalResults)
        tmpGlobalResult.Reset();*/

    return lastHitUpdateOK;
}

bool Particle::UpdateLog(ParticleLog *globalLog, size_t timeout){
    if (!tmpParticleLog.pLog.empty()) {
        //if (!LockMutex(worker->logMutex, timeout)) return false;
        if (!globalLog->tMutex.try_lock_for(std::chrono::milliseconds(timeout))) {
            return false;
        }
        size_t writeNb = model->otfParams.logLimit - globalLog->pLog.size();
        Saturate(writeNb, 0, tmpParticleLog.pLog.size());
        globalLog->pLog.insert(globalLog->pLog.begin(), tmpParticleLog.pLog.begin(), tmpParticleLog.pLog.begin() + writeNb);
        //myLogTarget = (model->otfParams.logLimit - globalLog->size()) / model->otfParams.nbProcess + 1; //+1 to avoid all threads rounding down
        globalLog->tMutex.unlock();
        tmpParticleLog.clear();
        tmpParticleLog.pLog.shrink_to_fit();
        tmpParticleLog.pLog.reserve(std::max(model->otfParams.logLimit - globalLog->pLog.size(), (size_t)0u));
        //SetLocalAndMasterState(0, GetMyStatusAsText(), false, true);
    }

    return true;
}
void Particle::RecordHit(const int &type) {
    if (tmpState.globalHits.hitCacheSize < HITCACHESIZE) {
        tmpState.globalHits.hitCache[tmpState.globalHits.hitCacheSize].pos = particle.origin;
        tmpState.globalHits.hitCache[tmpState.globalHits.hitCacheSize].type = type;
        ++tmpState.globalHits.hitCacheSize;
    }
    if(tmpState.hitBattery.buffer_per_facet)
        tmpState.hitBattery.rays[particle.lastIntersected].Add_Linear(TestRay(particle.origin, particle.direction, particle.lastIntersected));
    else
        tmpState.hitBattery.grays.Add_Linear(TestRay(particle.origin, particle.direction, particle.lastIntersected));
}

void Particle::RecordLeakPos() {
    // Source region check performed when calling this routine
    // Record leak for debugging
    RecordHit(HIT_REF);
    RecordHit(HIT_LAST);
    if (tmpState.globalHits.leakCacheSize < LEAKCACHESIZE) {
        tmpState.globalHits.leakCache[tmpState.globalHits.leakCacheSize].pos = particle.origin;
        tmpState.globalHits.leakCache[tmpState.globalHits.leakCacheSize].dir = particle.direction;
        ++tmpState.globalHits.leakCacheSize;
    }
}