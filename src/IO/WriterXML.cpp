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

#include <iomanip>      // std::setprecision
#include <sstream>
#include <Helper/StringHelper.h>
#include <Helper/ConsoleLogger.h>
#include "PugiXML/pugixml.hpp"

#include "WriterXML.h"
#include "versionId.h"
#include "Simulation/MolflowSimFacet.h"


using namespace FlowIO;
using namespace pugi;

void WriterXML::setWriteProgress(double newProgress) {
    writeProgress = newProgress;
}

void WriterXML::reportWriteStatus(const std::string &statusString) const {
    Log::console_msg(2, "[{}] {} [{:3.2f}%]", Util::getTimepointString(), statusString, writeProgress);
}

void WriterXML::reportNewWriteStatus(const std::string &statusString, double newProgress) {
    setWriteProgress(newProgress);
    Log::console_msg(2, "\r[{}] {} [{:3.2f}%]", Util::getTimepointString(), statusString, writeProgress);
}

void WriterXML::finishWriteStatus(const std::string &statusString) {
    setWriteProgress(100.0);
    Log::console_msg(2, "\r[{}] {} [{:3.2f}%]\n", Util::getTimepointString(), statusString, 100.0);
}

xml_node WriterXML::GetRootNode(xml_document &saveDoc) {
    // Check whether we do a old to new format update
    // If yes, move old scheme to new scheme by adding the root node
    if(update){
        bool oldFormatUsed = false;
        auto rootNode = saveDoc.document_element();
        if(!saveDoc.child("SimulationEnvironment")){
            rootNode = saveDoc.root();
            if(rootNode.child("Geometry")){
                oldFormatUsed = true;
            }
        }

        if(oldFormatUsed && !useOldXMLFormat){
            xml_document newDoc;
            auto newHead = newDoc.append_child("SimulationEnvironment");
            newDoc.append_attribute("type") = "molflow";
            newDoc.append_attribute("version") = appVersionId;
            for(auto& node : rootNode.children()){
                newHead.append_copy(node);
            }

            saveDoc.reset(newDoc);
        }
    }

    xml_node rootNode;
    if (useOldXMLFormat) {
        rootNode = saveDoc.root();
    } else {
        if(update) {
            rootNode = saveDoc.document_element();
            rootNode = saveDoc.child("SimulationEnvironment");
        }
        if(!rootNode) {
            if(!saveDoc.child("SimulationEnvironment")) {
                rootNode = saveDoc.append_child("SimulationEnvironment");
                rootNode.append_attribute("type") = "molflow";
                rootNode.append_attribute("version") = appVersionId;
            }
            else
                rootNode = saveDoc.document_element();
        }
    }

    return rootNode;
}

void WriterXML::SaveGeometry(pugi::xml_document &saveDoc, std::shared_ptr<MolflowSimulationModel> &model,
                             const std::vector<size_t> &selection) {
    xml_node rootNode = GetRootNode(saveDoc);

    if(update)
        rootNode.remove_child("Geometry");
    xml_node geomNode = rootNode.prepend_child("Geometry");
    geomNode.append_child("Vertices").append_attribute(
            "nb") = model->vertices3.size(); //creates Vertices node, adds nb attribute and sets its value to wp.nbVertex
    for (size_t i = 0; i < model->vertices3.size(); i++) {
        //prg->SetProgress(0.166*((double)i / (double)model->vertices3.size()));
        xml_node v = geomNode.child("Vertices").append_child("Vertex");
        v.append_attribute("id") = i;
        v.append_attribute("x") = model->vertices3[i].x;
        v.append_attribute("y") = model->vertices3[i].y;
        v.append_attribute("z") = model->vertices3[i].z;
    }

    geomNode.append_child("Facets");
    
    if (selection.empty()) {
        geomNode.child("Facets").append_attribute("nb") = model->facets.size();
        for (size_t i = 0; i < model->facets.size(); i++) {
            //prg->SetProgress(0.166 + ((double)i / (double)model->facets.size()) *0.166);
            //if (!saveSelected || model->facets[i]->selected) {
            xml_node f = geomNode.child("Facets").append_child("Facet");
            f.append_attribute("id") = i;
            SaveFacet(f, (MolflowSimFacet*) model->facets[i].get(), model->vertices3.size()); //model->facets[i]->SaveXML_geom(f);
            //}
        }
    }
    else
    {
        geomNode.child("Facets").append_attribute("nb") =selection.size();
        for (size_t i = 0; i < selection.size();i++) {
            //prg->SetProgress(0.166 + ((double)i / (double)model->facets.size()) *0.166);
            //if (!saveSelected || model->facets[i]->selected) {
            xml_node f = geomNode.child("Facets").append_child("Facet");
            f.append_attribute("id") = i; //Different from global facet id
            SaveFacet(f, (MolflowSimFacet*) model->facets[selection[i]].get(), model->vertices3.size()); //model->facets[i]->SaveXML_geom(f);
            //}
        }
    }

    geomNode.append_child("Structures").append_attribute("nb") = model->sh.nbSuper;

    if (model->structures.size() == model->sh.nbSuper) {
        for (size_t i = 0; i < model->sh.nbSuper; i++) {
            xml_node s = geomNode.child("Structures").append_child("Structure");
            s.append_attribute("id") = i;
            s.append_attribute("name") = model->structures[i].strName.c_str();

        }
    }

    // Simulation Settings
    if(update)
        rootNode.remove_child("MolflowSimuSettings");
    xml_node simuParamNode = rootNode.insert_child_after("MolflowSimuSettings",geomNode);

    simuParamNode.append_child("Gas").append_attribute("mass") = model->wp.gasMass;
    simuParamNode.child("Gas").append_attribute(
            "enableDecay") = (int) model->wp.enableDecay; //backward compatibility: 0 or 1
    simuParamNode.child("Gas").append_attribute("halfLife") = model->wp.halfLife;

    xml_node timeSettingsNode = simuParamNode.append_child("TimeSettings");

    xml_node userMomentsNode = timeSettingsNode.append_child("UserMoments");
    userMomentsNode.append_attribute("nb") = uInput.userMoments.size();
    for (size_t i = 0; i < uInput.userMoments.size(); i++) {
        xml_node newUserEntry = userMomentsNode.append_child("UserEntry");
        newUserEntry.append_attribute("id") = i;
        newUserEntry.append_attribute("content") = uInput.userMoments[i].first.c_str();
        newUserEntry.append_attribute("window") = uInput.userMoments[i].second;
    }

    timeSettingsNode.append_attribute("timeWindow") = model->wp.timeWindowSize;
    timeSettingsNode.append_attribute(
            "useMaxwellDistr") = (int) model->wp.useMaxwellDistribution; //backward compatibility: 0 or 1
    timeSettingsNode.append_attribute(
            "calcConstFlow") = (int) model->wp.calcConstantFlow; //backward compatibility: 0 or 1

    xml_node motionNode = simuParamNode.append_child("Motion");
    motionNode.append_attribute("type") = model->wp.motionType;
    if (model->wp.motionType == 1) { //fixed motion
        xml_node v = motionNode.append_child("VelocityVector");
        v.append_attribute("vx") = model->wp.motionVector2.x;
        v.append_attribute("vy") = model->wp.motionVector2.y;
        v.append_attribute("vz") = model->wp.motionVector2.z;
    } else if (model->wp.motionType == 2) { //rotation
        xml_node v = motionNode.append_child("AxisBasePoint");
        v.append_attribute("x") = model->wp.motionVector1.x;
        v.append_attribute("y") = model->wp.motionVector1.y;
        v.append_attribute("z") = model->wp.motionVector1.z;
        xml_node v2 = motionNode.append_child("RotationVector");
        v2.append_attribute("x") = model->wp.motionVector2.x;
        v2.append_attribute("y") = model->wp.motionVector2.y;
        v2.append_attribute("z") = model->wp.motionVector2.z;
    }

    xml_node paramNode = simuParamNode.append_child("Parameters");
    size_t nonCatalogParameters = 0;

    for (auto &parameter : uInput.parameters) {
        if (!parameter.fromCatalog) { //Don't save catalog parameters
            xml_node newParameter = paramNode.append_child("Parameter");
            newParameter.append_attribute("id") = nonCatalogParameters;
            newParameter.append_attribute("name") = parameter.name.c_str();
            newParameter.append_attribute("nbMoments") = (int) parameter.GetSize();
            newParameter.append_attribute("logXinterp") = parameter.logXinterp;
            newParameter.append_attribute("logYinterp") = parameter.logYinterp;
            for (size_t m = 0; m < parameter.GetSize(); m++) {
                xml_node newMoment = newParameter.append_child("Moment");
                newMoment.append_attribute("id") = m;
                newMoment.append_attribute("t") = parameter.GetX(m);
                newMoment.append_attribute("value") = parameter.GetY(m);
            }
            nonCatalogParameters++;
        }
    }
    paramNode.append_attribute("nb") = nonCatalogParameters;
    xml_node globalHistNode = simuParamNode.append_child("Global_histograms");
    if (model->wp.globalHistogramParams.recordBounce) {
        xml_node nbBounceNode = globalHistNode.append_child("Bounces");
        nbBounceNode.append_attribute("binSize") = model->wp.globalHistogramParams.nbBounceBinsize;
        nbBounceNode.append_attribute("max") = model->wp.globalHistogramParams.nbBounceMax;
    }
    if (model->wp.globalHistogramParams.recordDistance) {
        xml_node distanceNode = globalHistNode.append_child("Distance");
        distanceNode.append_attribute("binSize") = model->wp.globalHistogramParams.distanceBinsize;
        distanceNode.append_attribute("max") = model->wp.globalHistogramParams.distanceMax;
    }
#ifdef MOLFLOW
    if (model->wp.globalHistogramParams.recordTime) {
        xml_node timeNode = globalHistNode.append_child("Time");
        timeNode.append_attribute("binSize") = model->wp.globalHistogramParams.timeBinsize;
        timeNode.append_attribute("max") = model->wp.globalHistogramParams.timeMax;
    }
#endif
}

// Save XML document to file
bool
WriterXML::SaveXMLToFile(xml_document &saveDoc, const std::string &outputFileName) {
    if (!saveDoc.save_file(outputFileName.c_str())) {
        std::cerr << "Error writing XML file." << std::endl; //successful save
        return false;
    }
    return true;
}

// Directly append to file (load + save)
bool
WriterXML::SaveSimulationState(const std::string &outputFileName, std::shared_ptr<MolflowSimulationModel> model, GlobalSimuState &globState) {
    xml_document saveDoc;
    xml_parse_result parseResult = saveDoc.load_file(outputFileName.c_str()); //parse xml file directly

    SaveSimulationState(saveDoc, model, globState);

    if (!saveDoc.save_file(outputFileName.c_str())) {
        std::cerr << "Error writing XML file." << std::endl; //successful save
        return false;
    }
    return true;
}

// Append to open XML node
bool WriterXML::SaveSimulationState(xml_document &saveDoc, std::shared_ptr<MolflowSimulationModel> model, GlobalSimuState &globState) {
    //xml_parse_result parseResult = saveDoc.load_file(outputFileName.c_str()); //parse xml file directly

    xml_node rootNode = GetRootNode(saveDoc);
    rootNode.remove_child("MolflowResults"); // clear previous results to replace with new status

    xml_node resultNode = rootNode.append_child("MolflowResults");
    xml_node momentsNode = resultNode.append_child("Moments");
    momentsNode.append_attribute("nb") = model->tdParams.moments.size() + 1;
    size_t facetHitsSize = (1 + model->tdParams.moments.size()) * sizeof(FacetHitBuffer);

    reportWriteStatus("Writing simulation results...");
    for (size_t m = 0; m <= model->tdParams.moments.size(); m++) {
        //setWriteProgress(0.5 + 0.5 * (double) m / (1.0 + (double) model->tdParams.moments.size()));
        reportNewWriteStatus("Writing simulation results...", 0.5 + 0.5 * (double) m / (1.0 + (double) model->tdParams.moments.size()));
        xml_node newMoment = momentsNode.append_child("Moment");
        newMoment.append_attribute("id") = m;
        if (m == 0) {
            newMoment.append_attribute("time") = "Constant flow";
            newMoment.append_attribute("timeWindow") = 0;
        } else {
            newMoment.append_attribute("time") = model->tdParams.moments[m - 1].first;
            newMoment.append_attribute("timeWindow") = model->tdParams.moments[m - 1].second;
        }

        if (m == 0) { //Write global results. Later these results will probably be time-dependent as well.
            xml_node globalNode = newMoment.append_child("Global");

            xml_node hitsNode = globalNode.append_child("Hits");
            hitsNode.append_attribute("totalHit") = globState.globalHits.globalHits.nbMCHit;
            hitsNode.append_attribute("totalHitEquiv") = globState.globalHits.globalHits.nbHitEquiv;
            hitsNode.append_attribute("totalDes") = globState.globalHits.globalHits.nbDesorbed;
            hitsNode.append_attribute("totalAbsEquiv") = globState.globalHits.globalHits.nbAbsEquiv;
            hitsNode.append_attribute("totalDist_total") = globState.globalHits.distTraveled_total;
            hitsNode.append_attribute("totalDist_fullHitsOnly") = globState.globalHits.distTraveledTotal_fullHitsOnly;
            hitsNode.append_attribute("totalLeak") = globState.globalHits.nbLeakTotal;
            hitsNode.append_attribute("maxDesorption") = model->otfParams.desorptionLimit;

            xml_node hitCacheNode = globalNode.append_child("Hit_Cache");
            hitCacheNode.append_attribute("nb") = globState.globalHits.hitCacheSize;

            for (size_t i = 0; i < std::max(globState.globalHits.hitCacheSize,HITCACHESIZE); i++) {
                xml_node newHit = hitCacheNode.append_child("Hit");
                newHit.append_attribute("id") = i;
                newHit.append_attribute("posX") = globState.globalHits.hitCache[i].pos.x;
                newHit.append_attribute("posY") = globState.globalHits.hitCache[i].pos.y;
                newHit.append_attribute("posZ") = globState.globalHits.hitCache[i].pos.z;
                newHit.append_attribute("type") = globState.globalHits.hitCache[i].type;
            }

            xml_node leakCacheNode = globalNode.append_child("Leak_Cache");
            leakCacheNode.append_attribute("nb") = globState.globalHits.leakCacheSize;
            for (size_t i = 0; i < globState.globalHits.leakCacheSize; i++) {
                xml_node newLeak = leakCacheNode.append_child("Leak");
                newLeak.append_attribute("id") = i;
                newLeak.append_attribute("posX") = globState.globalHits.leakCache[i].pos.x;
                newLeak.append_attribute("posY") = globState.globalHits.leakCache[i].pos.y;
                newLeak.append_attribute("posZ") = globState.globalHits.leakCache[i].pos.z;
                newLeak.append_attribute("dirX") = globState.globalHits.leakCache[i].dir.x;
                newLeak.append_attribute("dirY") = globState.globalHits.leakCache[i].dir.y;
                newLeak.append_attribute("dirZ") = globState.globalHits.leakCache[i].dir.z;
            }
        } //end global node

        bool hasHistogram =
                model->wp.globalHistogramParams.recordBounce || model->wp.globalHistogramParams.recordDistance;
#ifdef MOLFLOW
        hasHistogram = hasHistogram || model->wp.globalHistogramParams.recordTime;
#endif
        if (hasHistogram) {
            xml_node histNode = newMoment.append_child("Histograms");
            //Retrieve histogram map from hits dp
            auto &globalHist = globState.globalHistograms[m];
            if (model->wp.globalHistogramParams.recordBounce) {
                auto &nbHitsHistogram = globalHist.nbHitsHistogram;
                xml_node hist = histNode.append_child("Bounces");
                size_t histSize = model->wp.globalHistogramParams.GetBounceHistogramSize();
                hist.append_attribute("size") = histSize;
                hist.append_attribute(
                        "binSize") = model->wp.globalHistogramParams.nbBounceBinsize; //redundancy for human-reading or export
                hist.append_attribute(
                        "max") = model->wp.globalHistogramParams.nbBounceMax; //redundancy for human-reading or export
                for (size_t h = 0; h < histSize; h++) {
                    xml_node bin = hist.append_child("Bin");
                    auto value = bin.append_attribute("start");
                    if (h == histSize - 1) value = "overRange";
                    else value = h * model->wp.globalHistogramParams.nbBounceBinsize;
                    bin.append_attribute("count") = nbHitsHistogram[h];
                }
            }
            if (model->wp.globalHistogramParams.recordDistance) {
                auto &distanceHistogram = globalHist.distanceHistogram;
                xml_node hist = histNode.append_child("Distance");
                size_t histSize = model->wp.globalHistogramParams.GetDistanceHistogramSize();
                hist.append_attribute("size") = histSize;
                hist.append_attribute(
                        "binSize") = model->wp.globalHistogramParams.distanceBinsize; //redundancy for human-reading or export
                hist.append_attribute(
                        "max") = model->wp.globalHistogramParams.distanceMax; //redundancy for human-reading or export
                for (size_t h = 0; h < histSize; h++) {
                    xml_node bin = hist.append_child("Bin");
                    auto value = bin.append_attribute("start");
                    if (h == histSize - 1) value = "overRange";
                    else value = h * model->wp.globalHistogramParams.distanceBinsize;
                    bin.append_attribute("count") = distanceHistogram[h];
                }
            }
            if (model->wp.globalHistogramParams.recordTime) {
                auto &timeHistogram = globalHist.timeHistogram;
                xml_node hist = histNode.append_child("Time");
                size_t histSize = model->wp.globalHistogramParams.GetTimeHistogramSize();
                hist.append_attribute("size") = histSize;
                hist.append_attribute(
                        "binSize") = model->wp.globalHistogramParams.timeBinsize; //redundancy for human-reading or export
                hist.append_attribute(
                        "max") = model->wp.globalHistogramParams.timeMax; //redundancy for human-reading or export
                for (size_t h = 0; h < histSize; h++) {
                    xml_node bin = hist.append_child("Bin");
                    auto value = bin.append_attribute("start");
                    if (h == histSize - 1) value = "overRange";
                    else value = h * model->wp.globalHistogramParams.timeBinsize;
                    bin.append_attribute("count") = timeHistogram[h];
                }
            }
        }

        xml_node facetResultsNode = newMoment.append_child("FacetResults");

        for (auto &fac : model->facets) {
            auto &sFac = *fac;
            //SimulationFacet& f = model->structures[0].facets[0].;
            xml_node newFacetResult = facetResultsNode.append_child("Facet");
            newFacetResult.append_attribute("id") = sFac.globalId;

            xml_node facetHitNode = newFacetResult.append_child("Hits");
            //FacetHitBuffer* facetCounter = (FacetHitBuffer *)(buffer + sFac.sh.hitOffset + m * sizeof(FacetHitBuffer));
            const auto &facetCounter = globState.facetStates[sFac.globalId].momentResults[m].hits;

            facetHitNode.append_attribute("nbHit") = facetCounter.nbMCHit;
            facetHitNode.append_attribute("nbHitEquiv") = facetCounter.nbHitEquiv;
            facetHitNode.append_attribute("nbDes") = facetCounter.nbDesorbed;
            facetHitNode.append_attribute("nbAbsEquiv") = facetCounter.nbAbsEquiv;
            facetHitNode.append_attribute("sum_v_ort") = facetCounter.sum_v_ort;
            facetHitNode.append_attribute("sum_1_per_v") = facetCounter.sum_1_per_ort_velocity;
            facetHitNode.append_attribute("sum_v") = facetCounter.sum_1_per_velocity;

            if (sFac.sh.isProfile) {
                xml_node profileNode = newFacetResult.append_child("Profile");
                profileNode.append_attribute("size") = PROFILE_SIZE;
                //ProfileSlice *pr = (ProfileSlice *)(buffer + sFac.sh.hitOffset + facetHitsSize + m * sizeof(ProfileSlice)*PROFILE_SIZE);
                const auto &pr = globState.facetStates[sFac.globalId].momentResults[m].profile;

                for (int p = 0; p < PROFILE_SIZE; p++) {
                    xml_node slice = profileNode.append_child("Slice");
                    slice.append_attribute("id") = p;
                    slice.append_attribute("countEquiv") = pr[p].countEquiv;
                    slice.append_attribute("sum_1_per_v") = pr[p].sum_1_per_ort_velocity;
                    slice.append_attribute("sum_v_ort") = pr[p].sum_v_ort;
                }
            }

            size_t profSize = (sFac.sh.isProfile) ? (PROFILE_SIZE * sizeof(ProfileSlice) *
                                                     (1 + model->tdParams.moments.size())) : 0;
            size_t height = sFac.sh.texHeight;
            size_t width = sFac.sh.texWidth;

            if (sFac.sh.texWidth * sFac.sh.texHeight > 0) {
                xml_node textureNode = newFacetResult.append_child("Texture");
                textureNode.append_attribute("width") = sFac.sh.texWidth;
                textureNode.append_attribute("height") = sFac.sh.texHeight;

                //TextureCell *texture = (TextureCell *)(buffer + sFac.sh.hitOffset + facetHitsSize + profSize + m * w*h * sizeof(TextureCell));
                const auto &texture = globState.facetStates[sFac.globalId].momentResults[m].texture;

                std::stringstream countText, sum1perText, sumvortText;
                countText << '\n'; //better readability in file
                sum1perText << std::setprecision(8) << '\n';
                sumvortText << std::setprecision(8) << '\n';

                for (size_t iy = 0; iy < height; iy++) {
                    for (size_t ix = 0; ix < width; ix++) {
                        countText << texture[iy * sFac.sh.texWidth + ix].countEquiv << '\t';
                        sum1perText << texture[iy * sFac.sh.texWidth + ix].sum_1_per_ort_velocity << '\t';
                        sumvortText << texture[iy * sFac.sh.texWidth + ix].sum_v_ort_per_area << '\t';
                    }
                    countText << '\n';
                    sum1perText << '\n';
                    sumvortText << '\n';
                }
                textureNode.append_child("count").append_child(node_cdata).set_value(countText.str().c_str());
                textureNode.append_child("sum_1_per_v").append_child(node_cdata).set_value(sum1perText.str().c_str());
                textureNode.append_child("sum_v_ort").append_child(node_cdata).set_value(sumvortText.str().c_str());

            } //end texture

            if (sFac.sh.countDirection/* && sFac.dirCache*/) {
                xml_node dirNode = newFacetResult.append_child("Directions");
                dirNode.append_attribute("width") = sFac.sh.texWidth;
                dirNode.append_attribute("height") = sFac.sh.texHeight;

                //DirectionCell *dirs = (DirectionCell *)(buffer + sFac.sh.hitOffset + facetHitsSize + profSize + (1 + (int)model->tdParams.moments.size())*w*h * sizeof(TextureCell) + m * w*h * sizeof(DirectionCell));
                const auto &dirs = globState.facetStates[sFac.globalId].momentResults[m].direction;

                std::stringstream dirText, dirCountText;
                dirText << std::setprecision(8) << '\n'; //better readability in file
                dirCountText << '\n';

                for (size_t iy = 0; iy < height; iy++) {
                    for (size_t ix = 0; ix < width; ix++) {
                        dirText << dirs[iy * sFac.sh.texWidth + ix].dir.x << ",";
                        dirText << dirs[iy * sFac.sh.texWidth + ix].dir.y << ",";
                        dirText << dirs[iy * sFac.sh.texWidth + ix].dir.z << "\t";
                        dirCountText << dirs[iy * sFac.sh.texWidth + ix].count << "\t";

                    }
                    dirText << "\n";
                    dirCountText << "\n";
                }
                dirNode.append_child("vel.vectors").append_child(node_cdata).set_value(dirText.str().c_str());
                dirNode.append_child("count").append_child(node_cdata).set_value(dirCountText.str().c_str());
            } //end directions

            //Facet histograms (1 per moment) comes here
            bool hasFHistogram =
                    sFac.sh.facetHistogramParams.recordBounce || sFac.sh.facetHistogramParams.recordDistance;
#ifdef MOLFLOW
            hasFHistogram = hasFHistogram || sFac.sh.facetHistogramParams.recordTime;
#endif
            if (hasFHistogram) {
                xml_node histNode = newFacetResult.append_child("Histograms");
                //Retrieve histogram map from hits dp
                auto &histogram = globState.facetStates[sFac.globalId].momentResults[m].histogram;
                if (sFac.sh.facetHistogramParams.recordBounce) {
                    auto &nbHitsHistogram = histogram.nbHitsHistogram;
                    xml_node hist = histNode.append_child("Bounces");
                    size_t histSize = sFac.sh.facetHistogramParams.GetBounceHistogramSize();
                    hist.append_attribute("size") = histSize;
                    hist.append_attribute(
                            "binSize") = sFac.sh.facetHistogramParams.nbBounceBinsize; //redundancy for human-reading or export
                    hist.append_attribute(
                            "max") = sFac.sh.facetHistogramParams.nbBounceMax; //redundancy for human-reading or export
                    for (size_t h = 0; h < histSize; h++) {
                        xml_node bin = hist.append_child("Bin");
                        auto value = bin.append_attribute("start");
                        if (h == histSize - 1) value = "overRange";
                        else value = h * sFac.sh.facetHistogramParams.nbBounceBinsize;
                        bin.append_attribute("count") = nbHitsHistogram[h];
                    }
                }
                if (sFac.sh.facetHistogramParams.recordDistance) {
                    auto &distanceHistogram = histogram.distanceHistogram;
                    xml_node hist = histNode.append_child("Distance");
                    size_t histSize = sFac.sh.facetHistogramParams.GetDistanceHistogramSize();
                    hist.append_attribute("size") = histSize;
                    hist.append_attribute(
                            "binSize") = sFac.sh.facetHistogramParams.distanceBinsize; //redundancy for human-reading or export
                    hist.append_attribute(
                            "max") = sFac.sh.facetHistogramParams.distanceMax; //redundancy for human-reading or export
                    for (size_t h = 0; h < histSize; h++) {
                        xml_node bin = hist.append_child("Bin");
                        auto value = bin.append_attribute("start");
                        if (h == histSize - 1) value = "overRange";
                        else value = h * sFac.sh.facetHistogramParams.distanceBinsize;
                        bin.append_attribute("count") = distanceHistogram[h];
                    }
                }
                if (sFac.sh.facetHistogramParams.recordTime) {
                    auto &timeHistogram = histogram.timeHistogram;
                    xml_node hist = histNode.append_child("Time");
                    size_t histSize = sFac.sh.facetHistogramParams.GetTimeHistogramSize();
                    hist.append_attribute("size") = histSize;
                    hist.append_attribute(
                            "binSize") = sFac.sh.facetHistogramParams.timeBinsize; //redundancy for human-reading or export
                    hist.append_attribute(
                            "max") = sFac.sh.facetHistogramParams.timeMax; //redundancy for human-reading or export
                    for (size_t h = 0; h < histSize; h++) {
                        xml_node bin = hist.append_child("Bin");
                        auto value = bin.append_attribute("start");
                        if (h == histSize - 1) value = "overRange";
                        else value = h * sFac.sh.facetHistogramParams.timeBinsize;
                        bin.append_attribute("count") = timeHistogram[h];
                    }
                }
            }

        }
    }

    //Texture Min/Max
    xml_node minMaxNode = resultNode.append_child("TextureMinMax");
    minMaxNode.append_child("With_constant_flow").append_child("Pressure").append_attribute("min") = 0.0;
    minMaxNode.child("With_constant_flow").child("Pressure").append_attribute("max") = 0.0;
    minMaxNode.child("With_constant_flow").append_child("Density").append_attribute("min") = 0.0;
    minMaxNode.child("With_constant_flow").child("Density").append_attribute("max") = 0.0;
    minMaxNode.child("With_constant_flow").append_child("Imp.rate").append_attribute("min") = 0.0;
    minMaxNode.child("With_constant_flow").child("Imp.rate").append_attribute("max") = 0.0;

    minMaxNode.append_child("Moments_only").append_child("Pressure").append_attribute("min") = 0.0;
    minMaxNode.child("Moments_only").child("Pressure").append_attribute("max") = 0.0;
    minMaxNode.child("Moments_only").append_child("Density").append_attribute("min") = 0.0;
    minMaxNode.child("Moments_only").child("Density").append_attribute("max") = 0.0;
    minMaxNode.child("Moments_only").append_child("Imp.rate").append_attribute("min") = 0.0;
    minMaxNode.child("Moments_only").child("Imp.rate").append_attribute("max") = 0.0;

    finishWriteStatus("Writing simulation results...");

    return true;
}

/**
* \brief To save facet data for the geometry in XML
* \param facetNode XML node representing a facet
*/
void WriterXML::SaveFacet(pugi::xml_node facetNode, MolflowSimFacet *facet, size_t nbTotalVertices) {
    xml_node e = facetNode.append_child("Sticking");
    e.append_attribute("constValue") = facet->sh.sticking;
    e.append_attribute("parameterId") = facet->sh.sticking_paramId;

    e = facetNode.append_child("Opacity");
    e.append_attribute("constValue") = facet->sh.opacity;
    e.append_attribute("parameterId") = facet->sh.opacity_paramId;
    e.append_attribute("is2sided") = (int) facet->sh.is2sided; //backward compatibility: 0 or 1

    e = facetNode.append_child("Outgassing");
    e.append_attribute("constValue") = facet->sh.outgassing;
    e.append_attribute("parameterId") = facet->sh.outgassing_paramId;
    e.append_attribute("desType") = facet->sh.desorbType;
    e.append_attribute("desExponent") = facet->sh.desorbTypeN;
    e.append_attribute(
            "hasOutgassingFile") = (int) facet->sh.useOutgassingFile;//TODO:(int)hasOutgassingFile; //backward compatibility: 0 or 1
    e.append_attribute("useOutgassingFile") = (int) facet->sh.useOutgassingFile; //backward compatibility: 0 or 1

    e = facetNode.append_child("Temperature");
    e.append_attribute("value") = facet->sh.temperature;
    e.append_attribute("accFactor") = facet->sh.accomodationFactor;

    e = facetNode.append_child("Reflection");

    e.append_attribute("diffusePart") = facet->sh.reflection.diffusePart;
    e.append_attribute("specularPart") = facet->sh.reflection.specularPart;
    e.append_attribute("cosineExponent") = facet->sh.reflection.cosineExponent;

    //For backward compatibility
    if (facet->sh.reflection.diffusePart > 0.99) {
        e.append_attribute("type") = REFLECTION_DIFFUSE;
    } else if (facet->sh.reflection.specularPart > 0.99) {
        e.append_attribute("type") = REFLECTION_SPECULAR;
    } else
        e.append_attribute("type") = REFLECTION_UNIFORM;

    e.append_attribute("enableSojournTime") = (int) facet->sh.enableSojournTime; //backward compatibility: 0 or 1
    e.append_attribute("sojournFreq") = facet->sh.sojournFreq;
    e.append_attribute("sojournE") = facet->sh.sojournE;

    e = facetNode.append_child("Structure");
    e.append_attribute("inStructure") = facet->sh.superIdx;
    e.append_attribute("linksTo") = facet->sh.superDest;

    e = facetNode.append_child("Teleport");
    e.append_attribute("target") = facet->sh.teleportDest;

    e = facetNode.append_child("Motion");
    e.append_attribute("isMoving") = (int) facet->sh.isMoving; //backward compatibility: 0 or 1

    e = facetNode.append_child("Recordings");
    xml_node t = e.append_child("Profile");
    t.append_attribute("type") = facet->sh.profileType;
    switch (facet->sh.profileType) {
        case 0:
            t.append_attribute("name") = "none";
            break;
        case 1:
            t.append_attribute("name") = "pressure u";
            break;
        case 2:
            t.append_attribute("name") = "pressure v";
            break;
        case 3:
            t.append_attribute("name") = "angular";
            break;
        case 4:
            t.append_attribute("name") = "speed";
            break;
        case 5:
            t.append_attribute("name") = "ortho.v";
            break;
        case 6:
            t.append_attribute("name") = "tang.v.";
            break;
    }
    t = e.append_child("Texture");
    //assert(!(cellPropertiesIds == NULL && (facet->sh.countAbs || facet->sh.countDes || facet->sh.countRefl || facet->sh.countTrans))); //Count texture on non-existent texture

    t.append_attribute("hasMesh") = (facet->sh.countAbs || facet->sh.countDes || facet->sh.countRefl ||
                                     facet->sh.countTrans);
    t.append_attribute("texDimX") = facet->sh.texWidth_precise;
    t.append_attribute("texDimY") = facet->sh.texHeight_precise;
    t.append_attribute("countDes") = (int) facet->sh.countDes; //backward compatibility: 0 or 1
    t.append_attribute("countAbs") = (int) facet->sh.countAbs; //backward compatibility: 0 or 1
    t.append_attribute("countRefl") = (int) facet->sh.countRefl; //backward compatibility: 0 or 1
    t.append_attribute("countTrans") = (int) facet->sh.countTrans; //backward compatibility: 0 or 1
    t.append_attribute("countDir") = (int) facet->sh.countDirection; //backward compatibility: 0 or 1
    t.append_attribute("countAC") = (int) facet->sh.countACD; //backward compatibility: 0 or 1

    if (facet->sh.anglemapParams.record) {
        t = e.append_child("IncidentAngleMap");
        t.append_attribute("record") = facet->sh.anglemapParams.record;
        t.append_attribute("phiWidth") = facet->sh.anglemapParams.phiWidth;
        t.append_attribute("thetaLimit") = facet->sh.anglemapParams.thetaLimit;
        t.append_attribute("thetaLowerRes") = facet->sh.anglemapParams.thetaLowerRes;
        t.append_attribute("thetaHigherRes") = facet->sh.anglemapParams.thetaHigherRes;
    }

    if (!uInput.facetViewSettings.empty()) {
        e = facetNode.append_child("ViewSettings");
        e.append_attribute("textureVisible") = (int) std::get<0>(
                uInput.facetViewSettings[facet->globalId]); //backward compatibility: 0 or 1
        e.append_attribute("volumeVisible") = (int) std::get<1>(
                uInput.facetViewSettings[facet->globalId]); //backward compatibility: 0 or 1
    }

    facetNode.append_child("Indices").append_attribute("nb") = facet->sh.nbIndex;
    for (size_t i = 0; i < facet->sh.nbIndex; i++) {
        xml_node indice = facetNode.child("Indices").append_child("Indice");
        indice.append_attribute("id") = i;
        indice.append_attribute("vertex") = facet->indices[i];
    }

    if (facet->sh.useOutgassingFile) { // hasOutgassingFile
        xml_node textureNode = facetNode.append_child("DynamicOutgassing");
        textureNode.append_attribute("width") = facet->ogMap.outgassingMapWidth;
        textureNode.append_attribute("height") = facet->ogMap.outgassingMapHeight;
        textureNode.append_attribute("ratioU") = facet->ogMap.outgassingFileRatioU;
        textureNode.append_attribute("ratioV") = facet->ogMap.outgassingFileRatioV;
        textureNode.append_attribute("totalDose") = facet->ogMap.totalDose;
        textureNode.append_attribute("totalOutgassing") = facet->sh.totalOutgassing;
        textureNode.append_attribute("totalFlux") = facet->ogMap.totalFlux;

        std::stringstream outgText;
        outgText << std::setprecision(8);
        outgText << '\n'; //better readability in file
        for (int iy = 0; iy < facet->ogMap.outgassingMapHeight; iy++) {
            for (int ix = 0; ix < facet->ogMap.outgassingMapWidth; ix++) {
                outgText << facet->ogMap.outgassingMap[iy * facet->ogMap.outgassingMapWidth + ix] << '\t';
            }
            outgText << '\n';
        }
        textureNode.append_child("map").append_child(node_cdata).set_value(outgText.str().c_str());

    } //end texture

    if (!facet->angleMap.pdf.empty()) {
        xml_node textureNode = facetNode.append_child("IncidentAngleMap");
        textureNode.append_attribute("angleMapPhiWidth") = facet->sh.anglemapParams.phiWidth;
        textureNode.append_attribute("angleMapThetaLimit") = facet->sh.anglemapParams.thetaLimit;
        textureNode.append_attribute("angleMapThetaLowerRes") = facet->sh.anglemapParams.thetaLowerRes;
        textureNode.append_attribute("angleMapThetaHigherRes") = facet->sh.anglemapParams.thetaHigherRes;

        std::stringstream angleText;
        angleText << '\n'; //better readability in file
        for (int iy = 0;
             iy < (facet->sh.anglemapParams.thetaLowerRes + facet->sh.anglemapParams.thetaHigherRes); iy++) {
            for (int ix = 0; ix < facet->sh.anglemapParams.phiWidth; ix++) {
                angleText << facet->angleMap.pdf[iy * facet->sh.anglemapParams.phiWidth + ix] << '\t';
            }
            angleText << '\n';
        }
        textureNode.append_child("map").append_child(node_cdata).set_value(angleText.str().c_str());

    } //end angle map

    xml_node histNode = facetNode.append_child("Histograms");
    if (facet->sh.facetHistogramParams.recordBounce) {
        xml_node nbBounceNode = histNode.append_child("Bounces");
        nbBounceNode.append_attribute("binSize") = facet->sh.facetHistogramParams.nbBounceBinsize;
        nbBounceNode.append_attribute("max") = facet->sh.facetHistogramParams.nbBounceMax;
    }
    if (facet->sh.facetHistogramParams.recordDistance) {
        xml_node distanceNode = histNode.append_child("Distance");
        distanceNode.append_attribute("binSize") = facet->sh.facetHistogramParams.distanceBinsize;
        distanceNode.append_attribute("max") = facet->sh.facetHistogramParams.distanceMax;
    }
#ifdef MOLFLOW
    if (facet->sh.facetHistogramParams.recordTime) {
        xml_node timeNode = histNode.append_child("Time");
        timeNode.append_attribute("binSize") = facet->sh.facetHistogramParams.timeBinsize;
        timeNode.append_attribute("max") = facet->sh.facetHistogramParams.timeMax;
    }
#endif
}

WriterXML::WriterXML(bool useOldXMLFormat, bool update) : useOldXMLFormat(useOldXMLFormat), update(update){

}
