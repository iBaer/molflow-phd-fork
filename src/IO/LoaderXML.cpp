//
// Created by Pascal Baehr on 20.07.20.
//

#include <sstream>
#include <set>
#include <Helper/MathTools.h>
#include <cmath>
#include <iomanip> // setprecision
#include "LoaderXML.h"
#include "TimeMoments.h"
#include "File.h"

using namespace pugi;
using namespace FlowIO;

static double loadProgress = 0.0;
void setLoadProgress(double newProgress) {
    loadProgress = newProgress;
}

void reportLoadStatus(const std::string& statusString) {
    printf("[Loader at %lf3.2%%] %s", loadProgress , statusString.c_str());
}

// Use work->InsertParametersBeforeCatalog(loadedParams);
// if loaded from GUI side
int LoaderXML::LoadGeometry(std::string inputFileName, std::shared_ptr<SimulationModel> model, double *progress) {
    if (!model->m.try_lock()) {
        return 1;
    }

    xml_document loadXML;
    auto inputFile = inputFileName.c_str();
    xml_parse_result parseResult = loadXML.load_file(inputFile); //parse xml file directly
    xml_node rootNode = loadXML.child("SimulationEnvironment");
    if(!rootNode){
        std::cerr << "XML file seems to be of older format, please generate a new file with the GUI application!"<<std::endl;
        rootNode = loadXML.root();
        //return 1;
    }
    model->sh.name = FileUtils::GetFilename(inputFile);

    xml_node geomNode = rootNode.child("Geometry");

    //Vertices
    model->sh.nbVertex = geomNode.child("Vertices").select_nodes("Vertex").size();

    model->vertices3.resize(model->sh.nbVertex); model->vertices3.shrink_to_fit();
    size_t idx = 0;
    for (xml_node vertex : geomNode.child("Vertices").children("Vertex")) {
        model->vertices3[idx].x = vertex.attribute("x").as_double();
        model->vertices3[idx].y = vertex.attribute("y").as_double();
        model->vertices3[idx].z = vertex.attribute("z").as_double();
        idx++;
    }

    //Structures
    model->sh.nbSuper = geomNode.child("Structures").select_nodes("Structure").size();
    idx = 0;
    model->structures.resize(model->sh.nbSuper);
    for (xml_node structure : geomNode.child("Structures").children("Structure")) {
        model->structures[idx].strName = structure.attribute("name").value();
        // For backward compatibilty with STR
        char tmp[256];
        sprintf(tmp, "%s.txt", model->structures[idx].strName.c_str());
        model->structures[idx].strFileName = tmp;
        idx++;
    }

    //Parameters (needs to precede facets)
    xml_node simuParamNode = rootNode.child("MolflowSimuSettings");
    bool isMolflowFile = (simuParamNode != nullptr); //if no "MolflowSimuSettings" node, it's a Synrad file

    {
        if (isMolflowFile) {
            xml_node paramNode = simuParamNode.child("Parameters");
            for (xml_node newParameter : paramNode.children("Parameter")) {
                Parameter& newPar = uInput.parameters.emplace_back();
                newPar.name = newParameter.attribute("name").as_string();
                if (newParameter.attribute("logXinterp")) {
                    newPar.logXinterp = newParameter.attribute("logXinterp").as_bool();
                } //else set to false by constructor
                if (newParameter.attribute("logYinterp")) {
                    newPar.logYinterp = newParameter.attribute("logYinterp").as_bool();
                } //else set to false by constructor
                for (xml_node newMoment : newParameter.children("Moment")) {
                    newPar.AddPair(std::make_pair(newMoment.attribute("t").as_double(),
                                                  newMoment.attribute("value").as_double()));
                }
            }
        }
        //TODO: Load parameters from catalog explicitly?
        //work->InsertParametersBeforeCatalog(loadedParams);
        model->tdParams.parameters.insert(model->tdParams.parameters.end(),uInput.parameters.begin(),uInput.parameters.end());
    }

    //Facets , load for now via temp pointers and convert to vector afterwards
    model->sh.nbFacet = geomNode.child("Facets").select_nodes("Facet").size();
    //SubprocessFac** loadFacets = (SubprocessFac **)malloc(model->sh.nbFacet * sizeof(SubprocessFac *));
    //loadFacets.reserve(model->sh.nbFacet);
    //memset(loadFacets, 0, model->sh.nbFacet * sizeof(SubprocessFac *));
    idx = 0;
    bool ignoreSumMismatch = false;
    std::vector<std::shared_ptr<SubprocessFacet>> loadFacets; // tmp facet holder
    for (xml_node facetNode : geomNode.child("Facets").children("Facet")) {
        size_t nbIndex = facetNode.child("Indices").select_nodes("Indice").size();
        if (nbIndex < 3) {
            char errMsg[128];
            sprintf(errMsg, "Facet %zd has only %zd vertices. ", idx + 1, nbIndex);
            throw Error(errMsg);
        }

        loadFacets.emplace_back(std::make_shared<SubprocessFacet>(nbIndex));
        auto prim = LoadFacet(facetNode, loadFacets[idx].get(), model->sh.nbVertex);
        prim->globalId = loadFacets.back()->globalId;
        model->primitives.emplace_back(prim);
        model->globalFacetIDs.emplace_back(loadFacets.back()->globalId);
        //Set param names for interface
        /*if (facets[idx]->sh.sticking_paramId > -1) facets[idx]->userSticking = work->parameters[facets[idx]->sh.sticking_paramId].name;
        if (facets[idx]->sh.opacity_paramId > -1) facets[idx]->userOpacity = work->parameters[facets[idx]->sh.opacity_paramId].name;
        if (facets[idx]->sh.outgassing_paramId > -1) facets[idx]->userOutgassing = work->parameters[facets[idx]->sh.outgassing_paramId].name;
*/
        idx++;
        *progress = (double)idx/(double)model->sh.nbFacet;
    }
    size_t countTri = 0;
    size_t countPoly = 0;

    for(auto& prim : model->primitives){
#if defined(COMPUTE_WITH_TRI)
        if(prim->indices.size() == 3){
            std::dynamic_pointer_cast<TriangleFacet>(prim)->vertices3 = &model->vertices3;
            ++countTri;
        }
        else
#endif
            ++countPoly;
    }

    std::cout << "Count (Tri/Poly): " << countTri << " / " << countPoly << std::endl;
    model->wp.gasMass = simuParamNode.child("Gas").attribute("mass").as_double();
    model->wp.halfLife = simuParamNode.child("Gas").attribute("halfLife").as_double();
    if (simuParamNode.child("Gas").attribute("enableDecay")) {
        model->wp.enableDecay = simuParamNode.child("Gas").attribute("enableDecay").as_bool();
    }
    else {
        model->wp.enableDecay = model->wp.halfLife < 1e100;
    }

    xml_node timeSettingsNode = simuParamNode.child("TimeSettings");
    model->wp.timeWindowSize = timeSettingsNode.attribute("timeWindow").as_double();
    // Default initialization
    /**/
    model->wp.useMaxwellDistribution = timeSettingsNode.attribute("useMaxwellDistr").as_bool();
    model->wp.calcConstantFlow = timeSettingsNode.attribute("calcConstFlow").as_bool();

    //uInput.userMoments.clear();
    xml_node userMomentsNode = timeSettingsNode.child("UserMoments");
    for (xml_node newUserEntry : userMomentsNode.children("UserEntry")) {
        char tmpExpr[512];
        double tmpWindow = 0.0;
        strcpy(tmpExpr, newUserEntry.attribute("content").as_string());
        tmpWindow = newUserEntry.attribute("window").as_double();
        if(tmpWindow==0.0){
            tmpWindow = model->wp.timeWindowSize;
        }
        uInput.userMoments.emplace_back(tmpExpr,tmpWindow);
    }
    if(TimeMoments::ParseAndCheckUserMoments(&model->tdParams.moments, &uInput.userMoments, nullptr)){
        model->m.unlock();
        return 1;
    }

    xml_node motionNode = simuParamNode.child("Motion");
    model->wp.motionType = motionNode.attribute("type").as_int();
    if (model->wp.motionType == 1) { //fixed motion
        xml_node v = motionNode.child("VelocityVector");
        model->wp.motionVector2.x = v.attribute("vx").as_double();
        model->wp.motionVector2.y = v.attribute("vy").as_double();
        model->wp.motionVector2.z = v.attribute("vz").as_double();
    }
    else if (model->wp.motionType == 2) { //rotation
        xml_node v = motionNode.child("AxisBasePoint");
        model->wp.motionVector1.x = v.attribute("x").as_double();
        model->wp.motionVector1.y = v.attribute("y").as_double();
        model->wp.motionVector1.z = v.attribute("z").as_double();
        xml_node v2 = motionNode.child("RotationVector");
        model->wp.motionVector2.x = v2.attribute("x").as_double();
        model->wp.motionVector2.y = v2.attribute("y").as_double();
        model->wp.motionVector2.z = v2.attribute("z").as_double();
    }

    xml_node globalHistNode = simuParamNode.child("Global_histograms");
    if (globalHistNode) { // Molflow version before 2.8 didn't save histograms
        xml_node nbBounceNode = globalHistNode.child("Bounces");
        if (nbBounceNode) {
            model->wp.globalHistogramParams.recordBounce=true;
            model->wp.globalHistogramParams.nbBounceBinsize=nbBounceNode.attribute("binSize").as_ullong();
            model->wp.globalHistogramParams.nbBounceMax=nbBounceNode.attribute("max").as_ullong();
        }
        xml_node distanceNode = globalHistNode.child("Distance");
        if (distanceNode) {
            model->wp.globalHistogramParams.recordDistance=true;
            model->wp.globalHistogramParams.distanceBinsize=distanceNode.attribute("binSize").as_double();
            model->wp.globalHistogramParams.distanceMax=distanceNode.attribute("max").as_double();
        }
#ifdef MOLFLOW
        xml_node timeNode = globalHistNode.child("Time");
        if (timeNode) {
            model->wp.globalHistogramParams.recordTime=true;
            model->wp.globalHistogramParams.timeBinsize=timeNode.attribute("binSize").as_double();
            model->wp.globalHistogramParams.timeMax=timeNode.attribute("max").as_double();
        }
#endif
    }

    model->tdParams.IDs = this->IDs;
    model->tdParams.CDFs = this->CDFs;
    model->facets = std::move(loadFacets);

    model->m.unlock();

    return 0;
}

std::vector<SelectionGroup> LoaderXML::LoadSelections(const std::string& inputFileName) {
    std::vector<SelectionGroup> selGroup;

    xml_document loadXML;
    auto inputFile = inputFileName.c_str();
    xml_parse_result parseResult = loadXML.load_file(inputFile); //parse xml file directly
    xml_node rootNode = loadXML.child("SimulationEnvironment");
    if(!rootNode){
        std::cerr << "XML file seems to be of older format, please generate a new file with the GUI application!"<<std::endl;
        return selGroup;
    }


    xml_node interfNode = loadXML.child("Interface");
    xml_node selNode = interfNode.child("Selections");
    //int nbS = selNode.select_nodes("Selection").size();

    selGroup.reserve(std::distance(selNode.children("Selection").begin(),selNode.children("Selection").end()));
    for (xml_node sNode : selNode.children("Selection")) {
        SelectionGroup s;
        s.name = sNode.attribute("name").as_string();
        s.selection.reserve(sNode.select_nodes("selItem").size());
        for (xml_node iNode : sNode.children("selItem"))
            s.selection.push_back(iNode.attribute("facet").as_llong());
        selGroup.push_back(s);
    }

    return selGroup;
}

int LoaderXML::LoadSimulationState(const std::string &inputFileName, std::shared_ptr<SimulationModel> model,
                                   GlobalSimuState *globState, double *progress) {

    xml_document loadXML;
    xml_parse_result parseResult = loadXML.load_file(inputFileName.c_str()); //parse xml file directly
    xml_node rootNode = loadXML.child("SimulationEnvironment");

    if(!rootNode){
        std::cerr << "XML file seems to be of older format, please generate a new file with the GUI application!"<<std::endl;
        rootNode = loadXML.root();
    }

    if (!rootNode.child("MolflowResults"))
        return 1; //simu state not saved with file

    if(!globState->tMutex.try_lock()){
        return 1;
    }

    xml_node resultNode = rootNode.child("MolflowResults");
    xml_node momentsNode = resultNode.child("Moments");
    size_t nbMoments = momentsNode.select_nodes("Moment").size(); //Contains constant flow!
    size_t facetHitsSize = (nbMoments) * sizeof(FacetHitBuffer);
    size_t m = 0;
    for (xml_node newMoment : momentsNode.children("Moment")) {

        if (m == 0) { //read global results
            xml_node globalNode = newMoment.child("Global");
            xml_node hitsNode = globalNode.child("Hits");
            globState->globalHits.globalHits.nbMCHit = hitsNode.attribute("totalHit").as_llong();
            if (hitsNode.attribute("totalHitEquiv")) {
                globState->globalHits.globalHits.nbHitEquiv = hitsNode.attribute("totalHitEquiv").as_double();
            }
            else {
                //Backward compatibility
                globState->globalHits.globalHits.nbHitEquiv = static_cast<double>(globState->globalHits.globalHits.nbMCHit);
            }
            globState->globalHits.globalHits.nbDesorbed = hitsNode.attribute("totalDes").as_llong();
            if (hitsNode.attribute("totalAbsEquiv")) {
                globState->globalHits.globalHits.nbAbsEquiv = hitsNode.attribute("totalAbsEquiv").as_double();
            }
            else {
                //Backward compatibility
                globState->globalHits.globalHits.nbAbsEquiv = hitsNode.attribute("totalAbs").as_double();
            }
            if (hitsNode.attribute("totalDist_total")) { //if it's in the new format where total/partial are separated
                globState->globalHits.distTraveled_total = hitsNode.attribute("totalDist_total").as_double();
                globState->globalHits.distTraveledTotal_fullHitsOnly = hitsNode.attribute("totalDist_fullHitsOnly").as_double();
            }
            else
                globState->globalHits.distTraveled_total = globState->globalHits.distTraveledTotal_fullHitsOnly = hitsNode.attribute("totalDist").as_double();
            globState->globalHits.nbLeakTotal = hitsNode.attribute("totalLeak").as_llong();
            //work->desorptionLimit=hitsNode.attribute("maxDesorption").as_llong();

            globState->globalHits.hitCacheSize = 0;
            xml_node hitCacheNode = globalNode.child("Hit_Cache");
            for (xml_node newHit : hitCacheNode.children("Hit")) {
                if (globState->globalHits.hitCacheSize < HITCACHESIZE) {
                    globState->globalHits.hitCache[globState->globalHits.hitCacheSize].pos.x = newHit.attribute("posX").as_double();
                    globState->globalHits.hitCache[globState->globalHits.hitCacheSize].pos.y = newHit.attribute("posY").as_double();
                    globState->globalHits.hitCache[globState->globalHits.hitCacheSize].pos.z = newHit.attribute("posZ").as_double();
                    globState->globalHits.hitCache[globState->globalHits.hitCacheSize].type = newHit.attribute("type").as_int();
                    globState->globalHits.hitCacheSize++;
                }
            }

            globState->globalHits.leakCacheSize = 0;
            xml_node leakCacheNode = globalNode.child("Leak_Cache");
            for (xml_node newLeak : leakCacheNode.children("Leak")) {
                if (globState->globalHits.leakCacheSize < LEAKCACHESIZE) {
                    globState->globalHits.leakCache[globState->globalHits.leakCacheSize].pos.x = newLeak.attribute("posX").as_double();
                    globState->globalHits.leakCache[globState->globalHits.leakCacheSize].pos.y = newLeak.attribute("posY").as_double();
                    globState->globalHits.leakCache[globState->globalHits.leakCacheSize].pos.z = newLeak.attribute("posZ").as_double();
                    globState->globalHits.leakCache[globState->globalHits.leakCacheSize].dir.x = newLeak.attribute("dirX").as_double();
                    globState->globalHits.leakCache[globState->globalHits.leakCacheSize].dir.y = newLeak.attribute("dirY").as_double();
                    globState->globalHits.leakCache[globState->globalHits.leakCacheSize].dir.z = newLeak.attribute("dirZ").as_double();
                    globState->globalHits.leakCacheSize++;
                }
            }
        } //end global node

        bool hasHistogram = model->wp.globalHistogramParams.recordBounce || model->wp.globalHistogramParams.recordDistance;
#ifdef MOLFLOW
        hasHistogram = hasHistogram || model->wp.globalHistogramParams.recordTime;
#endif
        if (hasHistogram) {
            xml_node histNode = newMoment.child("Histograms");
            if (histNode) { //Versions before 2.8 didn't save histograms
                //Retrieve histogram map from hits dp
                auto& globalHistogram = globState->globalHistograms[m];
                if (model->wp.globalHistogramParams.recordBounce) {
                    auto& nbHitsHistogram = globalHistogram.nbHitsHistogram;
                    xml_node hist = histNode.child("Bounces");
                    if (hist) {
                        size_t histSize = model->wp.globalHistogramParams.GetBounceHistogramSize();
                        size_t saveHistSize = hist.attribute("size").as_ullong();
                        if (histSize == saveHistSize) {
                            //Can do: compare saved with expected size
                            size_t h = 0;
                            for (auto bin : hist.children("Bin")) {
                                if (h < histSize) {
                                    nbHitsHistogram[h++] = bin.attribute("count").as_double();
                                }
                                else {
                                    //Treat errors
                                }
                            }
                        }
                        else {
                            //Treat errors
                        }
                    }
                }
                if (model->wp.globalHistogramParams.recordDistance) {
                    auto& distanceHistogram = globalHistogram.distanceHistogram;
                    xml_node hist = histNode.child("Distance");
                    if (hist) {
                        size_t histSize = model->wp.globalHistogramParams.GetDistanceHistogramSize();
                        size_t saveHistSize = hist.attribute("size").as_ullong();
                        if (histSize == saveHistSize) {
                            //Can do: compare saved with expected size
                            size_t h = 0;
                            for (auto bin : hist.children("Bin")) {
                                if (h < histSize) {
                                    distanceHistogram[h++] = bin.attribute("count").as_double();
                                }
                                else {
                                    //Treat errors
                                }
                            }
                        }
                        else {
                            //Treat errors
                        }
                    }
                }
                if (model->wp.globalHistogramParams.recordTime) {
                    auto& timeHistogram = globalHistogram.timeHistogram;
                    xml_node hist = histNode.child("Time");
                    if (hist) {
                        size_t histSize = model->wp.globalHistogramParams.GetTimeHistogramSize();
                        size_t saveHistSize = hist.attribute("size").as_ullong();
                        if (histSize == saveHistSize) {
                            //Can do: compare saved with expected size
                            size_t h = 0;
                            for (auto bin : hist.children("Bin")) {
                                if (h < histSize) {
                                    timeHistogram[h++] = bin.attribute("count").as_double();
                                }
                                else {
                                    //Treat errors
                                }
                            }
                        }
                        else {
                            //Treat errors
                        }
                    }
                }
            }
        }
        
        xml_node facetResultsNode = newMoment.child("FacetResults");
        for (xml_node newFacetResult : facetResultsNode.children("Facet")) {
            int facetId = newFacetResult.attribute("id").as_int();
            auto sFac = model->facets[facetId];
            xml_node facetHitNode = newFacetResult.child("Hits");
            //FacetHitBuffer* facetCounter = (FacetHitBuffer *)(buffer + loadFacets[facetId].sh.hitOffset + m * sizeof(FacetHitBuffer));
            FacetHitBuffer* facetCounter = &globState->facetStates[facetId].momentResults[m].hits;
            if (facetHitNode) { //If there are hit results for the current moment
                facetCounter->nbMCHit = facetHitNode.attribute("nbHit").as_llong();
                if (facetHitNode.attribute("nbHitEquiv")) {
                    facetCounter->nbHitEquiv = facetHitNode.attribute("nbHitEquiv").as_double();
                }
                else {
                    //Backward compatibility
                    facetCounter->nbHitEquiv = static_cast<double>(facetCounter->nbMCHit);
                }
                facetCounter->nbDesorbed = facetHitNode.attribute("nbDes").as_llong();
                if (facetHitNode.attribute("nbAbsEquiv")) {
                    facetCounter->nbAbsEquiv = facetHitNode.attribute("nbAbsEquiv").as_double();
                }
                else {
                    //Backward compatibility
                    facetCounter->nbAbsEquiv = facetHitNode.attribute("nbAbs").as_double();
                }
                facetCounter->sum_v_ort = facetHitNode.attribute("sum_v_ort").as_double();
                facetCounter->sum_1_per_ort_velocity = facetHitNode.attribute("sum_1_per_v").as_double();
                if (facetHitNode.attribute("sum_v")) {
                    facetCounter->sum_1_per_velocity = facetHitNode.attribute("sum_v").as_double();
                }
                else {
                    //Backward compatibility
                    facetCounter->sum_1_per_velocity = 4.0 * Sqr(facetCounter->nbHitEquiv + static_cast<double>(facetCounter->nbDesorbed)) / facetCounter->sum_1_per_ort_velocity;
                }

                // Do this after XML load
                /*if (model->displayedMoment == m) { //For immediate display in facet hits list and facet counter
                    facet.facetHitCache.hit = facetCounter->hit;
                }*/
            }
            else { //No hit information, so set to 0
                facetCounter->nbMCHit =
                facetCounter->nbDesorbed =
                        0;
                facetCounter->sum_v_ort =
                facetCounter->nbHitEquiv =
                facetCounter->sum_1_per_ort_velocity =
                facetCounter->sum_1_per_velocity =
                facetCounter->nbAbsEquiv =
                        0.0;
            }

            //Profiles
            auto& sh = sFac->prim->sh;
            if (sh.isProfile) {
                xml_node profileNode = newFacetResult.child("Profile");
                //ProfileSlice *profilePtr = (ProfileSlice *)(buffer + facet.sh.hitOffset + facetHitsSize + m * sizeof(ProfileSlice)*PROFILE_SIZE);
                std::vector<ProfileSlice>& profilePtr = globState->facetStates[facetId].momentResults[m].profile;

                size_t id = 0;
                for (xml_node slice : profileNode.children("Slice")) {
                    if (slice.attribute("countEquiv")) {
                        profilePtr[id].countEquiv = slice.attribute("countEquiv").as_double();
                    }
                    else {
                        //Old format before low-flux
                        profilePtr[id].countEquiv = static_cast<double>(slice.attribute("count").as_llong());
                    }
                    profilePtr[id].sum_1_per_ort_velocity = slice.attribute("sum_1_per_v").as_double();
                    profilePtr[id].sum_v_ort = slice.attribute("sum_v_ort").as_double();
                    id++;
                }
            }

            //Textures
            int profSize = (sh.isProfile) ? ((int)PROFILE_SIZE * (int)sizeof(ProfileSlice)*(1 + (int)model->tdParams.moments.size())) : 0;

            if (sh.texWidth * sh.texHeight > 0) {
                xml_node textureNode = newFacetResult.child("Texture");
                size_t texWidth_file = textureNode.attribute("width").as_llong();
                size_t texHeight_file = textureNode.attribute("height").as_llong();

                std::vector<TextureCell>& texture = globState->facetStates[facetId].momentResults[m].texture;

                std::stringstream countText, sum1perText, sumvortText;
                if (textureNode.child("countEquiv")) {
                    countText << textureNode.child_value("countEquiv");
                }
                else {
                    countText << textureNode.child_value("count");
                }
                sum1perText << textureNode.child_value("sum_1_per_v");
                sumvortText << textureNode.child_value("sum_v_ort");

                for (size_t iy = 0; iy < (Min(sh.texHeight, texHeight_file)); iy++) { //MIN: If stored texture is larger, don't read extra cells
                    for (size_t ix = 0; ix < (Min(sh.texWidth, texWidth_file)); ix++) { //MIN: If stored texture is larger, don't read extra cells
                        countText >> texture[iy*sh.texWidth + ix].countEquiv;
                        sum1perText >> texture[iy*sh.texWidth + ix].sum_1_per_ort_velocity;
                        sumvortText >> texture[iy*sh.texWidth + ix].sum_v_ort_per_area;

                    }
                    for (int ie = 0; ie < texWidth_file - sh.texWidth; ie++) {//Executed if file texture is bigger than expected texture
                        //Read extra cells from file without doing anything
                        size_t dummy_ll;
                        double dummy_d;
                        countText >> dummy_ll;
                        sum1perText >> dummy_d;
                        sumvortText >> dummy_d;

                    }
                }
                for (int ie = 0; ie < texHeight_file - sh.texHeight; ie++) {//Executed if file texture is bigger than expected texture
                    //Read extra cells ffrom file without doing anything
                    for (int iw = 0; iw < texWidth_file; iw++) {
                        size_t dummy_ll;
                        double dummy_d;
                        countText >> dummy_ll;
                        sum1perText >> dummy_d;
                        sumvortText >> dummy_d;
                    }

                }
            } //end texture

            if (sh.countDirection) {
                xml_node dirNode = newFacetResult.child("Directions");
                if (dirNode.attribute("width").as_int() != sh.texWidth ||
                    dirNode.attribute("height").as_int() != sh.texHeight) {
                    std::stringstream msg;
                    msg << "Direction texture size mismatch on facet " << facetId + 1 << ".\nExpected: " << sh.texWidth << "x" << sh.texHeight << "\n"
                        << "In file: " << dirNode.attribute("width").as_int() << "x" << dirNode.attribute("height").as_int();
                    globState->tMutex.unlock();
                    throw Error(msg.str().c_str());

                }
                /*DirectionCell *dirs = (DirectionCell *)(buffer + sh.hitOffset + facetHitsSize
                                                        + profSize + (1 + (int)model->tdParams.moments.size())*sh.texWidth*sh.texHeight * sizeof(TextureCell)
                                                        + m * sh.texWidth*sh.texHeight * sizeof(DirectionCell));*/
                std::vector<DirectionCell>& dirs = globState->facetStates[facetId].momentResults[m].direction;

                std::stringstream dirText, dirCountText;
                dirText << dirNode.child_value("vel.vectors");
                dirCountText << dirNode.child_value("count");

                for (size_t iy = 0; iy < sh.texHeight; iy++) {
                    for (size_t ix = 0; ix < sh.texWidth; ix++) {
                        std::string component;
                        std::getline(dirText, component, ',');
                        dirs[iy*sh.texWidth + ix].dir.x = std::stod(component);
                        std::getline(dirText, component, ',');
                        dirs[iy*sh.texWidth + ix].dir.y = std::stod(component);
                        dirText >> dirs[iy*sh.texWidth + ix].dir.z;
                        dirCountText >> dirs[iy*sh.texWidth + ix].count;
                    }
                }
            } //end directions

            // Facet histogram
            hasHistogram = sh.facetHistogramParams.recordBounce || sh.facetHistogramParams.recordDistance;
#ifdef MOLFLOW
            hasHistogram = hasHistogram || sh.facetHistogramParams.recordTime;
#endif
            if (hasHistogram) {
                xml_node histNode = newFacetResult.child("Histograms");
                if (histNode) { //Versions before 2.8 didn't save histograms
                    //Retrieve histogram map from hits dp
                    auto& facetHistogram = globState->facetStates[facetId].momentResults[m].histogram;
                    if (sh.facetHistogramParams.recordBounce) {
                        auto& nbHitsHistogram = facetHistogram.nbHitsHistogram;
                        xml_node hist = histNode.child("Bounces");
                        if (hist) {
                            size_t histSize = sh.facetHistogramParams.GetBounceHistogramSize();
                            size_t saveHistSize = hist.attribute("size").as_ullong();
                            if (histSize == saveHistSize) {
                                //Can do: compare saved with expected size
                                size_t h = 0;
                                for (auto bin : hist.children("Bin")) {
                                    if (h < histSize) {
                                        nbHitsHistogram[h++] = bin.attribute("count").as_double();
                                    }
                                    else {
                                        //Treat errors
                                    }
                                }
                            }
                            else {
                                //Treat errors
                            }
                        }
                    }
                    if (sh.facetHistogramParams.recordDistance) {
                        auto& distanceHistogram = facetHistogram.distanceHistogram;
                        xml_node hist = histNode.child("Distance");
                        if (hist) {
                            size_t histSize = sh.facetHistogramParams.GetDistanceHistogramSize();
                            size_t saveHistSize = hist.attribute("size").as_ullong();
                            if (histSize == saveHistSize) {
                                //Can do: compare saved with expected size
                                size_t h = 0;
                                for (auto bin : hist.children("Bin")) {
                                    if (h < histSize) {
                                        distanceHistogram[h++] = bin.attribute("count").as_double();
                                    }
                                    else {
                                        //Treat errors
                                    }
                                }
                            }
                            else {
                                //Treat errors
                            }
                        }
                    }
                    if (sh.facetHistogramParams.recordTime) {
                        auto& timeHistogram = facetHistogram.timeHistogram;
                        xml_node hist = histNode.child("Time");
                        if (hist) {
                            size_t histSize = sh.facetHistogramParams.GetTimeHistogramSize();
                            size_t saveHistSize = hist.attribute("size").as_ullong();
                            if (histSize == saveHistSize) {
                                //Can do: compare saved with expected size
                                size_t h = 0;
                                for (auto bin : hist.children("Bin")) {
                                    if (h < histSize) {
                                        timeHistogram[h++] = bin.attribute("count").as_double();
                                    }
                                    else {
                                        //Treat errors
                                    }
                                }
                            }
                            else {
                                //Treat errors
                            }
                        }
                    }
                }
            }
            setLoadProgress((double) ((m * model->sh.nbFacet) + facetId) / ((double) nbMoments * model->sh.nbFacet));
            if(progress) *progress = (double) ((m * model->sh.nbFacet) + facetId) / ((double) nbMoments * model->sh.nbFacet);
        } //end facetResult
        m++;
    } //end moment

    /*xml_node minMaxNode = resultNode.child("TextureMinMax");
    globState->globalHits.texture_limits[0].min.all = minMaxNode.child("With_constant_flow").child("Pressure").attribute("min").as_double();
    globState->globalHits.texture_limits[0].max.all = minMaxNode.child("With_constant_flow").child("Pressure").attribute("max").as_double();
    globState->globalHits.texture_limits[1].min.all = minMaxNode.child("With_constant_flow").child("Density").attribute("min").as_double();
    globState->globalHits.texture_limits[1].max.all = minMaxNode.child("With_constant_flow").child("Density").attribute("max").as_double();
    globState->globalHits.texture_limits[2].min.all = minMaxNode.child("With_constant_flow").child("Imp.rate").attribute("min").as_double();
    globState->globalHits.texture_limits[2].max.all = minMaxNode.child("With_constant_flow").child("Imp.rate").attribute("max").as_double();
    globState->globalHits.texture_limits[0].min.moments_only = minMaxNode.child("Moments_only").child("Pressure").attribute("min").as_double();
    globState->globalHits.texture_limits[0].max.moments_only = minMaxNode.child("Moments_only").child("Pressure").attribute("max").as_double();
    globState->globalHits.texture_limits[1].min.moments_only = minMaxNode.child("Moments_only").child("Density").attribute("min").as_double();
    globState->globalHits.texture_limits[1].max.moments_only = minMaxNode.child("Moments_only").child("Density").attribute("max").as_double();
    globState->globalHits.texture_limits[2].min.moments_only = minMaxNode.child("Moments_only").child("Imp.rate").attribute("min").as_double();
    globState->globalHits.texture_limits[2].max.moments_only = minMaxNode.child("Moments_only").child("Imp.rate").attribute("max").as_double();*/

    //TODO: globState->globalHits. = this->angleMapCache;
    globState->tMutex.unlock();
    return 0;
}

std::shared_ptr<GeomPrimitive> LoaderXML::LoadFacet(pugi::xml_node facetNode, SubprocessFacet *facet, size_t nbTotalVertices) {
    int idx = 0;
    bool ignoreSumMismatch = true;
    int facetId = facetNode.attribute("id").as_int();
    facet->globalId = facetId;

    std::shared_ptr<GeomPrimitive> prim;
    {
        auto index_nodes = facetNode.child("Indices").children("Indice");
        size_t nIndices = std::distance(index_nodes.begin(), index_nodes.end());

        std::vector<size_t> indices;
        for (xml_node indice : index_nodes) {
            indices.push_back(indice.attribute("vertex").as_int() + 0); //+ vertexOffset;
            if (indices.back() >= nbTotalVertices) {
                char err[128];
                sprintf(err, "Facet %d refers to vertex %d which doesn't exist", facetId + 1, idx + 1);
                throw Error(err);
            }
            idx++;
        }

#if defined(COMPUTE_WITH_TRI)
        if(nIndices == 3){
            prim = std::make_shared<TriangleFacet>();
            std::dynamic_pointer_cast<TriangleFacet>(prim)->indices.swap(indices);
        }
        else
#endif
        {
            prim = std::make_shared<Facet>();
            std::dynamic_pointer_cast<Facet>(prim)->indices.swap(indices);
        }
        prim->globalId = facetId;
    }
    facet->prim = prim.get();

    auto& sh = prim->sh;
    sh.opacity = facetNode.child("Opacity").attribute("constValue").as_double();
    sh.is2sided = facetNode.child("Opacity").attribute("is2sided").as_int();
    sh.superIdx = facetNode.child("Structure").attribute("inStructure").as_int();
    sh.superDest = facetNode.child("Structure").attribute("linksTo").as_int();
    sh.teleportDest = facetNode.child("Teleport").attribute("target").as_int();

    // TODO: Only parse Molflow files
    sh.sticking = facetNode.child("Sticking").attribute("constValue").as_double();
    sh.sticking_paramId = facetNode.child("Sticking").attribute("parameterId").as_int();
    sh.opacity_paramId = facetNode.child("Opacity").attribute("parameterId").as_int();
    sh.outgassing = facetNode.child("Outgassing").attribute("constValue").as_double();
    sh.desorbType = facetNode.child("Outgassing").attribute("desType").as_int();
    sh.desorbTypeN = facetNode.child("Outgassing").attribute("desExponent").as_double();
    sh.outgassing_paramId = facetNode.child("Outgassing").attribute("parameterId").as_int();
    bool hasOutgassingFile = facetNode.child("Outgassing").attribute("hasOutgassingFile").as_bool();
    sh.useOutgassingFile = facetNode.child("Outgassing").attribute("useOutgassingFile").as_bool();
    sh.temperature = facetNode.child("Temperature").attribute("value").as_double();
    sh.accomodationFactor = facetNode.child("Temperature").attribute("accFactor").as_double();
    xml_node reflNode = facetNode.child("Reflection");
    if (reflNode.attribute("diffusePart") && reflNode.attribute("specularPart")) { //New format
        sh.reflection.diffusePart = reflNode.attribute("diffusePart").as_double();
        sh.reflection.specularPart = reflNode.attribute("specularPart").as_double();
        if (reflNode.attribute("cosineExponent")) {
            sh.reflection.cosineExponent = reflNode.attribute("cosineExponent").as_double();
        }
        else {
            sh.reflection.cosineExponent = 0.0; //uniform
        }
    }
    else { //old XML format: fully diffuse / specular / uniform reflections
        int oldReflType = reflNode.attribute("type").as_int();
        if (oldReflType == REFLECTION_DIFFUSE) {
            sh.reflection.diffusePart = 1.0;
            sh.reflection.specularPart = 0.0;
        }
        else if (oldReflType == REFLECTION_SPECULAR) {
            sh.reflection.diffusePart = 0.0;
            sh.reflection.specularPart = 1.0;
        }
        else { //Uniform
            sh.reflection.diffusePart = 0.0;
            sh.reflection.specularPart = 0.0;
            sh.reflection.cosineExponent = 0.0;
        }
    }

    if (reflNode.attribute("enableSojournTime")) {
        sh.enableSojournTime = reflNode.attribute("enableSojournTime").as_bool();
        if (!reflNode.attribute("sojournFreq")) {//Backward compatibility with ver. before 2.6.25
            sh.sojournFreq = 1.0 / reflNode.attribute("sojournTheta0").as_double();
            sh.sojournE = 8.31 * reflNode.attribute("sojournE").as_double();
        }
        else {
            sh.sojournFreq = reflNode.attribute("sojournFreq").as_double();
            sh.sojournE = reflNode.attribute("sojournE").as_double();
        }
    }
    else {
        //Already set to default when calling Molflow::LoadFile()
    }
    sh.isMoving = facetNode.child("Motion").attribute("isMoving").as_bool();
    xml_node recNode = facetNode.child("Recordings");
    sh.profileType = recNode.child("Profile").attribute("type").as_int();
    xml_node incidentAngleNode = recNode.child("IncidentAngleMap");
    if (incidentAngleNode) {
        sh.anglemapParams.record = recNode.child("IncidentAngleMap").attribute("record").as_bool();
        sh.anglemapParams.phiWidth = recNode.child("IncidentAngleMap").attribute("phiWidth").as_ullong();
        sh.anglemapParams.thetaLimit = recNode.child("IncidentAngleMap").attribute("thetaLimit").as_double();
        sh.anglemapParams.thetaLowerRes = recNode.child("IncidentAngleMap").attribute("thetaLowerRes").as_ullong();
        sh.anglemapParams.thetaHigherRes = recNode.child("IncidentAngleMap").attribute("thetaHigherRes").as_ullong();
    }
    xml_node texNode = recNode.child("Texture");
    sh.texWidth_precise = texNode.attribute("texDimX").as_double();
    sh.texHeight_precise = texNode.attribute("texDimY").as_double();
    sh.countDes = texNode.attribute("countDes").as_bool();
    sh.countAbs = texNode.attribute("countAbs").as_bool();
    sh.countRefl = texNode.attribute("countRefl").as_bool();
    sh.countTrans = texNode.attribute("countTrans").as_bool();
    sh.countDirection = texNode.attribute("countDir").as_bool();
    sh.countACD = texNode.attribute("countAC").as_bool();
    
    xml_node outgNode = facetNode.child("DynamicOutgassing");
    if ((hasOutgassingFile) && outgNode && outgNode.child("map")) {
        facet->ogMap.outgassingMapWidth = outgNode.attribute("width").as_int();
        facet->ogMap.outgassingMapHeight = outgNode.attribute("height").as_int();
        if (outgNode.attribute("ratioU")) { //New format supporting non-square textures
            facet->ogMap.outgassingFileRatioU = outgNode.attribute("ratioU").as_double();
            facet->ogMap.outgassingFileRatioV = outgNode.attribute("ratioV").as_double();
        }
        else { //Old format for square textures
            facet->ogMap.outgassingFileRatioU = facet->ogMap.outgassingFileRatioV = outgNode.attribute("ratio").as_double();
        }
        facet->ogMap.totalDose = outgNode.attribute("totalDose").as_double();
        sh.totalOutgassing = outgNode.attribute("totalOutgassing").as_double();
        facet->ogMap.totalFlux = outgNode.attribute("totalFlux").as_double();

        double sum = 0.0;

        std::stringstream outgText;
        outgText << outgNode.child_value("map");

        auto& ogMap = facet->ogMap;
        std::vector<double>(ogMap.outgassingMapWidth*ogMap.outgassingMapHeight).swap(ogMap.outgassingMap);

        for (size_t iy = 0; iy < ogMap.outgassingMapHeight; iy++) {
            for (size_t ix = 0; ix < ogMap.outgassingMapWidth; ix++) {
                outgText >> ogMap.outgassingMap[iy*ogMap.outgassingMapWidth + ix];
                sum += ogMap.outgassingMap[iy*ogMap.outgassingMapWidth + ix];
            }
        }
        if (!ignoreSumMismatch && !IsEqual(sum, sh.totalOutgassing)) {
            std::stringstream msg; msg << std::setprecision(8);
            msg << "Facet " << facetId + 1 << ":\n";
            msg << "The total dynamic outgassing (" << 10.0 * sh.totalOutgassing << " mbar.l/s)\n";
            msg << "doesn't match the sum of the dynamic outgassing cells (" << 10.0 * sum << " mbar.l/s).";
            /*if (1 == GLMessageBox::Display(msg.str(), "Dynamic outgassing mismatch", { "OK","Ignore rest" },GLDLG_ICONINFO))
                ignoreSumMismatch = true;*/
            std::cerr << msg.str() << std::endl;
            ignoreSumMismatch = true; // TODO: For now silence after one
        }
    }
    else {
        hasOutgassingFile = sh.useOutgassingFile = false; //if outgassing map was incorrect, don't use it
        facet->ogMap.outgassingMapWidth = 0;
        facet->ogMap.outgassingMapHeight = 0;
        facet->ogMap.outgassingFileRatioU = 0.0;
        facet->ogMap.outgassingFileRatioV = 0.0;
        facet->ogMap.totalDose = 0.0;
        sh.totalOutgassing = 0.0;
        facet->ogMap.totalFlux = 0.0;
    }
    xml_node angleMapNode = facetNode.child("IncidentAngleMap");
    if (angleMapNode && angleMapNode.child("map") && angleMapNode.attribute("angleMapThetaLimit")) {

        sh.anglemapParams.phiWidth = angleMapNode.attribute("angleMapPhiWidth").as_ullong();
        sh.anglemapParams.thetaLimit = angleMapNode.attribute("angleMapThetaLimit").as_double();
        sh.anglemapParams.thetaLowerRes = angleMapNode.attribute("angleMapThetaLowerRes").as_ullong();
        sh.anglemapParams.thetaHigherRes = angleMapNode.attribute("angleMapThetaHigherRes").as_ullong();

        std::stringstream angleText;
        angleText << angleMapNode.child_value("map");


        //std::map<size_t,std::vector<size_t>> angleMapCache;
        //size_t* angleMapCache = (size_t*)malloc(sh.anglemapParams.GetDataSize());
        //angleMapCache.emplace(std::make_pair(facet->globalId,std::vector<size_t>()));
        auto& angleMap = facet->angleMap.pdf;
        try {
            angleMap.clear(); angleMap.resize( sh.anglemapParams.phiWidth * (sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes));

        }
        catch(...) {
            std::stringstream err;
            err << "Not enough memory for incident angle map on facet ";
            throw Error(err.str().c_str());
        }

        for (size_t iy = 0; iy < (sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes); iy++) {
            for (size_t ix = 0; ix < sh.anglemapParams.phiWidth; ix++) {
                angleText >> angleMap[iy*sh.anglemapParams.phiWidth + ix];
            }
        }
        sh.anglemapParams.hasRecorded = true;
    }
    else {
        sh.anglemapParams.hasRecorded = false; //if angle map was incorrect, don't use it
        if (sh.desorbType == DES_ANGLEMAP) sh.desorbType = DES_NONE;
    }

    std::tuple<bool,bool> viewSettings; // texture, volume visible
    bool textureVisible = facetNode.child("ViewSettings").attribute("textureVisible").as_bool();
    bool volumeVisible = facetNode.child("ViewSettings").attribute("volumeVisible").as_bool();
    uInput.facetViewSettings.emplace_back(std::make_tuple(textureVisible, volumeVisible));

    xml_node facetHistNode = facetNode.child("Histograms");
    if (facetHistNode) { // Molflow version before 2.8 didn't save histograms
        xml_node nbBounceNode = facetHistNode.child("Bounces");
        if (nbBounceNode) {
            sh.facetHistogramParams.recordBounce=true;
            sh.facetHistogramParams.nbBounceBinsize=nbBounceNode.attribute("binSize").as_ullong();
            sh.facetHistogramParams.nbBounceMax=nbBounceNode.attribute("max").as_ullong();
        }
        xml_node distanceNode = facetHistNode.child("Distance");
        if (distanceNode) {
            sh.facetHistogramParams.recordDistance=true;
            sh.facetHistogramParams.distanceBinsize=distanceNode.attribute("binSize").as_double();
            sh.facetHistogramParams.distanceMax=distanceNode.attribute("max").as_double();
        }
#ifdef MOLFLOW
        xml_node timeNode = facetHistNode.child("Time");
        if (timeNode) {
            sh.facetHistogramParams.recordTime=true;
            sh.facetHistogramParams.timeBinsize=timeNode.attribute("binSize").as_double();
            sh.facetHistogramParams.timeMax=timeNode.attribute("max").as_double();
        }
#endif
    }

    //Update flags
    sh.isProfile = (sh.profileType != PROFILE_NONE);
    //wp.isOpaque = (wp.opacity != 0.0);
    sh.isTextured = ((sh.texWidth_precise * sh.texHeight_precise) > 0);

    // Do some fixes
    bool hasAnyTexture = sh.countDes || sh.countAbs || sh.countRefl || sh.countTrans || sh.countACD;
    if (!sh.isTextured && (hasAnyTexture)) {
        sh.countDes = false;
        sh.countAbs = false;
        sh.countRefl = false;
        sh.countTrans = false;
        sh.countACD = false;
        
        std::stringstream msg; msg << std::setprecision(8);
        msg << "Facet (#"<< facetId << ") has no valid mesh, but active texture counters: removing...\n";
        std::cerr << msg.str();
    }

    return prim;
}

/*
void Loader::MoveFacetsToStructures(SimulationModel* model) {
    model->structures.resize(model->sh.nbSuper);
    for (size_t i = 0; i < model->sh.nbFacet; i++) { //Necessary because facets is not (yet) a vector in the interface
        if (loadFacets[i].sh.superIdx == -1) { //Facet in all structures
            for (auto &s : model->structures) {
                s.facets.emplace_back(loadFacets[i]);
                s.facets.back().globalId = i;
            }
        } else {
            model->structures[loadFacets[i].sh.superIdx].facets.emplace_back(loadFacets[i]); //Assign to structure
            model->structures[loadFacets[i].sh.superIdx].facets.back().globalId = i;
        }
    }
}*/
