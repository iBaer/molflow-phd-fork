//
// Created by pbahr on 04/10/2019.
//

// common MF helper tools
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include "optix7.h"
#include "SimulationControllerGPU.h"
#include "helper_output.h"
#include "helper_math.h"
#include "GPUDefines.h"

SimulationControllerGPU::SimulationControllerGPU(){
    optixHandle = nullptr;
    model = nullptr;
}

SimulationControllerGPU::~SimulationControllerGPU(){
    if(optixHandle){
        CloseSimulation();
    }
    if(model){
        delete model;
    }
}


/**
 *
 * @return 1=could not load GPU Sim, 0=successfully loaded
 */
int SimulationControllerGPU::LoadSimulation(flowgpu::Model* loaded_model, size_t launchSize) {
    if(loaded_model == nullptr)
        return 1;

    /*uint2 newSize = make_uint2(launchSize,1);
    if(newSize != kernelDimensions){
        kernelDimensions = newSize;
    }*/
    kernelDimensions = make_uint2(launchSize,1);

    try {
        model = loaded_model;
        optixHandle = new flowgpu::SimulationOptiX(loaded_model, kernelDimensions);
    } catch (std::runtime_error& e) {
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
 * Init random numbers and start RT launch
 * @return 1=could not load GPU Sim, 0=successfully loaded
 */
static uint32_t runCount = 0;
int SimulationControllerGPU::RunSimulation() {

    try {
        // for testing only generate and upload random numbers once
        // generate new numbers whenever necessary, recursion = TraceProcessing only, poly checks only for ray generation with polygons
        if(runCount%(NB_RAND/(8+MAX_DEPTH*2+NB_INPOLYCHECKS*2))==0){
#ifdef DEBUG
            //std::cout << "#flowgpu: generating random numbers at run #" << runCount << std::endl;
#endif
            optixHandle->generateRand();
        }
        optixHandle->launchMolecules();
        ++runCount;
    } catch (std::runtime_error& e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
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
    try {
        optixHandle->downloadDataFromDevice(&data); //download tmp counters
        IncreaseGlobalCounters(&data); //increase global counters
        optixHandle->resetDeviceBuffers(); //reset tmp counters
        if(writeData) WriteDataToFile("hitcounters.txt");
        if(printData) PrintData();
        if(printDataParent) PrintDataForParent();
        if(printCounters) PrintTotalCounters();
        return 0;//GetTotalHits();
    } catch (std::runtime_error& e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
}

void SimulationControllerGPU::IncreaseGlobalCounters(HostData* tempData){

    //facet hit counters + miss
    for(unsigned int i = 0; i < data.facetHitCounters.size(); i++) {
        unsigned int facIndex = i%this->model->nbFacets_total;
        globalCounter.facetHitCounters[facIndex].nbMCHit += tempData->facetHitCounters[i].nbMCHit;
        globalCounter.facetHitCounters[facIndex].nbDesorbed += tempData->facetHitCounters[i].nbDesorbed;
        globalCounter.facetHitCounters[facIndex].nbAbsEquiv += tempData->facetHitCounters[i].nbAbsEquiv;

        globalCounter.facetHitCounters[facIndex].nbHitEquiv += tempData->facetHitCounters[i].nbHitEquiv;
        globalCounter.facetHitCounters[facIndex].sum_v_ort += tempData->facetHitCounters[i].sum_v_ort;
        globalCounter.facetHitCounters[facIndex].sum_1_per_velocity += tempData->facetHitCounters[i].sum_1_per_velocity;
        globalCounter.facetHitCounters[facIndex].sum_1_per_ort_velocity += tempData->facetHitCounters[i].sum_1_per_ort_velocity;
    }

    globalCounter.leakCounter[0] += tempData->leakCounter[0];

    //textures
    if(!tempData->texels.empty()) {
        for (auto&[id, texels] : globalCounter.textures) {
            for (auto &mesh : model->triangle_meshes) {
                int parentFacetId = -1;
                for (auto &facet : mesh->poly) {
                    if(parentFacetId == facet.parentIndex) break;
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
        }
    }

    //profiles
    if(!tempData->profileSlices.empty()) {
        for (auto&[id, profiles] : globalCounter.profiles) {
            for (auto &mesh : model->triangle_meshes) {
                int parentFacetId = -1;
                for (auto &facet : mesh->poly) {
                    if(parentFacetId == facet.parentIndex) break;
                    if ((facet.profProps.profileType != flowgeom::PROFILE_FLAGS::noProfile) && (id == facet.parentIndex)) {
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
        }
    }
}

void SimulationControllerGPU::Resize(){
    //data.hit.resize(kernelDimensions.x*kernelDimensions.y);
    data.facetHitCounters.resize(model->nbFacets_total * CORESPERSM * WARPSCHEDULERS);
    data.texels.resize(model->textures.size());
    data.profileSlices.resize(model->profiles.size());
    data.leakCounter.resize(1);

    globalCounter.facetHitCounters.resize(model->nbFacets_total);
    globalCounter.leakCounter.resize(1);
    for(auto& mesh : model->triangle_meshes) {
        int lastTexture = -1;
        int lastProfile = -1;
        for (auto &facet : mesh->poly) {

            // has texture?
            if ((facet.texProps.textureFlags) &&
                (lastTexture < (int) facet.parentIndex)) { // prevent duplicates
                unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;
                std::vector<Texel64> texels(width*height);
                globalCounter.textures.insert(std::pair<uint32_t,std::vector<Texel64>>(facet.parentIndex,std::move(texels)));
            }

            // has profile?
            if ((facet.profProps.profileType != flowgeom::PROFILE_FLAGS::noProfile) &&
                (lastProfile < (int) facet.parentIndex)) {
                std::vector<Texel64> texels(PROFILE_SIZE);
                globalCounter.profiles.insert(std::pair<uint32_t,std::vector<Texel64>>(facet.parentIndex,std::move(texels)));
            }
        }
    }

#ifdef DEBUGCOUNT
    data.detCounter.resize(NCOUNTBINS);
    data.uCounter.resize(NCOUNTBINS);
    data.vCounter.resize(NCOUNTBINS);
#endif

#ifdef DEBUGPOS
    data.positions.resize(NBCOUNTS*kernelDimensions.x*kernelDimensions.y);
    data.posOffset.resize(kernelDimensions.x*kernelDimensions.y);
#endif
}

/*! download the rendered color buffer and return the total amount of hits (= followed rays) */
void SimulationControllerGPU::PrintDataForParent()
{
    // Find amount of Polygons, we don't have this information anymore
    unsigned int maxPoly = 0;
    for(auto& mesh : model->triangle_meshes){
        for(auto& facet : mesh->poly){
            maxPoly = std::max(maxPoly,facet.parentIndex);
        }
    }

    std::vector<unsigned long long int> counterMCHit(maxPoly+1, 0);
    std::vector<unsigned long long int> counterDesorp(maxPoly+1, 0);
    std::vector<double> counterAbsorp(maxPoly+1, 0);

    for(unsigned int i = 0; i < data.facetHitCounters.size(); i++) {
        unsigned int facIndex = i%this->model->nbFacets_total;
        unsigned int facParent = model->triangle_meshes[0]->poly[facIndex].parentIndex;
        counterMCHit[facParent] += data.facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        counterDesorp[facParent] += data.facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        counterAbsorp[facParent] += data.facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    for(unsigned int i = 0; i <= maxPoly; i++){
        if(counterMCHit[i] > 0 || counterAbsorp[i] > 0 || counterDesorp[i] > 0)
            std::cout << i+1 << " " << counterMCHit[i] << " " << counterDesorp[i] << " " << static_cast<unsigned long long int>(counterAbsorp[i]) << std::endl;
    }

    for(auto& mesh : model->triangle_meshes){
        int lastTexture = -1;
        for(auto& facet : mesh->poly){
            if((facet.texProps.textureFlags & flowgeom::TEXTURE_FLAGS::countRefl) && (lastTexture<(int)facet.parentIndex)){
                std::cout << "Texture for #"<<facet.parentIndex << std::endl << " ";
                unsigned int total = 0;
                unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;

                for(unsigned int h = 0; h < height; ++h){
                    for(unsigned int w = 0; w < width; ++w){
                        unsigned int index = w+h*model->facetTex[facet.texProps.textureOffset].texWidth  + model->facetTex[facet.texProps.textureOffset].texelOffset;
                        std::cout << data.texels[index].countEquiv << "  ";
                        total += data.texels[index].countEquiv;
                    }
                    std::cout << std::endl << " ";
                }
                std::cout << std::endl;
                std::cout << "  total: "<<total << std::endl;

                lastTexture = facet.parentIndex;
            }
        }
    }
}

/*! download the rendered color buffer and return the total amount of hits (= followed rays) */
void SimulationControllerGPU::PrintData()
{
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

    /*for(int i=0;i<data.detCounter.size();i++) std::cout << "" << ((float)i/NBCOUNTS)*(DETHIGH - DETLOW) + DETLOW << " " << data.detCounter[i] << std::endl;
    for(int i=0;i<data.uCounter.size();i++) std::cout << "" << ((float)i/NBCOUNTS)*(UHIGH - ULOW) + ULOW << " " << data.uCounter[i] << std::endl;
    for(int i=0;i<data.vCounter.size();i++) std::cout << "" << ((float)i/NBCOUNTS)*(VHIGH - VLOW) + VLOW << " " << data.vCounter[i] << std::endl;*/

#endif

#ifdef DEBUGPOS

    int nbPos = NBCOUNTS;

    const int hitPositionsPerMol = 30;
    for(int i=0;i<data.positions.size();){
        //std::cout << i/(NBCOUNTS) << " " << data.posOffset[i/(NBCOUNTS)] << " ";
        std::cout <<"{";
        for(int pos=0;pos<hitPositionsPerMol;pos++){
            size_t index = i/(NBCOUNTS)*NBCOUNTS+pos;
            std::cout <<"{"<<data.positions[index].x << "," << data.positions[index].y << "," << data.positions[index].z <<"}";
            if(pos != hitPositionsPerMol-1) std::cout <<",";
            //std::cout << data.positions[index].x << "," << data.positions[index].y << "," << data.positions[index].z <<"   ";
        }
        i+=nbPos; // jump to next molecule/thread
        std::cout <<"},"<<std::endl;
    }
#endif

    std::vector<unsigned int> counterMCHit(this->model->nbFacets_total, 0);
    std::vector<unsigned int> counterDesorp(this->model->nbFacets_total, 0);
    std::vector<double> counterAbsorp(this->model->nbFacets_total, 0);

    for(unsigned int i = 0; i < data.facetHitCounters.size(); i++) {
        unsigned int facIndex = i%this->model->nbFacets_total;
        counterMCHit[facIndex] += data.facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        counterDesorp[facIndex] += data.facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        counterAbsorp[facIndex] += data.facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)

        /*if(data.facetHitCounters[i].nbMCHit == 0*//* || data.facetHitCounters[i].nbAbsEquiv == 0*//*){
            std::cout << "["<<i/(this->model->nbFacets_total)<<"] on facet #"<<i%this->model->nbFacets_total<<" has total hits >>> "<< data.facetHitCounters[i].nbMCHit<< " / total abs >>> " << data.facetHitCounters[i].nbAbsEquiv<<" ---> "<< i<<std::endl;
        }*/
    }

    for(unsigned int i = 0; i < this->model->nbFacets_total; i++){
        if(counterMCHit[i] > 0 || counterAbsorp[i] > 0 || counterDesorp[i] > 0)
            std::cout << i+1 << " " << counterMCHit[i] << " " << counterDesorp[i] << " " << static_cast<unsigned int>(counterAbsorp[i]) << std::endl;
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
    std::cout << " total miss >>> "<< *data.leakCounter.data()<< " -- miss/hit ratio: "<<(double)(*data.leakCounter.data()) / total_counter <<std::endl;*/
}

/*! download the rendered color buffer and return the total amount of hits (= followed rays) */
void SimulationControllerGPU::PrintTotalCounters()
{
    unsigned long long int total_counter = 0;
    unsigned long long int total_abs = 0;
    double total_absd = 0;
    unsigned long long int total_des = 0;

    for(unsigned int i = 0; i < globalCounter.facetHitCounters.size(); i++) {
        total_counter += globalCounter.facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        total_des += globalCounter.facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        total_absd += globalCounter.facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
    }

    std::cout << " total hits >>> "<< total_counter<<std::endl;
    std::cout << " total  des >>> "<< total_des<<std::endl;
    std::cout << " total  abs >>> "<< static_cast<unsigned long long int>(total_absd) <<std::endl;
    std::cout << " total miss >>> "<< *globalCounter.leakCounter.data()<< " -- miss/hit ratio: "<<static_cast<double>(*globalCounter.leakCounter.data()) / total_counter <<std::endl;
}

/*! download the rendered color buffer and return the total amount of hits (= followed rays) */
void SimulationControllerGPU::WriteDataToFile(std::string fileName)
{
    uint32_t nbFacets = this->model->nbFacets_total;

#ifdef DEBUGCOUNT
        std::ofstream detfile,ufile,vfile;
        detfile.open ("det_counter.txt");
        ufile.open ("u_counter.txt");
        vfile.open ("v_counter.txt");

        for(int i=0;i<data.detCounter.size();i++) detfile << "" << ((float)i/NBCOUNTS)*(DETHIGH - DETLOW) + DETLOW << " " << data.detCounter[i] << std::endl;
        for(int i=0;i<data.uCounter.size();i++) ufile << "" << ((float)i/NBCOUNTS)*(UHIGH - ULOW) + ULOW << " " << data.uCounter[i] << std::endl;
        for(int i=0;i<data.vCounter.size();i++) vfile << "" << ((float)i/NBCOUNTS)*(VHIGH - VLOW) + VLOW << " " << data.vCounter[i] << std::endl;

        detfile.close();
        ufile.close();
        vfile.close();
#endif

#ifdef DEBUGPOS
        std::ofstream posFile;
        posFile.open ("debug_positions.txt");

        int nbPos = NBCOUNTS;

        const int hitPositionsPerMol = 30;
        for(int i=0;i<data.positions.size();){
            //posFile << i/(NBCOUNTS) << " " << data.posOffset[i/(NBCOUNTS)] << " ";
            posFile <<"{";
            for(int pos=0;pos<hitPositionsPerMol;pos++){
                size_t index = i/(NBCOUNTS)*NBCOUNTS+pos;
                posFile <<"{"<<data.positions[index].x << "," << data.positions[index].y << "," << data.positions[index].z <<"}";
                if(pos != hitPositionsPerMol-1) posFile <<",";
            }
            i+=nbPos; // jump to next molecule/thread
            posFile <<"},"<<std::endl;
        }
        posFile.close();
#endif

    std::vector<uint64_t> counterMCHit(nbFacets, 0);
    std::vector<uint64_t> counterDesorp(nbFacets, 0);
    std::vector<double> counterAbsorp(nbFacets, 0);


    //std::ofstream facetCounterEveryFile("every"+fileName);

    for(unsigned int i = 0; i < globalCounter.facetHitCounters.size(); i++) {
        unsigned int facIndex = i%nbFacets;
        counterMCHit[facIndex] += globalCounter.facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
        counterDesorp[facIndex] += globalCounter.facetHitCounters[i].nbDesorbed; // let misses count as 0 (-1+1)
        counterAbsorp[facIndex] += globalCounter.facetHitCounters[i].nbAbsEquiv; // let misses count as 0 (-1+1)
        //if(data.facetHitCounters[i].nbMCHit>0 || data.facetHitCounters[i].nbDesorbed> 0 || data.facetHitCounters[i].nbAbsEquiv>0)
           // facetCounterEveryFile << (i/this->model->nbFacets_total) << " " << (i%this->model->nbFacets_total)+1 << " " << data.facetHitCounters[i].nbMCHit << " " << data.facetHitCounters[i].nbDesorbed << " " << static_cast<unsigned int>(data.facetHitCounters[i].nbAbsEquiv) << std::endl;
    }
    //facetCounterEveryFile.close();

    std::ofstream facetCounterFile;
    facetCounterFile.open (fileName);
    for(unsigned int i = 0; i < nbFacets; i++) {
        //if(counter2[i]>0 || absorb[i]> 0 || desorb[i]>0)
        facetCounterFile << std::setprecision(12) << i+1 << " " << counterMCHit[i] << " " << counterDesorp[i] << " " << static_cast<unsigned int>(counterAbsorp[i]) << std::endl;
    }
    facetCounterFile.close();

    // Texture output
    for(auto& mesh : model->triangle_meshes){
        int lastTexture = -1;
        for(auto& facet : mesh->poly){
            if(((facet.texProps.textureFlags & flowgeom::TEXTURE_FLAGS::countRefl) || (facet.texProps.textureFlags & flowgeom::TEXTURE_FLAGS::countAbs)) && (lastTexture<(int)facet.parentIndex)){
                facetCounterFile.open ("textures"+std::to_string(facet.parentIndex)+".txt");

                unsigned long long int total0 = 0;
                double total1 = 0;
                double total2 = 0;

                unsigned int width = model->facetTex[facet.texProps.textureOffset].texWidth;
                unsigned int height = model->facetTex[facet.texProps.textureOffset].texHeight;

                auto& texels = globalCounter.textures[facet.parentIndex];
                // textures
                for(unsigned int h = 0; h < height; ++h){
                    for(unsigned int w = 0; w < width; ++w){
                        unsigned int index = w+h*width;
                        facetCounterFile << w << " " << h << " " << texels[index].countEquiv << "  "<< texels[index].sum_v_ort_per_area << "  "<< texels[index].sum_1_per_ort_velocity<<std::endl;
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
                for(unsigned int h = 0; h < height; ++h){
                    for(unsigned int w = 0; w < width; ++w){
                        unsigned int index = w+h*width;
                        //if(h == (height/2)){
                        unsigned int bin_index = index/std::max((unsigned int)(width*height/binSize),1u);
                        if(width*height<binSize) {
                            bin_count[bin_index] += texels[index].countEquiv;
                            bin_sumv[bin_index] += texels[index].sum_v_ort_per_area;
                            bin_sum1[bin_index] += texels[index].sum_1_per_ort_velocity;
                        }
                       // }

                        /*unsigned int index = w+h*model->facetTex[facet.texProps.textureOffset].texWidth;
                        facetCounterFile << data.texels[index].countEquiv << "  "<< data.texels[index].sum_v_ort_per_area << "  "<< data.texels[index].sum_1_per_ort_velocity<<std::endl;
                        total0 += data.texels[index].countEquiv;
                        total1 += data.texels[index].sum_v_ort_per_area;
                        total2 += data.texels[index].sum_1_per_ort_velocity;

                        bin_count[index/(width*height/binSize)] += data.texels[index].countEquiv;
                        bin_sumv[index/(width*height/binSize)] += data.texels[index].sum_v_ort_per_area;
                        bin_sum1[index/(width*height/binSize)] += data.texels[index].sum_1_per_ort_velocity;*/
                    }
                }
                facetCounterFile << std::endl;
                facetCounterFile.close();

                facetCounterFile.open ("profiles"+std::to_string(facet.parentIndex)+".txt");
                for(int i=0; i<binSize; ++i){
                    facetCounterFile << std::setprecision(12) << bin_count[i] << "  "<< bin_sumv[i] << "  "<< bin_sum1[i]<<std::endl;
                }
                facetCounterFile.close();
                lastTexture = facet.parentIndex;
            }
        }
    }

}

unsigned long long int SimulationControllerGPU::GetTotalHits(){

    unsigned long long int total_counter = 0;
    for(unsigned int i = 0; i < data.facetHitCounters.size(); i++) {
        total_counter += data.facetHitCounters[i].nbMCHit; // let misses count as 0 (-1+1)
    }

    return total_counter;
}
/**
 *
 * @return 1=could not load GPU Sim, 0=successfully loaded
 */
int SimulationControllerGPU::CloseSimulation() {
    try {
        optixHandle->cleanup();
        delete optixHandle;
    } catch (std::runtime_error& e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}

GlobalCounter *SimulationControllerGPU::GetGlobalCounter()  {
    return &this->globalCounter;
}

/**
 *
 * @return 1=could not load GPU Sim, 0=successfully loaded
 *//*

int SimulationControllerGPU::LoadSimulation(const std::vector<Vector3d> &geomVertices, const std::vector<SuperStructure> &structures, const WorkerParams& wp, const std::vector<std::vector<std::pair<double,double>>> CDFs) {
    try {
        model = flowgpu::loadFromMolflow(geomVertices, structures, wp, CDFs);
        */
/*(model = flowgpu::loadOBJ(
#ifdef _WIN32
                // on windows, visual studio creates _two_ levels of build dir
                // (x86/Release)
                "../../models/sponza.obj"
#else
        // on linux, common practice is to have ONE level of build dir
      // (say, <project>/build/)...
      "../models/sponza.obj"
#endif
        );*//*

    } catch (std::runtime_error& e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
        exit(1);
    }

    */
/*flowgpu::Camera camera = { *//*
*/
/*model->bounds.center()-*//*
*/
/*flowgpu::float3(7.637f, -2.58f, -5.03f),
                                                                                        model->bounds.center()-flowgpu::float3(0.f, 0.f, 0.0f),
                                                                                        flowgpu::float3(0.f,1.f,0.f) };*//*

    flowgpu::Camera camera = {flowgpu::float3(11.0f, -4.6f, -4.0f),
                              model->bounds.center(),
                              flowgpu::float3(0.f, 1.f, 0.f) };
    */
/*flowgpu::Camera camera = { model->bounds.center(),
                               model->bounds.center()-flowgpu::float3(0.1f, 0.1f, 0.1f),
                       flowgpu::float3(0.f,1.f,0.f) };*//*


    // something approximating the scale of the world, so the
    // camera knows how much to move for any given user interaction:
    const float worldScale = length(model->bounds.span());
    const std::string windowTitle = "Optix 7 OBJ Model";
    window = new flowgpu::SampleWindow(windowTitle, model, camera, worldScale);
    return 0;
}

*/
/**
 *
 * @return 1=could not load GPU Sim, 0=successfully loaded
 *//*

int SimulationControllerGPU::RunSimulation() {

    try {
        window->run();
    } catch (std::runtime_error& e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}

*/
/**
 *
 * @return 1=could not load GPU Sim, 0=successfully loaded
 *//*

int SimulationControllerGPU::CloseSimulation() {

    try {
        delete window;
    } catch (std::runtime_error& e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}*/

/**
*
* @return 1=could not load GPU Sim, 0=successfully loaded
*/
/*int SimulationControllerGPU::LoadSimulation(const std::vector<Vector3d> &geomVertices, const std::vector<SuperStructure> &structures, const WorkerParams& wp, const std::vector<std::vector<std::pair<double,double>>> CDFs) {
    try {
        model = flowgpu::loadFromMolflow(geomVertices, structures, wp, CDFs);
    } catch (std::runtime_error& e) {
        std::cout << MF_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << MF_TERMINAL_DEFAULT << std::endl;
        std::cout << "Does GPUMolflow support this geometry yet?" << std::endl;
        exit(1);
    }

    *//*std::ofstream file( "test_geom.xml" );
    cereal::XMLOutputArchive archive( file );
    archive(
            //CEREAL_NVP(model->poly_meshes[0]->poly) ,
            CEREAL_NVP(model->poly_meshes[0]->facetProbabilities) ,
            CEREAL_NVP(model->poly_meshes[0]->cdfs) ,
            CEREAL_NVP(model->poly_meshes[0]->vertices2d) ,
            CEREAL_NVP(model->poly_meshes[0]->vertices3d) ,
            CEREAL_NVP(model->poly_meshes[0]->indices) ,
            CEREAL_NVP(model->poly_meshes[0]->nbFacets) ,
            CEREAL_NVP(model->poly_meshes[0]->nbVertices) ,
            CEREAL_NVP(model->nbFacets_total) ,
            CEREAL_NVP(model->nbVertices_total) ,
            CEREAL_NVP(model->useMaxwell) ,
            CEREAL_NVP(model->bounds.lower) ,
            CEREAL_NVP(model->bounds.upper)
            );*//*


    tracer = new flowgpu::MolTracer(model);
    tracer->setup();
    return 0;
}*/