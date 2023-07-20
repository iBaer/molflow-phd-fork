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

#include "ParameterParser.h"
#include "Helper/StringHelper.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <Helper/ConsoleLogger.h>

namespace Parameters {

    //! Enum that describes the allowed facet parameters to change
    enum FacetParam : size_t {
        opacity,
        temperature,
        sticking,
        outgassing
    };

    //! Enum that describes the allowed global simulation parameters to change
    enum SimuParam : size_t {
        mass,
        enableDecay,
        halfLife
    };

    //! Table that maps facet parameters against strings
    static std::unordered_map<std::string,FacetParam> const tableFac = {
            {"opacity",FacetParam::opacity},
            {"temperature",FacetParam::temperature},
            {"sticking",FacetParam::sticking},
            {"outgassing",FacetParam::outgassing}
    };
    //! Table that maps simulation parameters against strings
    static std::unordered_map<std::string,SimuParam> const tableSim = {
            {"mass",SimuParam::mass},
            {"enableDecay",SimuParam::enableDecay},
            {"halfLife",SimuParam::halfLife}
    };
    std::vector<std::tuple<size_t, FacetParam, double>> facetParams;
    std::vector<std::tuple<SimuParam, double>> simuParams;
}

//! Parse CLI argument for facet parameter changes, then add entry as a tuple for later usage
// considers Selection group names if part of the input file
void parseFacet(std::istringstream &facetString, const std::vector<SelectionGroup> &selections) {
    std::string id_str;
    std::string param_str;
    std::string paramVal_str;
    std::getline(facetString, id_str, '.');
    std::getline(facetString, param_str, '=');
    std::getline(facetString, paramVal_str);
    std::vector<size_t> id_range;

    bool fromSelection = false; // for corresponding output message
    if(id_str.find('\"') != std::string::npos){ // selection
        id_str.erase (std::remove(id_str.begin(), id_str.end(), '\"'), id_str.end());
        for(const auto& sel : selections){
            if(sel.name == id_str) {
                id_range = sel.selection;
                fromSelection = true;
                break;
            }
        }
    }
    else { // facet id or range
        // facet list returns indices [0,inf], input is given by [1,inf]

        try {
            // For now get facet list for all combinations (with 3 parameter), check for valid ids later
            splitFacetList(id_range, id_str, 1e7);
        } catch (const std::exception &e) {
            Log::console_error("[{}] Could not parse facet id or range:\n", __FUNCTION__);
            Log::console_error("\t{}\n", id_str);
        }
    }
    auto tablePair = Parameters::tableFac.find(param_str);
    if(tablePair == Parameters::tableFac.end()) {
        Log::console_error("[{}] Invalid option was given:\n", __FUNCTION__);
        Log::console_error("\t{}\n", param_str);
        return;
    }
    auto param = tablePair->second;
    double paramVal = std::strtod(paramVal_str.c_str(),nullptr);
    for(auto& id : id_range)
        Parameters::facetParams.emplace_back(std::make_tuple(id, param, paramVal));
    if(fromSelection)
        Log::console_msg_master(3, "[ParameterChange][Facet][Group: {}] Changing parameter {} to {}\n", id_str, param_str, paramVal_str);
    else
        Log::console_msg_master(3, "[ParameterChange][Facet][ID: {}] Changing parameter {} to {}\n", id_str, param_str, paramVal_str);
}

//! Parse CLI argument for simulation parameter changes, then add entry as a tuple for later usage
void parseSimu(std::istringstream& facetString){
    std::string param_str;
    std::string paramVal_str;
    std::getline(facetString, param_str, '=');
    std::getline(facetString, paramVal_str);
    auto tablePair = Parameters::tableSim.find(param_str);
    if(tablePair == Parameters::tableSim.end()) {
        Log::console_error("[{}] Invalid option was given:\n", __FUNCTION__);
        Log::console_error("\t{}\n", param_str);
        return;
    }
    auto param = tablePair->second;
    double paramVal = std::strtod(paramVal_str.c_str(),nullptr);
    Parameters::simuParams.emplace_back(std::make_tuple(param, paramVal));

    Log::console_msg_master(3, "[ParameterChange][Simulation] Changing parameter {} to {}\n", param_str, paramVal_str);

}

//! First parse CLI argument as input stream and call individual routines for facet or simulation changes
void parseInputStream(std::stringstream& inputLineStream, const std::vector<SelectionGroup> &selections){
    Parameters::facetParams.clear();
    Parameters::simuParams.clear();

    size_t i = 0;
    for (std::string line; inputLineStream >> line; ) {
        std::istringstream lineStream(line);
        std::string optionType;
        std::getline(lineStream, optionType, '.');

        if (optionType == "facet") {
            parseFacet(lineStream, selections);
            //printf("[%zu] Parsing %s\n", i, lineStream.str().c_str());
        } else if (optionType == "simulation") {
            parseSimu(lineStream);
        } else {
            Log::console_error("[Line #{}] Unknown input {}\n", i, line);
        }
        ++i;
    }
}

//! Parse parameter sweeps from file by reading into Input Stream
void ParameterParser::ParseFile(const std::string &paramFile, const std::vector<SelectionGroup> &selections) {
    //convDistr=std::vector<std::pair<double,double>>();

    std::ifstream inputFileStream(paramFile);
    std::stringstream inputStream;

    copy(std::istreambuf_iterator<char>(inputFileStream),
         std::istreambuf_iterator<char>(),
         std::ostreambuf_iterator<char>(inputStream));

    parseInputStream(inputStream, selections);

}

//! Parse parameter sweeps from file
void ParameterParser::ParseInput(const std::vector<std::string> &paramSweep, const std::vector<SelectionGroup> &selections) {
    std::stringstream inputStream(std::ios_base::app | std::ios_base::out | std::ios_base::in);
    const char delimiter = ';';
    for(auto sweep : paramSweep){
        size_t pos = 0;
        std::string token;
        while ((pos = sweep.find(delimiter)) != std::string::npos) {
            token = sweep.substr(0, pos);
            inputStream << token << '\n';
            sweep.erase(0, pos + 1);
        }
        inputStream << sweep << '\n';
    }

    parseInputStream(inputStream, selections);
}

//! Read values for parsed simulation parameters
void ParameterParser::ChangeSimuParams(WorkerParams& params){
    for(auto& par : Parameters::simuParams){
        auto type = std::get<0>(par);
        switch (type) {
            case (Parameters::SimuParam::enableDecay):
                params.enableDecay = (std::get<1>(par) > 0.0);
                break;
            case (Parameters::SimuParam::mass):
                params.gasMass = std::get<1>(par);
                break;
            case (Parameters::SimuParam::halfLife):
                params.halfLife = std::get<1>(par);
                break;
            default:
                Log::console_error("Unknown SimuParam {}\n", (size_t)std::get<0>(par));
        }
    }
}

//! Read values for parsed facet parameters
int ParameterParser::ChangeFacetParams(std::vector<std::shared_ptr<SimulationFacet>> &facets) {
    int nbError = 0;
    for(auto& par : Parameters::facetParams){
        size_t id = std::get<0>(par);
        if(id < facets.size()) {
            auto& facet = *facets.at(id);
            auto type = std::get<1>(par);
            switch (type) {
                case (Parameters::FacetParam::opacity):
                    facet.sh.opacity = std::get<2>(par);
                    if(facet.sh.opacity < 0.0 || facet.sh.opacity > 1.0) {
                        nbError++;
                        Log::console_error("[ParameterChange][Facet][ID: {}] Invalid opacity on facet: {}\n", id,
                                           facet.sh.opacity);
                    }
                    break;
                case (Parameters::FacetParam::outgassing):
                    facet.sh.outgassing = std::get<2>(par);
                    break;
                case (Parameters::FacetParam::sticking):
                    facet.sh.sticking = std::get<2>(par);
                    if(facet.sh.sticking < 0.0 || facet.sh.sticking > 1.0) {
                        nbError++;
                        Log::console_error(
                                "[ParameterChange][Facet][ID: {}] Invalid sticking coefficient on facet: {}\n", id,
                                facet.sh.sticking);
                    }
                    break;
                case (Parameters::FacetParam::temperature):
                    facet.sh.temperature = std::get<2>(par);
                    break;
                default:
                    Log::console_error("Unknown FacetParam {}\n", (size_t)std::get<1>(par));
            }
        }
        else{
            Log::console_error("[ParameterChange][Facet][ID: {}] Facet ID out of range\n", id);
            nbError++;
        }
    }
    return nbError;
}