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

#include "Initializer.h"
#include "IO/LoaderXML.h"
#include "ParameterParser.h"

#include <CLI11/CLI11.hpp>
#include <Helper/StringHelper.h>
#include <Helper/ConsoleLogger.h>
#include <SettingsIO.h>

namespace Settings {
    size_t nbThreads = 0;
    uint64_t simDuration = 10;
    uint64_t outputDuration = 60;
    uint64_t autoSaveDuration = 600; // default: autosave every 600s=10min
    bool loadAutosave = false;
    std::list<uint64_t> desLimit;
    bool resetOnStart = false;
    std::string paramFile;
    std::vector<std::string> paramSweep;
}

void initDefaultSettings() {
    Settings::nbThreads = 0;
    Settings::simDuration = 0;
    Settings::outputDuration = 60;
    Settings::autoSaveDuration = 600;
    Settings::loadAutosave = false;
    Settings::desLimit.clear();
    Settings::resetOnStart = false;
    Settings::paramFile.clear();
    Settings::paramSweep.clear();

    SettingsIO::outputFacetDetails = false;
    SettingsIO::outputFacetQuantities = false;
    SettingsIO::overwrite = false;
    SettingsIO::autogenerateTest = 0.0;

    SettingsIO::workFile.clear();
    SettingsIO::inputFile.clear();
    SettingsIO::outputFile.clear();
    SettingsIO::workPath.clear();
    SettingsIO::inputPath.clear();
    SettingsIO::outputPath.clear();
}

/**
* \brief Modifies the default --help command output
 */
class FlowFormatter : public CLI::Formatter {
public:
    std::string make_usage(const CLI::App *app, std::string name) const override {
        return "Usage: ./"
               + std::filesystem::path(name).filename().string()
               + " [options]";
    }
};

/**
* \brief Parse all CLI commands to use with the corresponding variables (@Settings / @SettingsIO)
 * \return 0> error code, 0 when ok
 */
int Initializer::parseCommands(int argc, char **argv) {
    CLI::App app{"Molflow+/Synrad+ Simulation Management"};
    app.formatter(std::make_shared<FlowFormatter>());

    // Local variables for parsing and immediate processing
    bool verbose = false;
    std::vector<double> limits;

    // Define options
    app.add_option("-j,--threads", Settings::nbThreads, "# Threads to be deployed");
    app.add_option("-t,--time", Settings::simDuration, "Simulation duration in seconds");
    app.add_option("-d,--ndes", limits, "Desorption limit for simulation end");

    auto group = app.add_option_group("subgroup");
    group->add_option("-f,--file", SettingsIO::inputFile, "Required input file (XML/ZIP only)")
            ->check(CLI::ExistingFile);
    group->add_flag("--auto", SettingsIO::autogenerateTest, "Use auto generated test case");
    group->require_option(1);

    CLI::Option *optOfile = app.add_option("-o,--output", SettingsIO::outputFile,
                                           R"(Output file name (e.g. 'outfile.xml', defaults to 'out_{inputFileName}')");
    CLI::Option *optOpath = app.add_option("--outputPath", SettingsIO::outputPath,
                                           "Output path, defaults to \'Results_{date}\'");
    app.add_option("-s,--outputDuration", Settings::outputDuration, "Seconds between each stat output if not zero");
    app.add_option("-a,--autosaveDuration", Settings::autoSaveDuration, "Seconds for autoSave if not zero");
    app.add_flag("--writeFacetDetails", SettingsIO::outputFacetDetails,
                   "Will write a CSV file containing all facet details including physical quantities");
    app.add_flag("--writeFacetQuantities", SettingsIO::outputFacetQuantities,
                   "Will write a CSV file containing all physical quantities for each facet");
    app.add_flag("--analyzeScene", SettingsIO::analyzeScene,
                 "Will analyze the scene, print output and quit");

    app.add_option("--setParamsByFile", Settings::paramFile,
                   "Parameter file for ad hoc change of the given geometry parameters")
            ->check(CLI::ExistingFile);
    app.add_option("--setParams", Settings::paramSweep,
                   "Direct parameter input for ad hoc change of the given geometry parameters");
    app.add_option("--verbosity", Settings::verbosity, "Restrict console output to different levels");

    app.add_flag("--loadAutosave", Settings::loadAutosave, "Whether autosave_ file should be used if exists");
    app.add_flag("-r,--reset", Settings::resetOnStart, "Resets simulation status loaded from file");
    app.add_flag("--verbose", verbose, "Verbose console output (all levels)");
    CLI::Option *optOverwrite = app.add_flag("--overwrite", SettingsIO::overwrite,
                                             "Overwrite input file with new results")->excludes(optOfile, optOpath);
    optOfile->excludes(optOverwrite);
    optOpath->excludes(optOverwrite);
    app.set_config("--config");

    CLI11_PARSE(app, argc, argv);

    if (verbose)
        Settings::verbosity = 42;

    //std::cout<<app.config_to_str(true,true);
    for (auto& lim : limits)
        Settings::desLimit.emplace_back(static_cast<size_t>(lim));

    if (Settings::simDuration == 0 && Settings::desLimit.empty()) {
        Log::console_error("No end criterion has been set!\n");
        Log::console_error(" Either use: -t or -d\n");
        return 0;
    }

    return -1;
}

/**
* \brief Comfort function wrapping around default initialization from command line arguments, taking care of default initializations
 * \return 0> error code, 0 when ok
 */
int Initializer::initFromArgv(int argc, char **argv, SimulationManager *simManager,
                              const std::shared_ptr<MolflowSimulationModel>& model) {

#if defined(WIN32) || defined(__APPLE__)
    setlocale(LC_ALL, "C");
#else
    std::setlocale(LC_ALL, "C");
#endif

    initDefaultSettings();

    int err;
    if (-1 < (err = parseCommands(argc, argv))) {
        Log::console_error("Error: Initialising parsing arguments\n");
        return err;
    }

    Log::console_header(1, "Commence: Initialising!\n");

    simManager->nbThreads = Settings::nbThreads;
    simManager->useCPU = true;

    if (simManager->InitSimUnits()) {
        Log::console_error("Error: Initialising simulation units: {}\n", simManager->nbThreads);
        return 1;
    }

    model->otfParams.nbProcess = simManager->nbThreads;
    model->otfParams.timeLimit = (double) Settings::simDuration;
    //model->otfParams.desorptionLimit = Settings::desLimit.front();
    Log::console_msg_master(4, "Active cores: {}\n", simManager->nbThreads);
    Log::console_msg_master(4, "Running simulation for: {} sec\n", Settings::simDuration);

    return -1;
}

/**
* \brief Initializes the simulation model from a valid input file and handles parameter sweeps
 * \return 0> error code, 0 when ok
 */
int Initializer::initFromFile(SimulationManager *simManager, const std::shared_ptr<MolflowSimulationModel>& model,
                              GlobalSimuState *globState) {
    if (SettingsIO::prepareIO()) {
        Log::console_error("Error preparing I/O folders\n");
        return 1;
    }

    if (std::filesystem::path(SettingsIO::workFile).extension() == ".xml") {
        if (loadFromXML(SettingsIO::workFile, !Settings::resetOnStart, model, globState)) {
            return 1;
        }
    }
    else {
        Log::console_error("Invalid file extension for input file detected: {}\n",
                           std::filesystem::path(SettingsIO::workFile).extension().string());
        return 1;
    }
    if (!Settings::paramFile.empty() || !Settings::paramSweep.empty()) {
        // 1. Load selection groups in case we need them for parsing
        std::vector<SelectionGroup> selGroups = FlowIO::LoaderXML::LoadSelections(SettingsIO::workFile);
        // 2. Sweep parameters from file
        if (!Settings::paramFile.empty())
            ParameterParser::ParseFile(Settings::paramFile, selGroups);
        if (!Settings::paramSweep.empty())
            ParameterParser::ParseInput(Settings::paramSweep, selGroups);
        ParameterParser::ChangeSimuParams(model->wp);
        if(ParameterParser::ChangeFacetParams(model->facets)){
            return 1;
        }
    }

    // Set desorption limit if used
    if (initDesLimit(model, *globState)) {
        return 1;
    }

    simManager->simulationChanged = true;
    Log::console_msg_master(2, "Forwarding model to simulation units!\n");
    try {
        if(simManager->InitSimulation(model, globState))
            throw std::runtime_error("Could not init simulation");
    }
    catch (std::exception &ex) {
        Log::console_error("Failed initialising simulation units:\n{}\n", ex.what());
        return 1;
    }
    Log::console_footer(1, "Finalize: Initialising!\n");

    return 0;
}

/**
* \brief Initialize simulation from automatically generated test case (prism)
 * \return 0> error code, 0 when ok
 */
int Initializer::initAutoGenerated(SimulationManager *simManager, const std::shared_ptr<MolflowSimulationModel> &model,
                                   GlobalSimuState *globState, double ratio, int steps, double angle) {
    if (SettingsIO::prepareIO()) {
        Log::console_error("Error preparing I/O folders\n");
        return 1;
    }

    loadFromGeneration(model, globState, ratio, steps, angle);

    // Set desorption limit if used
    if (initDesLimit(model, *globState)) {
        return 1;
    }

    simManager->simulationChanged = true;
    Log::console_msg_master(2, "Forwarding model to simulation units!\n");
    try {
        if(simManager->InitSimulation(model, globState))
            throw std::runtime_error("Could not init simulation");
    }
    catch (std::exception &ex) {
        Log::console_error("Failed initialising simulation units:\n{}\n", ex.what());
        return 1;
    }
    Log::console_footer(1, "Finalize: Initialising!\n");

    return 0;
}

/**
* \brief Generate a test case with an oblique prism given an angle
 * \return 0> error code, 0 when ok
 */
int
Initializer::loadFromGeneration(const std::shared_ptr<MolflowSimulationModel> &model, GlobalSimuState *globState, double ratio,
                                int step, double angle) {

    Log::console_header(1, "[ ] Loading geometry : PRISM\n");

    //double ratio = 10.0;
    double R = 1.0;
    double L = ratio * R;
    //int    step = 10;
    //1. Load Input File (regular XML)
    // Geometry
    double sticking = 0.0; // none
    model->BuildPrisma(L, R, angle, sticking, step);
    // Settings
    // Previous results

    // InsertParamsBeforeCatalog
    std::vector<Parameter> paramCatalog;
    Parameter::LoadParameterCatalog(paramCatalog);
    model->tdParams.parameters.insert(model->tdParams.parameters.end(), paramCatalog.begin(), paramCatalog.end());

    Log::console_msg_master(3, " Loaded geometry of {} bytes!\n", model->size());

    //InitializeGeometry();
    model->InitialiseFacets();

    Log::console_msg_master(3, " Initializing geometry!\n");
    initSimModel(model);
    if(model->PrepareToRun()){
        return 1;
    }

    // 2. Create simulation dataports
    try {
        //progressDlg->SetMessage("Creating Logger...");
        /*size_t logDpSize = 0;
        if (model->otfParams.enableLogging) {
            logDpSize = sizeof(size_t) + model->otfParams.logLimit * sizeof(ParticleLoggerItem);
        }
        simManager->ReloadLogBuffer(logDpSize, true);*/

        Log::console_msg_master(3, " Resizing state!\n");
        globState->Resize(model);
    }
    catch (const std::exception &e) {
        Log::console_error("[Warning] {}\n", e.what());
    }

    Log::console_footer(1, "[x] Loaded geometry\n");

    return 0;
}

/**
* \brief Wrapper for XML loading with LoaderXML
 * \return 0> error code, 0 when ok
 */
int Initializer::loadFromXML(const std::string &fileName, bool loadState, const std::shared_ptr<MolflowSimulationModel>& model,
                             GlobalSimuState *globState) {

    Log::console_header(1, "[ ] Loading geometry from file {}\n", fileName);

    //1. Load Input File (regular XML)
    FlowIO::LoaderXML loader;
    // Easy if XML is split into multiple parts
    // Geometry
    // Settings
    // Previous results
    double progress = 0.0;
    if (loader.LoadGeometry(fileName, model, &progress)) {
        Log::console_error("Please check the input file!\n");
        return 1;
    }

    // InsertParamsBeforeCatalog
    std::vector<Parameter> paramCatalog;
    Parameter::LoadParameterCatalog(paramCatalog);
    model->tdParams.parameters.insert(model->tdParams.parameters.end(), paramCatalog.begin(), paramCatalog.end());

    //TODO: Load parameters from catalog explicitly?
    // For GUI
    // work->InsertParametersBeforeCatalog(loadedParams);
    // Load viewsettings for each facet

    Log::console_msg_master(3, " Loaded geometry of {} bytes!\n", model->size());

    //InitializeGeometry();
    model->InitialiseFacets();

    Log::console_msg_master(3, " Initializing geometry!\n");
    initSimModel(model);
    if(model->PrepareToRun()){
        return 1;
    }
    model->AnalyzeGeom();

    // 2. Create simulation dataports
    try {

        Log::console_msg_master(3, " Resizing state!\n");
        globState->Resize(model);

        // 3. init counters with previous results
        if (loadState) {
            Log::console_msg_master(3, " Initializing previous simulation state!\n");

            if (Settings::loadAutosave) {
                std::string autosaveFileName = std::filesystem::path(SettingsIO::workFile).filename().string();
                std::string autoSavePrefix = "autosave_";
                autosaveFileName = autoSavePrefix + autosaveFileName;
                if (std::filesystem::exists(autosaveFileName)) {
                    Log::console_msg_master(2, " Found autosave file! Loading simulation state...\n");
                    FlowIO::LoaderXML::LoadSimulationState(autosaveFileName, model, globState, nullptr);
                }
            } else {
                FlowIO::LoaderXML::LoadSimulationState(SettingsIO::workFile, model, globState, nullptr);
            }

            // Update Angle map status
            for(int i = 0; i < model->facets.size(); i++ ) {
#if defined(MOLFLOW)
                auto f = std::dynamic_pointer_cast<MolflowSimFacet>(model->facets[i]);
                if (f->sh.anglemapParams.record) { //Recording, so needs to be updated
                    //Retrieve angle map from hits dp
                    globState->facetStates[i].recordedAngleMapPdf = f->angleMap.pdf;
                }
#endif
            }
        }
    }
    catch (const std::exception &e) {
        Log::console_error("[Warning] {}\n", e.what());
    }

    Log::console_footer(1, "[x] Loaded geometry\n");

    return 0;
}

/**
* \brief Initialize a desorption limit by checking against the given input (vector)
 * \return 0> error code, 0 when ok
 */
int Initializer::initDesLimit(const std::shared_ptr<MolflowSimulationModel>& model, GlobalSimuState &globState) {
    if (!model->m.try_lock()) {
        return 1;
    }

    model->otfParams.desorptionLimit = 0;

    // Skip desorptions if limit was already reached
    if (!Settings::desLimit.empty()) {
        size_t oldDesNb = globState.globalHits.globalHits.nbDesorbed;
        size_t listSize = Settings::desLimit.size();
        for (size_t l = 0; l < listSize; ++l) {
            model->otfParams.desorptionLimit = Settings::desLimit.front();
            Settings::desLimit.pop_front();

            if (oldDesNb > model->otfParams.desorptionLimit) {
                Log::console_msg_master(1, "Skipping desorption limit: {}\n", model->otfParams.desorptionLimit);
            } else {
                Log::console_msg_master(1, "Starting with desorption limit: {} from {}\n",
                                        model->otfParams.desorptionLimit, oldDesNb);

                model->m.unlock();
                return 0;
            }
        }
        if (Settings::desLimit.empty()) {
            Log::console_msg_master(1,
                                    "All given desorption limits have been reached. Consider resetting the simulation results from the input file (--reset): Starting desorption {}\n",
                                    oldDesNb);
            model->m.unlock();
            return 1;
        }
    }
    model->m.unlock();
    return 0;
}

/**
* \brief Initializes time limit for the simulation
 * \return 0> error code, 0 when ok
 */
[[maybe_unused]] int Initializer::initTimeLimit(const std::shared_ptr<MolflowSimulationModel>& model, double time) {
    if (!model->m.try_lock()) {
        return 1;
    }

    model->otfParams.timeLimit = time;
    Settings::simDuration = time;

    model->m.unlock();
    return 0;
}

// TODO: Combine with loadXML function
/**
* \brief Prepares autosave file that is created in user specified intervals
 * \return output file name
 */
std::string Initializer::getAutosaveFile() {
    // Create copy of input file for autosave
    std::string autoSave;
    if (Settings::autoSaveDuration > 0) {
        autoSave = std::filesystem::path(SettingsIO::workFile).filename().string();

        std::string autoSavePrefix = "autosave_";
        // Check if autosave_ is part of the input filename, if yes, generate a new input file without the prefix
        if (autoSave.size() > autoSavePrefix.size() &&
            std::search(autoSave.begin(), autoSave.begin() + autoSavePrefix.size(), autoSavePrefix.begin(),
                        autoSavePrefix.end()) == autoSave.begin()) {
            // TODO: Revisit wether input/output is acceptable here
            autoSave = std::filesystem::path(SettingsIO::workPath).append(SettingsIO::workFile).filename().string();
            SettingsIO::inputFile = autoSave.substr(autoSavePrefix.size(), autoSave.size() - autoSavePrefix.size());
            Log::console_msg_master(2, "Using autosave file {} for {}\n", autoSave, SettingsIO::inputFile);
        } else {
            // create autosavefile from copy of original
            autoSave = std::filesystem::path(SettingsIO::workPath).append(autoSavePrefix).concat(autoSave).string();
            try {
                if(!SettingsIO::workFile.empty() && std::filesystem::exists(SettingsIO::workFile))
                    std::filesystem::copy_file(SettingsIO::workFile, autoSave,
                                           std::filesystem::copy_options::overwrite_existing);
            } catch (std::filesystem::filesystem_error &e) {
                Log::console_error("Could not copy file to create autosave file: {}\n", e.what());
            }
        }
    }

    return autoSave;
}

/**
* \brief Prepares data structures for use in simulation
* \return error code: 0=no error, 1=error
*/
int Initializer::initSimModel(std::shared_ptr<MolflowSimulationModel> model) {

    if (!model->m.try_lock()) {
        return 1;
    }

    std::vector<Moment> momentIntervals;
    momentIntervals.reserve(model->tdParams.moments.size());
    for (auto &moment : model->tdParams.moments) {
        momentIntervals.emplace_back(
                std::make_pair(moment.first - (0.5 * moment.second), moment.first + (0.5 * moment.second)));
    }

    model->tdParams.moments = momentIntervals;


    model->structures.resize(model->sh.nbSuper); //Create structures

    bool hasVolatile = false;

    for (size_t facIdx = 0; facIdx < model->sh.nbFacet; facIdx++) {
        auto* sFac = (MolflowSimFacet*) model->facets[facIdx].get();

        std::vector<double> textIncVector;
        // Add surface elements area (reciprocal)
        if (sFac->sh.isTextured) {
            auto meshAreas = sFac->InitTextureMesh();
            textIncVector.resize(sFac->sh.texHeight * sFac->sh.texWidth);

            double rw = sFac->sh.U.Length() / (double) (sFac->sh.texWidth_precise);
            double rh = sFac->sh.V.Length() / (double) (sFac->sh.texHeight_precise);
            double area = rw * rh;
            area *= (sFac->sh.is2sided) ? 2.0 : 1.0;
            size_t add = 0;

            for (size_t j = 0; j < sFac->sh.texHeight; j++) {
                for (size_t i = 0; i < sFac->sh.texWidth; i++) {
                    if (meshAreas[add] < 0.0) {
                        textIncVector[add] = 1.0 / area;
                    } else {
                        textIncVector[add] = 1.0 / (meshAreas[add] * ((sFac->sh.is2sided) ? 2.0 : 1.0));
                    }
                    add++;
                }
            }
        }
        sFac->textureCellIncrements = textIncVector;

        //Some initialization
        try {
            if (!sFac->InitializeOnLoad(facIdx, model->tdParams.moments.size())) return false;
        }
        catch (const std::exception& err){
            Log::console_error("Failed to initialize facet (F#{})\n{}\n", facIdx + 1, err.what());
            model->m.unlock();
            return 1;
        }
        hasVolatile |= sFac->sh.isVolatile;

        if ((sFac->sh.superDest || sFac->sh.isVolatile) &&
            ((sFac->sh.superDest - 1) >= model->sh.nbSuper || sFac->sh.superDest < 0)) {
            // Geometry error
            Log::console_error("Invalid structure (wrong link on F#{})\n", facIdx + 1);
            model->m.unlock();
            return 1;
        }
    }
    model->m.unlock();

    return 0;
}