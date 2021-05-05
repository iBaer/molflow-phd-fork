//
// Created by Pascal Baehr on 01.08.20.
//

#include "Initializer.h"
#include "IO/LoaderXML.h"
#include "ParameterParser.h"

#include <CLI11/CLI11.hpp>
#include <ziplib/ZipArchive.h>
#include <ziplib/ZipFile.h>
#include <File.h>
#include <Helper/StringHelper.h>
#include <Helper/ConsoleLogger.h>

namespace Settings {
    size_t nbThreads = 0;
    uint64_t simDuration = 10;
    uint64_t autoSaveDuration = 600; // default: autosave every 600s=10min
    bool loadAutosave = false;
    std::list<uint64_t> desLimit;
    bool resetOnStart = false;
    std::string inputFile;
    std::string outputFile;
    std::string paramFile;
    std::vector<std::string> paramSweep;
    std::string outputPath;
}

void initDefaultSettings(){
    Settings::nbThreads = 0;
    Settings::simDuration = 10;
    Settings::autoSaveDuration = 600;
    Settings::loadAutosave = false;
    Settings::desLimit.clear();
    Settings::resetOnStart = false;
    Settings::inputFile.clear();
    Settings::outputFile.clear();
    Settings::paramFile.clear();
    Settings::paramSweep.clear();
    Settings::outputPath.clear();
}

class FlowFormatter : public CLI::Formatter {
public:
    std::string make_usage(const CLI::App *app, std::string name) const override {
        return "Usage: ./"
               +std::filesystem::path(name).filename().string()
               +" [options]";
    }
};
int initDirectories(){

    int err = 0;

    // Use a default outputpath if unset
    if(Settings::outputPath.empty()) {
        Settings::outputPath = "Results_" + Util::getTimepointString();
    }
    else if(std::filesystem::path(Settings::outputFile).has_parent_path()) {
        Log::console_error("Output path was set to %s, but Output file also contains a parent path %s\n"
                           "Output path will be appended!\n", Settings::outputPath.c_str() , std::filesystem::path(Settings::outputFile).parent_path().c_str());
    }

    // Use a default outputfile name if unset
    if(Settings::outputFile.empty())
        Settings::outputFile = "out_" + std::filesystem::path(Settings::inputFile).filename().string();

    if(std::filesystem::path(Settings::outputFile).extension() != ".xml"){
        Settings::outputFile = std::filesystem::path(Settings::outputFile).replace_extension(".xml").string();
    }
    // Try to create directories
    // First for outputpath, with tmp/ and lastly ./ as fallback plans
    try {
        std::filesystem::create_directory(Settings::outputPath);
    }
    catch (std::exception& e){
        Log::console_error("Couldn't create directory [ %s ], falling back to binary folder for output files\n", Settings::outputPath.c_str());
        ++err;

        // use fallback dir
        Settings::outputPath = "tmp/";
        try {
            std::filesystem::create_directory(Settings::outputPath);
        }
        catch (std::exception& e){
            Settings::outputPath = "./";
            Log::console_error("Couldn't create fallback directory [ tmp/ ], falling back to binary folder instead for output files\n");
            ++err;
        }
    }

    // Next check if outputfile name has parent path as name
    // Additional directory in outputpath
    if(std::filesystem::path(Settings::outputFile).has_parent_path()) {
        std::string outputFilePath = Settings::outputPath + '/' + std::filesystem::path(Settings::outputFile).parent_path().string();
        try {
            std::filesystem::create_directory(outputFilePath);
        }
        catch (std::exception& e) {
            Log::console_error("Couldn't create parent directory set by output filename [ %s ], will only use default output path instead\n", outputFilePath.c_str());

            ++err;
        }
    }

    return err;
}


int Initializer::initFromArgv(int argc, char **argv, SimulationManager *simManager, SimulationModel *model) {
    Log::console_msg_master(1,"Commence: Initialising!\n");

#if defined(WIN32) || defined(__APPLE__)
    setlocale(LC_ALL, "C");
#else
    std::setlocale(LC_ALL, "C");
#endif

    initDefaultSettings();
    parseCommands(argc, argv);

    simManager->nbThreads = Settings::nbThreads;
    simManager->useCPU = true;

    if(simManager->InitSimUnits()) {
        Log::console_error("Error: Initialising simulation units: %zu\n", simManager->nbThreads);
        return 1;
    }
    Log::console_msg_master(2, "Active cores: %zu\n", simManager->nbThreads);

    model->otfParams.nbProcess = simManager->nbThreads;
    model->otfParams.timeLimit = (double) Settings::simDuration;
    //model->otfParams.desorptionLimit = Settings::desLimit.front();
    Log::console_msg_master(2, "Running simulation for: '%zu'sec\n", Settings::simDuration);

    return 0;
}

int Initializer::initFromFile(int argc, char **argv, SimulationManager *simManager, SimulationModel *model,
                              GlobalSimuState *globState) {
    initDirectories();
    if(std::filesystem::path(Settings::inputFile).extension() == ".zip"){
        //decompress file
        std::string parseFileName;
        Log::console_msg_master(2, "Decompressing zip file...\n");

        ZipArchive::Ptr zip = ZipFile::Open(Settings::inputFile);
        if (zip == nullptr) {
            Log::console_error("Can't open ZIP file\n");
        }
        size_t numItems = zip->GetEntriesCount();
        bool notFoundYet = true;
        for (int i = 0; i < numItems && notFoundYet; i++) { //extract first xml file found in ZIP archive
            auto zipItem = zip->GetEntry(i);
            std::string zipFileName = zipItem->GetName();

            if(std::filesystem::path(zipFileName).extension() == ".xml"){ //if it's an .xml file
                notFoundYet = false;

                if(Settings::outputPath != "tmp/")
                    FileUtils::CreateDir("tmp");// If doesn't exist yet

                parseFileName = "tmp/" + zipFileName;
                ZipFile::ExtractFile(Settings::inputFile, zipFileName, parseFileName);
            }
        }
        if(parseFileName.empty()) {
            Log::console_error("Zip file does not contain a valid geometry file!\n");
            exit(0);
        }
        Settings::inputFile = parseFileName;
        Log::console_msg_master(2, "New input file: %s\n", Settings::inputFile.c_str());
    }

    if(std::filesystem::path(Settings::inputFile).extension() == ".xml")
        loadFromXML(Settings::inputFile, !Settings::resetOnStart, model, globState);
    else{
        Log::console_error("[ERROR] Invalid file extension for input file detected: %s\n", std::filesystem::path(Settings::inputFile).extension().c_str());
        return 1;
    }
    if(!Settings::paramFile.empty() || !Settings::paramSweep.empty()){
        // 1. Load selection groups in case we need them for parsing
        std::vector<SelectionGroup> selGroups = FlowIO::LoaderXML::LoadSelections(Settings::inputFile);
        // 2. Sweep parameters from file
        if(!Settings::paramFile.empty())
            ParameterParser::ParseFile(Settings::paramFile, selGroups);
        if(!Settings::paramSweep.empty())
            ParameterParser::ParseInput(Settings::paramSweep, selGroups);
        ParameterParser::ChangeSimuParams(model->wp);
        ParameterParser::ChangeFacetParams(model->facets);
    }

    // Set desorption limit if used
    if(initDesLimit(*model,*globState)) {
        exit(0);
        return 1;
    }
    simManager->InitSimulation(model, globState);

    Log::console_msg_master(1,"Finalize: Initialising!\n");

    return 0;
}

int Initializer::parseCommands(int argc, char** argv) {
    CLI::App app{"Molflow+/Synrad+ Simulation Management"};
    app.formatter(std::make_shared<FlowFormatter>());

    std::vector<double> limits;
    // Define options
    app.add_option("-j,--threads", Settings::nbThreads, "# Threads to be deployed");
    app.add_option("-t,--time", Settings::simDuration, "Simulation duration in seconds");
    app.add_option("-d,--ndes", limits, "Desorption limit for simulation end");
    app.add_option("-f,--file", Settings::inputFile, "Required input file (XML only)")
            ->required()
            ->check(CLI::ExistingFile);
    app.add_option("-o,--output", Settings::outputFile, R"(Output file name (e.g. 'outfile.xml', defaults to 'out_{inputFileName}')");
    app.add_option("--outputPath", Settings::outputPath, "Output path, defaults to \'Results_{date}\'");
    app.add_option("-a,--autosaveDuration", Settings::autoSaveDuration, "Seconds for autoSave if not zero");
    app.add_flag("--loadAutosave", Settings::loadAutosave, "Whether autoSave_ file should be used if exists");
    app.add_option("--setParamsByFile", Settings::paramFile, "Parameter file for ad hoc change of the given geometry parameters")
            ->check(CLI::ExistingFile);
    app.add_option("--setParams", Settings::paramSweep, "Direct parameter input for ad hoc change of the given geometry parameters");

    app.add_flag("-r,--reset", Settings::resetOnStart, "Resets simulation status loaded from while");
    app.set_config("--config");
    CLI11_PARSE(app, argc, argv);

    //std::cout<<app.config_to_str(true,true);
    for(auto& lim : limits)
        Settings::desLimit.emplace_back(lim);
    return 0;
}

int Initializer::loadFromXML(const std::string &fileName, bool loadState, SimulationModel *model,
                             GlobalSimuState *globState) {

    //1. Load Input File (regular XML)
    FlowIO::LoaderXML loader;
    // Easy if XML is split into multiple parts
    // Geometry
    // Settings
    // Previous results
    model->m.lock();
    if(loader.LoadGeometry(fileName, model)){
        Log::console_error("[Error (LoadGeom)] Please check the input file!\n");
        model->m.unlock();
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

    Log::console_msg_master(1,"[LoadGeom] Loaded geometry of %zu bytes!\n", model->size());

    //InitializeGeometry();
    model->InitialiseFacets();
    model->PrepareToRun();

    Log::console_msg_master(1,"[LoadGeom] Initializing geometry!\n");
    initSimModel(model);

    // 2. Create simulation dataports
    try {
        //progressDlg->SetMessage("Creating Logger...");
        /*size_t logDpSize = 0;
        if (model->otfParams.enableLogging) {
            logDpSize = sizeof(size_t) + model->otfParams.logLimit * sizeof(ParticleLoggerItem);
        }
        simManager->ReloadLogBuffer(logDpSize, true);*/

        Log::console_msg_master(1,"[LoadGeom] Resizing state!\n");
        globState->Resize(*model);

        // 3. init counters with previous results
        if(loadState) {
            Log::console_msg_master(1,"[LoadGeom] Initializing previous simulation state!\n");

            if(Settings::loadAutosave){
                std::string fileName = std::filesystem::path(Settings::inputFile).filename().string();
                std::string autoSavePrefix = "autosave_";
                fileName = autoSavePrefix + fileName;
                if(std::filesystem::exists(fileName)) {
                    Log::console_msg_master(1,"Found autosave file! Loading simulation state...\n");
                    FlowIO::LoaderXML::LoadSimulationState(fileName, model, *globState);
                }
            }
            else {
                FlowIO::LoaderXML::LoadSimulationState(Settings::inputFile, model, *globState);
            }
        }
    }
    catch (std::exception& e) {
        Log::console_error("[Warning (LoadGeom)] %s\n", e.what());
    }

    model->m.unlock();

    return 0;
}

int Initializer::initSimUnit(SimulationManager *simManager, SimulationModel *model, GlobalSimuState *globState) {

    model->m.lock();

    // Prepare simulation unit
    Log::console_msg_master(2, "[LoadGeom] Forwarding model to simulation units!\n");

    simManager->ResetSimulations();
    simManager->ForwardSimModel(model);
    simManager->ForwardGlobalCounter(globState, nullptr);

    if(simManager->LoadSimulation()){
        model->m.unlock();
        std::string errString = "Failed to send geometry to sub process:\n";
        errString.append(simManager->GetErrorDetails());
        throw std::runtime_error(errString);
    }

    model->m.unlock();

    return 0;
}

int Initializer::initDesLimit(SimulationModel& model, GlobalSimuState& globState){
    model.otfParams.desorptionLimit = 0;

    // Skip desorptions if limit was already reached
    if(!Settings::desLimit.empty())
    {
        size_t oldDesNb = globState.globalHits.globalHits.nbDesorbed;
        size_t listSize = Settings::desLimit.size();
        for(size_t l = 0; l < listSize; ++l) {
            model.otfParams.desorptionLimit = Settings::desLimit.front();
            Settings::desLimit.pop_front();

            if (oldDesNb > model.otfParams.desorptionLimit){
                Log::console_msg_master(1,"Skipping desorption limit: %zu\n", model.otfParams.desorptionLimit);
            }
            else{
                Log::console_msg_master(1,"Starting with desorption limit: %zu from %zu\n", model.otfParams.desorptionLimit , oldDesNb);
                return 0;
            }
        }
        if(Settings::desLimit.empty()){
            Log::console_msg_master(1,"All given desorption limits have been reached. Consider resetting the simulation results from the input file (--reset): Starting desorption %zu\n", oldDesNb);
            return 1;
        }
    }

    return 0;
}

// TODO: Combine with loadXML function
std::string Initializer::getAutosaveFile(){
    // Create copy of input file for autosave
    std::string autoSave;
    if(Settings::autoSaveDuration > 0)
    {
        autoSave = std::filesystem::path(Settings::inputFile).filename().string();

        std::string autoSavePrefix = "autosave_";
        if(autoSave.size() > autoSavePrefix.size() && std::search(autoSave.begin(), autoSave.begin()+autoSavePrefix.size(), autoSavePrefix.begin(), autoSavePrefix.end()) == autoSave.begin())
        {
            autoSave = std::filesystem::path(Settings::inputFile).filename().string();
            Settings::inputFile = autoSave.substr(autoSavePrefix.size(), autoSave.size() - autoSavePrefix.size());
            Log::console_msg_master(2, "Using autosave file %s for %s\n", autoSave.c_str(), Settings::inputFile.c_str());
        }
        else {
            // create autosavefile from copy of original
            std::stringstream autosaveFile;
            autosaveFile << Settings::outputPath << "/" << autoSavePrefix << autoSave;
            autoSave = autosaveFile.str();

            try {
                std::filesystem::copy_file(Settings::inputFile, autoSave,
                                           std::filesystem::copy_options::overwrite_existing);
            } catch (std::filesystem::filesystem_error &e) {
                Log::console_error("Could not copy file: %s\n", e.what());
            }
        }
    }

    return autoSave;
}

/**
* \brief Prepares data structures for use in simulation
* \return error code: 0=no error, 1=error
*/
int Initializer::initSimModel(SimulationModel* model) {

    std::vector<Moment> momentIntervals;
    momentIntervals.reserve(model->tdParams.moments.size());
    for(auto& moment : model->tdParams.moments){
        momentIntervals.emplace_back(std::make_pair(moment.first - (0.5 * moment.second), moment.first + (0.5 * moment.second)));
    }

    model->tdParams.moments = momentIntervals;


    model->structures.resize(model->sh.nbSuper); //Create structures

    bool hasVolatile = false;

    auto& loadFacets = model->facets;
    for (size_t facIdx = 0; facIdx < model->sh.nbFacet; facIdx++) {
        SubprocessFacet& sFac = loadFacets[facIdx];

        std::vector<double> textIncVector;
        // Add surface elements area (reciprocal)
        if (sFac.sh.isTextured) {
            textIncVector.resize(sFac.sh.texHeight*sFac.sh.texWidth);

            double rw = sFac.sh.U.Norme() / (double)(sFac.sh.texWidthD);
            double rh = sFac.sh.V.Norme() / (double)(sFac.sh.texHeightD);
            double area = rw * rh;
            area *= (sFac.sh.is2sided) ? 2.0 : 1.0;
            size_t add = 0;
            for (size_t j = 0; j < sFac.sh.texHeight; j++) {
                for (size_t i = 0; i < sFac.sh.texWidth; i++) {
                    if (area > 0.0) {
                        textIncVector[add] = 1.0 / area;
                    }
                    else {
                        textIncVector[add] = 0.0;
                    }
                    add++;
                }
            }
        }
        sFac.textureCellIncrements = textIncVector;

        //Some initialization
        if (!sFac.InitializeOnLoad(facIdx, model->tdParams.moments.size())) return false;

        hasVolatile |= sFac.sh.isVolatile;

        if ((sFac.sh.superDest || sFac.sh.isVolatile) && ((sFac.sh.superDest - 1) >= model->sh.nbSuper || sFac.sh.superDest < 0)) {
            // Geometry error
            //ClearSimulation();
            //ReleaseDataport(loader);
            std::ostringstream err;
            err << "Invalid structure (wrong link on F#" << facIdx + 1 << ")";
            //SetErrorSub(err.str().c_str());
            Log::console_error("Invalid structure (wrong link on F#%d)\n", facIdx + 1 );

            return 1;
        }
    }

    return 0;
}