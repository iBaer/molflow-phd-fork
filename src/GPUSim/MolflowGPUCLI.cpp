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

// M_PI define
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define _USE_MATH_DEFINES // activate defines, e.g. M_PI_2
#endif
#include <cmath>

#include "SimulationControllerGPU.h"
// debug output
#include <fstream>

#include <list>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include "ModelReader.h"
#include "SimulationManager.h"
#include "InitializerGPU.h"
#include "IO/WriterXML.h"
#include "fmt/core.h"
#include "Helper/MathTools.h"
#include "File.h"
#include <SettingsIO.h>
#include <ZipLib/ZipFile.h>

#include "SimulationManager.h"
#include "SMP.h"
#include "Buffer_shared.h"
#include <fstream>
#include <filesystem>
#include <Parameter.h>
#include <IO/LoaderXML.h>
#include <IO/WriterXML.h>
#include "InitializerGPU.h"
#include "Helper/MathTools.h"
#include <sstream>
#include <Helper/Chronometer.h>
#include <Helper/StringHelper.h>
#include <Helper/ConsoleLogger.h>
#include <ZipLib/ZipFile.h>

//#if defined(MOLFLOW)
#include <SettingsIO.h>
#include <IO/CSVExporter.h>
//#endif

#include "FlowMPI.h"
#include "File.h"

class RuntimeStatPrinter {
    size_t oldHitsNb{0};
    size_t oldDesNb{0};
public:
    RuntimeStatPrinter() = default;
    RuntimeStatPrinter(size_t n_hits, size_t n_des) {
        oldHitsNb = n_hits;
        oldDesNb = n_des;
    };
    void PrintHeader() const{
        // Print Header at the beginning
        Log::console_msg_master(1, "\n");
        if(MFMPI::world_size > 1) {
            Log::console_msg_master(1, "{:<6} ",
                                    "Node#");
        }
        Log::console_msg_master(1, "{:<14} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}\n",
                                "Time",
                                "#Hits (run)", "#Hits (total)","Hit/sec",
                                "#Des (run)", "#Des (total)","Des/sec");
        if(MFMPI::world_size > 1) {
            Log::console_msg_master(1, "{}",std::string(6,'-').c_str());
        }
        Log::console_msg_master(1, "{}\n",std::string(14+20+20+20+20+20+20,'-').c_str());
    }
    void Print(double elapsedTime, GlobalSimuState& globState, bool printSum=false) const{
        if(printSum) {
            Log::console_msg_master(1, "{}\n",std::string(6+14+20+20+20+20+20+20,'=').c_str());
            Log::console_msg_master(1, "{:<6} ", "x");
        }
        else if(MFMPI::world_size > 1) {
            Log::console_msg(1, "{:<6} ", MFMPI::world_rank);
        }

        Log::console_msg(1,"{:<14.2f} {:<20} {:<20} {:<20.2f} {:<20} {:<20} {:<20.2f}\n",
                         elapsedTime,
                         globState.globalHits.globalHits.nbMCHit - oldHitsNb, globState.globalHits.globalHits.nbMCHit,
                         (double) (globState.globalHits.globalHits.nbMCHit - oldHitsNb) /
                         (elapsedTime),
                         globState.globalHits.globalHits.nbDesorbed - oldDesNb, globState.globalHits.globalHits.nbDesorbed,
                         (double) (globState.globalHits.globalHits.nbDesorbed - oldDesNb) /
                         (elapsedTime));
    }
};

void printUsageAndExit( const char* argv0 )
{
    fprintf( stderr, "Usage  : %s [options]\n", argv0 );
    fprintf( stderr, "Options: --file    | -f <filename>        Specify file for model input\n" );
    fprintf( stderr, "         --auto    |                      Use auto generated\n" );
    fprintf( stderr, "         --angle   |                      Angle for auto generated geometry\n" );
    fprintf( stderr, "         --help    | -h                   Print this usage message\n" );
    fprintf( stderr, "         --size    | -s <launchsize>      Set kernel launch size\n" );
    fprintf( stderr, "         --size=<width>x<height>[x<depth>]\n" );
    fprintf( stderr, "         --loop    | -l <nbLoops>         Set number of simulation loops\n" );
    fprintf( stderr, "         --ndes    | -d <nbDesorptions>   Set number of desorptions for sim. end\n" );
    fprintf( stderr, "         --ndes=<d1>x<d2>[x<d3>]\n" );
    fprintf( stderr, "         --nhit    | -n <nbHits>          Set approx. number of hits for the simulation (overwrites --loop)\n" );
    fprintf( stderr, "         --quiet   | -q                   Set terminal output messages to a minimum\n" );
    fprintf( stderr, "         --time    | -t                   Time limit for simulation in seconds (e.g. 0.5)\n" );
    fprintf( stderr, "         --nprints | -i                   Print runtime output n times based on -l or -t {default 10}\n" );
    fprintf( stderr, "      --printevery | -j                   Print runtime output every n_th loop\n" );
    fprintf( stderr, "  --printEveryNMin | -k                   Print runtime output every k minutes\n" );
    fprintf( stderr, "      --depth      |                      Recursive max depth for secondary rays (reflections)\n" );
    fprintf( stderr, "    --cyclesForRNG | -r                   Number of cycles the RNG should be buffered for {default 1}\n" );
    fprintf( stderr, "      --directRand |                      Unimplemented\n" );
    //fprintf( stderr, "         --dim=<width>x<height>        Set image dimensions; defaults to 512x384\n" );
    exit(1);
}

void parseSize( const char* arg, size_t& kernelSize )
{

    // look for an 'x': <width>x<height>
    size_t width_end = strchr( arg, 'x' ) - arg;
    size_t height_begin = width_end + 1;

    if ( height_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        const char *height_arg = &arg[height_begin];

        // copy width to null-terminated string
        char width_arg[32];
        strncpy( width_arg, arg, width_end );
        width_arg[width_end] = '\0';

        // terminate the width string
        width_arg[width_end] = '\0';

        kernelSize = atoi(width_arg) * atoi(height_arg);

        size_t height_end = strchr( strchr( arg, 'x' )+1, 'x' ) - arg;
        size_t depth_begin = height_end + 1;

        if (depth_begin < strlen( arg ) )
        {
            const char *depth_arg = &arg[depth_begin];
            kernelSize *= atoi(depth_arg);
            if(kernelSize>0)
                return;
        }

        if(kernelSize>0)
            return;
    }

    std::cout << "#GPUTestsuite: Failed to parse width, height from string '" << std::string( arg ) << "'"  << std::endl;
    throw;
}

void parseDesLimits( const char* arg, std::list<size_t>& limits )
{

    // look for an 'x': <width>x<height>
    size_t lim_end = strchr( arg, 'x' ) - arg;
    size_t lim_begin = 0;
    lim_end = std::min(lim_end,strlen(arg));

    while( lim_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        //const char *this_arg = &arg[next_begin];

        // copy width to null-terminated string
        char this_arg[32];
        strncpy( this_arg, &arg[lim_begin], lim_end-lim_begin );

        // terminate the width string
        this_arg[lim_end-lim_begin] = '\0';

        limits.emplace_back(strtod(this_arg,nullptr));
        lim_begin = lim_end + 1;
        lim_end = strchr( &arg[lim_end]+1, 'x' ) - arg;
        lim_end = std::min(lim_end,strlen(arg));
    }
    if(!limits.empty())
        return;
    std::cout << "#GPUTestsuite: Failed to parse desorption limits from string '" << std::string( arg ) << "'"  << std::endl;
    throw;
}

// helper class for flexible argument initialization
class CharPVec {
public:
    CharPVec() {};

    CharPVec(int size) : vec(size, nullptr) {};

    CharPVec(const std::vector<std::string> &svec) : vec(svec.size(), nullptr) {
        cpy(svec);
    };

    ~CharPVec() { clear(); };

    char **data() { return vec.data(); }

    void cpy(const std::vector<std::string> &svec) {
        if (vec.size() != svec.size()) vec.resize(svec.size());
        for (int i = 0; i < vec.size(); ++i) {
            vec[i] = new char[svec[i].size() + 1];
            std::snprintf(vec[i], std::size(svec[i])+1, "%s", svec[i].c_str());
        }
    }

    void clear() {
        for (auto &c: vec) delete[] c;
    }

    // member
    std::vector<char *> vec;
};

// TODO: model->tri_facetOffset all 0
int mainV1(int argc, char **argv) {

    std::string fileName ;

    std::list<size_t> limits; // Number of desorptions: empty=use other end condition
    size_t nbLoops = 0;               // Number of Simulation loops
    size_t launchSize = 1;                  // Kernel launch size
    size_t nPrints = 10;
    size_t printPerN = 100000;
    double angle = 0.0; // timeLimit / -i or direct by -k
    double printEveryNMinutes = 0.0; // timeLimit / -i or direct by -k
    double timeLimit = 0.0;
    bool silentMode = false;
    flowgpu::MolflowGPUSettings simParams{};
    std::string outputPath;

    for(int i = 1; i < argc; ++i ) {
        char* p;
        if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 ) {
            printUsageAndExit( argv[0] );
        } else if( strcmp( argv[i], "--file" ) == 0 || strcmp( argv[i], "-f" ) == 0 ) {
            if( i < argc-1 ) {
                fileName = argv[++i];
                if(!std::filesystem::exists(fileName)){
                    std::cout << "File "<<fileName<<" does not exist!"<< std::endl;
                    exit(0);
                }
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if( strcmp( argv[i], "--auto" ) == 0) {
            if( i < argc-1 ) {
                angle = strtod(argv[++i], &p);
                // TODO: Angle check in bounds ]0,90[
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--size") == 0 || strcmp( argv[i], "-s" ) == 0 ) {
            if( i < argc-1 ) {
                launchSize = strtoul(argv[++i],&p,10);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strncmp( argv[i], "--size=", 7 ) == 0 ) {
            const char *size_arg = &argv[i][7];
            parseSize(size_arg, launchSize);
        } else if ( strcmp( argv[i], "--loop") == 0  || strcmp( argv[i], "-l" ) == 0 ) {
            if( i < argc-1 ) {
                nbLoops = strtoul(argv[++i],&p,10);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--nhit") == 0  || strcmp( argv[i], "-n" ) == 0 ) {
            if( i < argc-1 ) {
                nbLoops = (size_t)(strtoul(argv[++i],&p,10) / launchSize);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strncmp( argv[i], "--ndes=",7) == 0  || strcmp( argv[i], "-d" ) == 0 ) {
            if ( strncmp( argv[i], "--ndes=",7) == 0) {
                const char *size_arg = &argv[i][7];
                parseDesLimits(size_arg, limits);
                //++i;
                for(auto& lim : limits) std::cout << " lim : "<< lim << std::endl;
            }
            else if( i < argc-1 ) {
                limits.emplace_back(strtod(argv[++i],&p));
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--nprints") == 0  || strcmp( argv[i], "-i" ) == 0 ) {
            if( i < argc-1 ) {
                nPrints = strtoul(argv[++i],&p,10);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--time") == 0  || strcmp( argv[i], "-t" ) == 0 ) {
            if( i < argc-1 ) {
                timeLimit = strtod(argv[++i],&p);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--printevery") == 0  || strcmp( argv[i], "-j" ) == 0 ) {
            if( i < argc-1 ) {
                printPerN = strtoul(argv[++i],&p,10);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--printEveryNMin") == 0  || strcmp( argv[i], "-k" ) == 0 ) {
            if( i < argc-1 ) {
                printEveryNMinutes = strtod(argv[++i],&p);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--depth") == 0) {
            if( i < argc-1 ) {
                simParams.recursiveMaxDepth = strtoul(argv[++i],&p,10);
            } else {
                printUsageAndExit( argv[0] );
            }
        }
#ifdef RNG_BULKED
        else if ( strcmp( argv[i], "--cyclesForRNG") == 0 || strcmp( argv[i], "-r") == 0) {
            if( i < argc-1 ) {
                simParams.cyclesRNG = strtoul(argv[++i],&p,10);
            } else {
                printUsageAndExit( argv[0] );
            }
        }
#endif
        else if ( strcmp( argv[i], "--directRand") == 0) {
            simParams.randomNumberMethod = true;
        } else if ( strcmp( argv[i], "--quiet") == 0  || strcmp( argv[i], "-q" ) == 0 ) {
            silentMode = true;
        } else if ( strcmp( argv[i], "--offset") == 0) {
            if( i < argc-1 ) {
                simParams.offsetMagnitude = strtof(argv[++i],&p);
                fmt::print("   Offset to center: {}\n", simParams.offsetMagnitude);
            } else {
                printUsageAndExit( argv[0] );
            }
        } else if ( strcmp( argv[i], "--offset2N") == 0) {
            if( i < argc-1 ) {
                simParams.offsetMagnitudeN = strtof(argv[++i],&p);
                fmt::print("   Offset to center: {}\n", simParams.offsetMagnitudeN);
            } else {
                printUsageAndExit( argv[0] );
            }
        }
        else if ( strcmp( argv[i], "--op") == 0) {
            if( i < argc-1 ) {
                outputPath = argv[++i];
                fmt::print("   Output path: {}\n", outputPath);
            } else {
                printUsageAndExit( argv[0] );
            }
        }
        else {
            fprintf( stderr, "Unknown option '%s'\n", argv[i] );
            printUsageAndExit( argv[0] );
        }
    }

    // Check for valid end conditions
    if(!(nbLoops > 0 || timeLimit >= 1e-6)){
        Log::console_error("No valid end condition set!\n");
        return 42;
    }

    SimulationControllerGPU gpuSim(0, 0, 1, nullptr, std::make_shared<ProcComm>());
    std::shared_ptr<flowgpu::Model> model;
    std::shared_ptr<flowgpu::MolflowGPUSettings> settings;

    // for export
    GlobalSimuState globState{};
    std::shared_ptr<MolflowSimulationModel> simModel = std::make_shared<MolflowSimulationModel>();

    //flowgpu::Model* model = flowgeom::initializeModel(fileName);
    {
        SimulationManager simManager{};
        simManager.interactiveMode = true;

        std::string output_filename = std::filesystem::path(fileName).replace_extension().string();
        std::vector<std::string> argv = {"tester", "--reset", "-t", "10",
                                         "-o", FileUtils::GetFilename(output_filename)+"_out.xml"};
        if(!outputPath.empty()) {
            argv.emplace_back("--outputPath");
            argv.emplace_back(outputPath);
        }
        if(!fileName.empty()) {
            argv.emplace_back("--file");
            argv.push_back(fileName);
        }
        else {
            argv.emplace_back("--auto");
            //argv.push_back(std::to_string(angle));
        }


        CharPVec argc_v(argv);
        char **args = argc_v.data();
       /* InitializerGPU::initFromArgv(argv.size(), (args), &simManager, model);
        InitializerGPU::initFromFile(&simManager, model, &globState);
*/
        if(-1 < InitializerGPU::initFromArgv(argv.size(), (args), &simManager, simModel, model)){
#if defined(USE_MPI)
            MPI_Finalize();
#endif
            return 41;
        }

#if defined(USE_MPI)
        MFMPI::mpi_transfer_simu();
#endif

        if(angle > 0.0 && InitializerGPU::initAutoGenerated(&simManager, simModel, &globState, 100.0, 10, DegToRad(angle))){
#if defined(USE_MPI)
            MPI_Finalize();
#endif
            return 43;
        }
        if(!fileName.empty() && InitializerGPU::initFromFile(&simManager, simModel, &globState, model)){
#if defined(USE_MPI)
            MPI_Finalize();
#endif
            return 42;
        }


        if(flowgpu::loadFromSimModel(model, settings, *simModel.get())){
            Log::console_error("Error on loading and parsing geometry model.");
            return 44;
        }
    }
    //flowgpu::Model* model = flowgpu::loadFromExternalSerialization(fileName);
    //flowgpu::Model* test = flowgeom::loadFromExternalSerialization("minimalout.xml");
    //delete model;
    //model = test;

    // Set desorption limit if used
    if(!limits.empty()) {
        model->ontheflyParams.desorptionLimit = limits.front();
        limits.pop_front();
    }
    else{
        model->ontheflyParams.desorptionLimit = 0;
    }
    // Get parsed methods etc
    *settings = simParams;

    std::cout << "#GPUTestsuite: Loading simulation with kernel size: " << launchSize << std::endl;
    gpuSim.LoadSimulation(model, launchSize);

    std::cout << "#GPUTestsuite: Starting simulation with " << launchSize << " threads per launch => " << nbLoops << " runs "<<std::endl;

    //const int printPerNRuns = std::max(1, static_cast<int>(nbLoops/nPrints)); // prevent n%0 operation
    uint64_t printPerNRuns = std::min(static_cast<uint64_t>(printPerN), static_cast<uint64_t>(nbLoops/nPrints)); // prevent n%0 operation
    printPerNRuns = std::max(printPerNRuns, static_cast<uint64_t>(1));

    // -i
    if(nbLoops==0 && printEveryNMinutes <= 0.0 && timeLimit > 1e-6){
            printEveryNMinutes = timeLimit / nPrints;
    }

    auto start_total = std::chrono::steady_clock::now();
    auto t0 = start_total;
    double nextPrintAtMin = printEveryNMinutes;

    double raysPerSecondMax = 0.0;
    double desPerSecondMax = 0.0;
    /*double raysPerSecondSum = 0.0;
    uint64_t nRaysSum = 0.0;*/

    size_t refreshForStop = std::numeric_limits<size_t>::max();
    size_t loopN = 0;
    do{
    //for(loopN = 0; loopN < nbLoops && !gpuSim.hasEnded; ++loopN){
        //auto start = std::chrono::high_resolution_clock::now();

        gpuSim.RunSimulation();

        auto t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double,std::ratio<60,1>> elapsedMinutes = t1 - start_total;

        // Fetch end conditions
        if(nbLoops > 0 && loopN >= nbLoops)
            gpuSim.hasEnded = true;
        else if(timeLimit >= 1e-6 && elapsedMinutes.count() >= timeLimit)
            gpuSim.hasEnded = true;

        if((!silentMode && ((nbLoops > 0 && (loopN + 1) % printPerNRuns == 0) || (printEveryNMinutes > 0.0 && elapsedMinutes.count() > nextPrintAtMin)) || refreshForStop <= loopN || gpuSim.hasEnded)){
            if(printEveryNMinutes > 0.0 && elapsedMinutes.count() > nextPrintAtMin)
                nextPrintAtMin += printEveryNMinutes;

            auto t1 = std::chrono::steady_clock::now();
            std::chrono::duration<double,std::milli> elapsed = t1 - t0;
            t0 = t1;

            uint64_t nbHits = gpuSim.GetSimulationData();

            // end on leak
            /*if(gpuSim.figures.total_leak)
                break;//return gpuSim.figures.total_leak;*/
            if(model->ontheflyParams.desorptionLimit != 0) {
                // add remaining steps to current loop count, this is the new approx. stop until desorption limit is reached
               refreshForStop = loopN + gpuSim.RemainingStepsUntilStop();
               std::cout << " Stopping at " << loopN << " / " << refreshForStop << std::endl;
            }
            if(gpuSim.hasEnded){
                // if there is a next des limit, handle that
                if(!limits.empty()) {
                    model->ontheflyParams.desorptionLimit = limits.front();
                    limits.pop_front();
                    gpuSim.hasEnded = false;
                    gpuSim.endCalled = false;
                    gpuSim.AllowNewParticles();
                    std::cout << " Handling next des limit " << model->ontheflyParams.desorptionLimit << std::endl;
                }
            }
            static const uint64_t nRays = launchSize * printPerNRuns;
            //double rpsRun = (double)nRays / elapsed.count() / 1000.0;
            double rpsRun = (double)(nbHits) / elapsed.count() / 1000.0;
            raysPerSecondMax = std::max(raysPerSecondMax,rpsRun);
            //raysPerSecondSum += rpsRun;
            std::cout << "--- Run #" << loopN + 1 << " \t- Elapsed Time: " << elapsed.count() / 1000.0 << " s \t--- " << rpsRun << " MRay/s ---" << std::endl;
            printf("--- Trans Prob: %e\n",gpuSim.GetTransProb());
        }

        ++loopN;
    } while(!gpuSim.hasEnded);
    auto finish_total = std::chrono::steady_clock::now();
    gpuSim.GetSimulationData(false);
    std::chrono::duration<double> elapsed = finish_total - start_total;
    if(elapsed.count() / 60.0 >= 1.0)
        fmt::print("--         Total Elapsed Time: {} min ---\n",elapsed.count() / 60.0);
    else
        fmt::print("--         Total Elapsed Time: {} sec ---\n",elapsed.count());
    std::cout << "--         Avg. Rays per second: " << (double)gpuSim.figures.total_counter/elapsed.count()/1.0e6 << " MRay/s ---" << std::endl;
    //std::cout << "--         Avg. Rays per second: " << raysPerSecondSum/(nbLoops/printPerNRuns) << " MRay/s ---" << std::endl;
    std::cout << "--         Max  Rays per second: " << raysPerSecondMax << " MRay/s ---" << std::endl;

    // Export results
    // 1st convert from GPU types to CPU types
    // 2nd save with XML
    gpuSim.ConvertSimulationData(globState);




    bool createZip = std::filesystem::path(SettingsIO::outputFile).extension() == ".zip";
    SettingsIO::outputFile = std::filesystem::path(SettingsIO::outputFile).replace_extension(".xml").string();

    std::string fullOutFile = std::filesystem::path(SettingsIO::outputPath).append(SettingsIO::outputFile).string();

    FlowIO::WriterXML writer(false, true);
    pugi::xml_document newDoc;
    newDoc.load_file(fullOutFile.c_str());
    writer.SaveGeometry(newDoc, simModel);
    writer.SaveSimulationState(newDoc, simModel, globState);
    if(!writer.SaveXMLToFile(newDoc, fullOutFile)){
        return 42;
    }

    if(createZip){
        Log::console_msg_master(3, "Compressing xml to zip...\n");

        //Zipper library
        std::string fileNameWithZIP = std::filesystem::path(fullOutFile).replace_extension(".zip").string();
        if (std::filesystem::exists(fileNameWithZIP)) { // should be workFile == inputFile
            try {
                std::filesystem::remove(fileNameWithZIP);
            }
            catch (const std::exception &e) {
                Log::console_error("Error compressing to \n{}\nMaybe file is in use:\n{}",fileNameWithZIP.c_str(),e.what());
            }
        }
        ZipFile::AddFile(fileNameWithZIP, fullOutFile, FileUtils::GetFilename(fullOutFile));
        //At this point, if no error was thrown, the compression is successful
        try {
            std::filesystem::remove(fullOutFile);
        }
        catch (const std::exception &e) {
            Log::console_error("Error removing\n{}\nMaybe file is in use:\n{}",fullOutFile.c_str(),e.what());
        }
    }
    /*std::chrono::duration<double,std::milli> elapsed = finish_total - start_total;
    std::cout << "-- Total Elapsed Time: " << elapsed.count() << " ms ---" << std::endl;
    double raysPerSecond = (double)(gpuSim.GetSimulationData())/(elapsed.count()/1000);
    std::cout << "-- Rays per second: " << raysPerSecond << " Ray/s ---" << std::endl;*/

    //gpuSim.CloseSimulation();

    return 0;
}

static constexpr const char* molflowCliLogo = R"(
  __  __     _  __ _             ___ ___ _   _
 |  \/  |___| |/ _| |_____ __ __/ __| _ \ | | |
 | |\/| / _ \ |  _| / _ \ V  V / (_ |  _/ |_| |
 |_|  |_\___/_|_| |_\___/\_/\_/ \___|_|  \___/
    )";

void GatherResults(MolflowSimulationModel& model, GlobalSimuState& globSim){
    for(size_t i = 0; i < model.facets.size(); i++ ) {
#if defined(MOLFLOW)
        auto f = std::dynamic_pointer_cast<MolflowSimFacet>(model.facets[i]);
        if (f->sh.anglemapParams.record) { //Recording, so needs to be updated
            //Retrieve angle map from hits dp
            f->angleMap.pdf = globSim.facetStates[i].recordedAngleMapPdf;
        }
#endif
    }
}

int main(int argc, char** argv) {

#if defined(WIN32) || defined(__APPLE__)
    setlocale(LC_ALL, "C");
#else
    std::setlocale(LC_ALL, "C");
#endif

#if defined(USE_MPI)
    MFMPI::mpi_initialize();
#endif

    Log::console_msg_master(1, "{}\n", molflowCliLogo);
#if defined(DEBUG)
    Log::console_msg_master(1, " Built at: {} {}\n", __DATE__, __TIME__);
#else
    Log::console_msg_master(1, " Built on: {}\n", __DATE__);
#endif

    SimulationManager simManager{MFMPI::world_rank};
    simManager.interactiveMode = true;
    simManager.useGPU = true;
    simManager.useCPU = false;
    std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
    std::shared_ptr<flowgpu::Model> gpu_model = std::make_shared<flowgpu::Model>();
    std::shared_ptr<flowgpu::MolflowGPUSettings> settings = std::make_shared<flowgpu::MolflowGPUSettings>();

    GlobalSimuState globState{};

    if(-1 < InitializerGPU::initFromArgv(argc, argv, &simManager, model, gpu_model)){
        Log::console_error("Error: Initializing from argument list\n");
#if defined(USE_MPI)
        MPI_Finalize();
#endif
        return 41;
    }

    // Start to transfer simulatsion data to other nodes via broadcast
#if defined(USE_MPI)
    MFMPI::mpi_transfer_simu();
#endif


    if(SettingsIO::autogenerateTest < 0.0 && InitializerGPU::initFromFile(&simManager, model, &globState, gpu_model)){
        Log::console_error("Error: Loading test case\n");
#if defined(USE_MPI)
        MPI_Finalize();
#endif
        return 42;
    }
    else if(SettingsIO::autogenerateTest >= 0 && InitializerGPU::initAutoGenerated(&simManager, model, &globState,
                                                                           10.0, 10, DegToRad(SettingsIO::autogenerateTest)/*M_PI_4*/)){
        Log::console_msg_master(1,"Error: Generating test case\n");
#if defined(USE_MPI)
        MPI_Finalize();
#endif
        return 43;
    }

    *settings = Settings::simParams;
    settings->kernelDimensions[0] = Settings::kernelDimensions[0];
    settings->kernelDimensions[1] = Settings::kernelDimensions[1];

    if(settings->kernelDimensions[0] == 0 || settings->kernelDimensions[1] == 0) {
        Log::console_error("Invalid kernel dimension set: [{} x {}]\n", settings->kernelDimensions[0], settings->kernelDimensions[1]);
#if defined(USE_MPI)
        MPI_Finalize();
#endif
        return 40;
    }
    auto controllers = simManager.GetGPUControllers();
    for(auto& con : controllers) {
        std::dynamic_pointer_cast<SimulationControllerGPU>(con)->ChangeParams(settings);
    }
    /*gpu_model = */flowgpu::loadFromSimModel(gpu_model, settings, *model.get());

    if(Settings::simDuration == 0 && model->otfParams.desorptionLimit == 0){
        Log::console_error("Neither a time limit nor a desorption limit has been set!\n");
        return 44;
    }
    size_t oldHitsNb = globState.globalHits.globalHits.nbMCHit;
    size_t oldDesNb = globState.globalHits.globalHits.nbDesorbed;
    RuntimeStatPrinter printer(oldHitsNb, oldDesNb);
    // Get autosave file name
    std::string autoSave = InitializerGPU::getAutosaveFile();

    //simManager.ReloadHitBuffer();
    //simManager.IncreasePriority();
    if(Settings::simDuration > 0)
        Log::console_msg_master(1,"[{}] Commencing simulation for {} seconds from {} desorptions.\n", Util::getTimepointString().c_str(), Settings::simDuration, globState.globalHits.globalHits.nbDesorbed);
    else if(model->otfParams.desorptionLimit > 0)
        Log::console_msg_master(1,"[{}] Commencing simulation to {} desorptions from {} desorptions.\n", Util::getTimepointString().c_str(), model->otfParams.desorptionLimit, globState.globalHits.globalHits.nbDesorbed);

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    simManager.interactiveMode = false;
#endif
    try {
        simManager.StartSimulation();
    }
    catch (const std::exception& e) {
        Log::console_error("[{}] ERROR: Starting simulation: {}\n",MFMPI::world_rank, e.what());
        Log::console_error("[{}] File folder {} -- {}\n",MFMPI::world_rank,SettingsIO::workPath.c_str(), SettingsIO::workFile.c_str());

#if defined(USE_MPI)
        MPI_Finalize();
#endif
        return 43;
    }

    Chronometer simTimer;
    simTimer.Start();
    double elapsedTime = 0.0;
    double nextPrintAtMin = Settings::outputDuration;

    bool endCondition = false;
    do {
        ProcessSleep(1000);

        elapsedTime = simTimer.Elapsed();
        if(model->otfParams.desorptionLimit != 0)
            endCondition = globState.globalHits.globalHits.nbDesorbed/* - oldDesNb*/ >= model->otfParams.desorptionLimit;

        if(endCondition){
            /*std::stringstream outFile;
            outFile << SettingsIO::outputPath << "/" <<"desorped_" << model->otfParams.desorptionLimit << "_" <<
                 std::filesystem::path(SettingsIO::outputFile).filename().string();*/
            std::string outFile = std::filesystem::path(SettingsIO::outputPath)
                    .append("desorped_")
                    .concat(std::to_string(model->otfParams.desorptionLimit))
                    .concat("_")
                    .concat(std::filesystem::path(SettingsIO::outputFile).filename().string()).string();

            try {
                FlowIO::WriterXML writer;
                if(!SettingsIO::workFile.empty()) {
                    std::filesystem::copy_file(SettingsIO::workFile, outFile,
                                               std::filesystem::copy_options::overwrite_existing);
                }
                else {
                    pugi::xml_document newDoc;
                    writer.SaveGeometry(newDoc, model);
                    //writer.SaveSimulationState(newDoc, model, globState);
                    writer.SaveXMLToFile(newDoc, outFile);
                    //SettingsIO::workFile = outFile;
                }
                writer.SaveSimulationState(outFile, model, globState);
            } catch(std::filesystem::filesystem_error& e) {
                Log::console_msg_master(1,"Warning: Could not create file: {}\n", e.what());
            }

            // if there is a next des limit, handle that
            if(!Settings::desLimit.empty()) {
                model->otfParams.desorptionLimit = Settings::desLimit.front();
                Settings::desLimit.pop_front();
                simManager.ForwardOtfParams(&model->otfParams);
                endCondition = false;
                Log::console_msg_master(1, " Handling next des limit {}\n", model->otfParams.desorptionLimit);

                try {
                    ProcessSleep(1000);
                    simManager.StartSimulation();
                }
                catch (const std::exception& e) {
                    Log::console_error("ERROR: Starting simulation: {}\n", e.what());
                    endCondition = true;
                }
            }
        }
        else if(Settings::autoSaveDuration && (uint64_t)(elapsedTime)%Settings::autoSaveDuration==0){ // autosave every x seconds
            Log::console_msg_master(2,"[{:.2}s] Creating auto save file {}\n", elapsedTime, autoSave.c_str());
            FlowIO::WriterXML writer;
            writer.SaveSimulationState(autoSave, model, globState);
        }

        if(Settings::outputDuration > 0.0 && elapsedTime > nextPrintAtMin) {// autosave every x seconds
            nextPrintAtMin += Settings::outputDuration;
            if((uint64_t)elapsedTime / Settings::outputDuration <= 1){
                printer.PrintHeader();
            }
            printer.Print(elapsedTime, globState);
        }

        // Check for potential time end
        if(Settings::simDuration > 0) {
            endCondition |= (elapsedTime >= (double) Settings::simDuration);
        }
    } while(!endCondition);
    simTimer.Stop();
    elapsedTime = simTimer.Elapsed();

    // Terminate simulation
    simManager.StopSimulation();
    simManager.KillAllSimUnits();
    GatherResults(*model, globState);
    Log::console_msg(1,"[{}][{}] Simulation finished!\n", MFMPI::world_rank, Util::getTimepointString().c_str());

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    printer.PrintHeader();
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    //TODO: Send output to master node for ordered output
    if(elapsedTime > 1e-4) {
        // Global result print --> TODO: ()
        printer.Print(elapsedTime, globState);
    }

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MFMPI::mpi_receive_states(model, globState);
    if(MFMPI::world_rank != 0){
        // Cleanup all files from nodes tmp path
        if (SettingsIO::outputPath.find("tmp") != std::string::npos) {
            std::filesystem::remove_all(SettingsIO::outputPath);
        }
        if(std::filesystem::exists(autoSave)){
            std::filesystem::remove(autoSave);
        }
    }
    // Finalize the MPI environment.
    MPI_Finalize();
    if(MFMPI::world_rank != 0){
        return 0;
    }
#endif //USE_MPI

    if(elapsedTime > 1e-4) {
        if(MFMPI::world_size > 1)
            printer.Print(elapsedTime, globState, true);
    }

    Log::console_msg_master(0,"Leaks: {}\n", globState.globalHits.nbLeakTotal);

    if(MFMPI::world_rank == 0){
        if(SettingsIO::outputFacetDetails) {
            FlowIO::Exporter::export_facet_details(&globState, model.get());

        }
        if(SettingsIO::outputFacetQuantities) {
            FlowIO::Exporter::export_facet_quantities(&globState, model.get());
        }

        // Export results
        //  a) Use existing autosave as base
        //  b) Create copy of input file
        // update geometry info (in case of param sweep)
        // and simply update simulation results
        bool createZip = std::filesystem::path(SettingsIO::outputFile).extension() == ".zip";
        SettingsIO::outputFile = std::filesystem::path(SettingsIO::outputFile).replace_extension(".xml").string();

        std::string fullOutFile = std::filesystem::path(SettingsIO::outputPath).append(SettingsIO::outputFile).string();
        if(std::filesystem::exists(autoSave)){
            std::filesystem::rename(autoSave, fullOutFile);
        }
        else if(!SettingsIO::overwrite){
            // Copy full file description first, in case outputFile is different
            if(!SettingsIO::workFile.empty() && std::filesystem::exists(SettingsIO::workFile)){
                try {
                    std::filesystem::copy_file(SettingsIO::workFile, fullOutFile,
                                               std::filesystem::copy_options::overwrite_existing);
                } catch (std::filesystem::filesystem_error &e) {
                    Log::console_error("Could not copy file to preserve initial file layout: {}\n", e.what());
                }
            }
        }
        FlowIO::WriterXML writer(false, true);
        pugi::xml_document newDoc;
        newDoc.load_file(fullOutFile.c_str());
        writer.SaveGeometry(newDoc, model);
        writer.SaveSimulationState(newDoc, model, globState);
        writer.SaveXMLToFile(newDoc, fullOutFile);

        if(createZip){
            Log::console_msg_master(3, "Compressing xml to zip...\n");

            //Zipper library
            std::string fileNameWithZIP = std::filesystem::path(fullOutFile).replace_extension(".zip").string();
            if (std::filesystem::exists(fileNameWithZIP)) { // should be workFile == inputFile
                try {
                    std::filesystem::remove(fileNameWithZIP);
                }
                catch (std::exception &e) {
                    Log::console_error("Error compressing to \n{}\nMaybe file is in use:\n{}",fileNameWithZIP.c_str(),e.what());
                }
            }
            ZipFile::AddFile(fileNameWithZIP, fullOutFile, FileUtils::GetFilename(fullOutFile));
            //At this point, if no error was thrown, the compression is successful
            try {
                std::filesystem::remove(fullOutFile);
            }
            catch (std::exception &e) {
                Log::console_error("Error removing\n{}\nMaybe file is in use:\n{}",fullOutFile.c_str(),e.what());
            }
        }
    }

    // Cleanup
    SettingsIO::cleanup_files();

    return 0;
}