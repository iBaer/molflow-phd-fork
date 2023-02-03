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

#include "../src_shared/SimulationManager.h"
#include "gtest/gtest.h"
#include "../src/Initializer.h"
#include "../src/ParameterParser.h"
#include "../src/Simulation/MolflowSimFacet.h"
//#define MOLFLOW_PATH ""

#include <filesystem>
#include <fstream>

// hash time to create random file name
#include <ctime>
#include <functional>
#include <Helper/Chronometer.h>
#include <memory>
#include <numeric>
#include <cmath>
#include <IO/WriterXML.h>
#include <IO/CSVExporter.h>
#include <SettingsIO.h>
#include <fmt/core.h>

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?"
#endif

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
namespace {

    class SimulationFixture : public ::testing::TestWithParam<std::string> {
    protected:
        SimulationFixture() {
            // You can do set-up work for each test here.
        }

        ~SimulationFixture() override {
            // You can do clean-up work that doesn't throw exceptions here.
        }

        void SetUp() override {
            // Code here will be called immediately after the constructor (right
            // before each test).
        }

        void TearDown() override {
            if (!SettingsIO::workPath.empty() && (SettingsIO::workPath != "." || SettingsIO::workPath != "./"))
                std::filesystem::remove_all(SettingsIO::workPath);
            // Code here will be called immediately after each test (right
            // before the destructor).
        }

        // Objects declared here can be used by all tests in the test suite for Foo.
    };

    class ValidationFixture : public SimulationFixture {

    };


    INSTANTIATE_TEST_SUITE_P(
            Performance,
            SimulationFixture,
            ::testing::Values(
                    "TestCases/B01-lr1000_pipe.zip",
                    "TestCases/B02-lr10_pipe_tex.zip",
                    "TestCases/B03-lr10_pipe_prof.zip",
                    "TestCases/B04-lr10_pipe_trans.zip"
            ));

    INSTANTIATE_TEST_SUITE_P(
            Results,
            ValidationFixture,
            ::testing::Values(
                    "TestCases/01-quick_pipe_profiles_textures_2sided.zip",
                    "TestCases/02-timedependent_with_two_parameters.zip",
                    "TestCases/02b-timedependent_with_three_parameters.zip",
                    "TestCases/03-anglemap_record.zip",
                    "TestCases/03b-anglemap_record_transparent_target.zip",
                    "TestCases/04-anglemap_desorb.zip",
                    "TestCases/04b-anglemap_desorb-sticking source.zip",
                    "TestCases/05-three_structures_nonsquare_textures.zip",
                    "TestCases/06-dynamic_desorption_from_synrad.zip",
                    "TestCases/07-volume_decay.zip",
                    "TestCases/08-wall_sojourn_time.zip",
                    "TestCases/09-histograms.zip"
            ));

    struct Stats {
        std::string commitHash;
        double min{-1.0}, max{-1.0}, avg{-1.0}, med{-1.0};

        friend std::istream &operator>>(std::istream &is, Stats &r) {
            // Get current position
            auto len = is.tellg();
            char line[256];
            is.getline(line, 256);
            // Return to position before "Read line".
            is.seekg(len, std::ios_base::beg);

            size_t countDelimiter = 0;
            size_t pos = 0;
            while ((line[pos] != '\n' && line[pos] != '\0') && pos < 256) {
                if (line[pos] == ' ' || line[pos] == '\t') {
                    ++countDelimiter;
                }
                ++pos;
            }

            r = Stats();
            if (countDelimiter == 4) {
                is >> r.commitHash >> r.max >> r.min >> r.med >> r.avg;
                is.getline(line, 256);
                return is;
            } else if (countDelimiter == 1) {
                is >> r.commitHash >> r.max;
                is.getline(line, 256);
                //is.getline(line, 256);
                return is;
            } else { // unknown format, skip line
                is.getline(line, 256);
                return is;
            }
        }

        friend std::ostream &operator<<(std::ostream &os, Stats const &r) {
            return os << r.commitHash << "\t" << std::scientific << std::setprecision(8) << r.max << "\t" << r.min
                      << "\t" << r.med << "\t" << r.avg;
        }

        bool operator<(Stats const &other) const {
            return other.max < max;
        }

        /*Stats& operator=(const OldStats& src){
            commitHash = src.commitHash;
            max = src.max;
            min = src.min;
            avg = src.avg;
            med = src.med;

            return *this;
        }*/
    };

    TEST_P(SimulationFixture, PerformanceOkay) {
        std::string testFile = GetParam();
        std::string outPath = "TPath_PO_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        Log::console_msg(1, "Filename: {}\n", testFile);
        std::string timeRecFile = "./time_record_" + testFile.substr(0, testFile.size() - 4) + ".txt";

        {
            // Check if test was already successful, restarting the test suite will then skip the test
            if (std::filesystem::exists(timeRecFile)) {
                std::ifstream ifs(timeRecFile);
                std::vector<Stats> prevRun;
                prevRun.insert(prevRun.begin(), std::istream_iterator<Stats>(ifs), std::istream_iterator<Stats>());

                std::string currentCommit = GIT_COMMIT_HASH;
                currentCommit = currentCommit.substr(0, 8); // only first 8 hex digits
                for (auto &run: prevRun) {
                    if (run.commitHash == currentCommit) {
                        Log::console_msg(1, "Test was successfully run in a previous attempt, skip ...\n");
                        GTEST_SKIP();
                    }
                }
            }
        }

        size_t nbFails = 0;
        bool fastEnough = false;
        const size_t nRuns = 5;
        const size_t keepNEntries = 20;
        const size_t runForTSec = 20;
        std::vector<double> perfTimes;
        for (size_t runNb = 0; runNb < nRuns; ++runNb) {
            SimulationManager simManager{0};
            std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
            GlobalSimuState globState{};

            /*std::vector<char *> argv = {"tester", "--config", "simulation.cfg", "--reset", "--file"};
            char * fileName_c = new char[testFile.size() + 1];
            std::copy(testFile.begin(), testFile.end(), fileName_c);
            fileName_c[testFile.size()] = '\0';
            argv.push_back(fileName_c);
*/
            {
                std::vector<std::string> argv = {"tester", "--config", "simulation.cfg", "--reset",
                                                 "--file", testFile,
                                                 "--outputPath", outPath};

                CharPVec argc_v(argv);
                char **args = argc_v.data();
                Initializer::initFromArgv(argv.size(), (args), &simManager, model);
                Initializer::initFromFile(&simManager, model, &globState);
            }

            size_t oldHitsNb = globState.globalHits.globalHits.nbMCHit;
            size_t oldDesNb = globState.globalHits.globalHits.nbDesorbed;

            EXPECT_NO_THROW(simManager.StartSimulation());

            Chronometer simTimer(false);
            simTimer.Start();
            double elapsedTime;

            bool endCondition = false;
            do {
                ProcessSleep(1000);
                elapsedTime = simTimer.Elapsed();
                if (model->otfParams.desorptionLimit != 0)
                    endCondition = globState.globalHits.globalHits.nbDesorbed >= model->otfParams.desorptionLimit;
                // Check for potential time end
                if (Settings::simDuration > 0) {
                    endCondition |= elapsedTime >= Settings::simDuration;
                }
            } while (!endCondition);
            simTimer.Stop();

            // Stop and copy results
            simManager.StopSimulation();
            simManager.KillAllSimUnits();

            perfTimes.emplace_back((double) (globState.globalHits.globalHits.nbMCHit - oldHitsNb) / (elapsedTime));
            EXPECT_EQ(0, oldDesNb);
            EXPECT_EQ(0, oldHitsNb);
            EXPECT_LT(0, globState.globalHits.globalHits.nbDesorbed);
            EXPECT_LT(0, globState.globalHits.globalHits.nbMCHit);

            Log::console_msg(1, "[Run {}/{}] Current Hit/s: {:e}\n", runNb, nRuns, perfTimes.back());
        };

        // Compare to old performance values here

        if (!perfTimes.empty()) {
            std::sort(perfTimes.begin(), perfTimes.end());

            Stats currentRun;
            currentRun.commitHash = GIT_COMMIT_HASH;
            currentRun.commitHash = currentRun.commitHash.substr(0, 8); // only first 8 hex digits
            currentRun.min = perfTimes.front();
            currentRun.max = perfTimes.back();
            double sum = std::accumulate(perfTimes.begin(), perfTimes.end(), 0.0);
            currentRun.avg = sum / perfTimes.size();
            currentRun.med = (perfTimes.size() % 2 == 0)
                             ? (perfTimes[perfTimes.size() / 2 - 1] + perfTimes[perfTimes.size() / 2]) / 2
                             : perfTimes[perfTimes.size() / 2];

            std::cout << "Current Run: " << currentRun << std::endl;

            std::vector<Stats> prevRun;
            std::string testName(::testing::UnitTest::GetInstance()->current_test_info()->name());

            // If a record file for this test already exists
            //  load the number 1 entry
            if (std::filesystem::exists(timeRecFile)) {
                std::ifstream ifs(timeRecFile);
                prevRun.insert(prevRun.begin(), std::istream_iterator<Stats>(ifs), std::istream_iterator<Stats>());

                std::cout << "Prev Run: " << prevRun.front() << std::endl;
                // Check either, only if previous results could be found
                fastEnough = currentRun.max >= 0.95 * prevRun.front().max;

                if (!fastEnough &&
                    prevRun.front().med > 0.0) { // check to prevent free pass for old entries with only max
                    EXPECT_GE(currentRun.med, 0.95 * prevRun.front().med);
                    fastEnough = true;
                }
                if (!fastEnough && prevRun.front().med > 0.0 && currentRun.max > prevRun.front().med) {
                    EXPECT_GE(currentRun.max, prevRun.front().med);
                    fastEnough = true;
                }
                if (!fastEnough) { // check to prevent free pass for old entries with only max
                    EXPECT_GE(currentRun.max, 0.95 * prevRun.front().max);
                    if (currentRun.max >= 0.95 * prevRun.front().max)
                        fastEnough = true;
                }
            } else {
                fastEnough = true; // force update
            }

            // Only enter a test if it meets any of the success criteria
            if (fastEnough) {
                // Keep top 20 in list
                prevRun.push_back(currentRun);
                std::sort(prevRun.begin(), prevRun.end());
                for (auto iter_o = prevRun.begin(); iter_o != prevRun.end(); ++iter_o) {
                    for (auto iter_i = iter_o + 1; iter_i != prevRun.end(); ++iter_i) { // "slower entry"
                        if (iter_i->commitHash == iter_o->commitHash) {
                            prevRun.erase(iter_i--); // remove outer entry, as it is slower
                        }
                    }
                }
                std::ofstream ofs(timeRecFile);
                for (size_t lineN = 0; lineN < std::min(prevRun.size(), keepNEntries); ++lineN) {
                    if (!prevRun[lineN].commitHash.empty())
                        ofs << prevRun[lineN] << std::endl;
                }
            }
        }
    }


    TEST_P(ValidationFixture, ResultsOkay) {
        std::string testFile = GetParam();
        std::string outPath = "TPath_RO_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        Log::console_msg(1, "Filename: {}\n", testFile);
        size_t nbFails = 0;
        size_t nCorrect = 0;
        bool fastEnough = false;
        const size_t nRuns = 20;
        const size_t correctStreak = 3;
        const size_t keepNEntries = 20;
        const size_t runForTSec = 30;
        std::vector<double> perfTimes;

        SimulationManager simManager{0};
        simManager.interactiveMode = false;
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        {
            {
                std::vector<std::string> argv = {"tester", "--verbosity", "1", "-t", std::to_string(runForTSec),
                                             "--file", testFile, "--outputPath", outPath};
                CharPVec argc_v(argv);
                char **args = argc_v.data();
                if (-1 < Initializer::initFromArgv(argv.size(), (args), &simManager, model)) {
                    exit(41);
                }
                if (Initializer::initFromFile(&simManager, model, &globState)) {
                    exit(42);
                }
            }
            {
                double timeExpect = std::log(model->facets.size());
                //timeExpect = timeExpect * timeExpect;
                timeExpect = std::pow(timeExpect, 1.5);
                if (!model->tdParams.moments.empty())
                    timeExpect += std::pow(std::log(model->tdParams.moments.size()), 3.0);

                timeExpect += std::max(0.0,
                                       std::pow(std::log(std::sqrt(model->sh.nbFacet * sizeof(FacetHitBuffer))), 2.0) -
                                       10.0);
                timeExpect += std::max(0.0, 1.1 * std::sqrt(std::exp(std::log(std::sqrt(model->size())))));
                timeExpect *= 4.0;
                Settings::simDuration = static_cast<uint64_t>(std::min(50.0 + timeExpect, 600.0));
            }
        }


        GlobalSimuState oldState = globState;
        globState.Reset();
        size_t oldHitsNb = oldState.globalHits.globalHits.nbMCHit;
        size_t oldDesNb = oldState.globalHits.globalHits.nbDesorbed;
        EXPECT_NE(0, oldDesNb);
        EXPECT_NE(0, oldHitsNb);
        EXPECT_EQ(0, globState.globalHits.globalHits.nbDesorbed);
        EXPECT_EQ(0, globState.globalHits.globalHits.nbMCHit);

        int stepSizeTime = (int) (Settings::simDuration * ((double) (1.0) / nRuns));
        Settings::simDuration = stepSizeTime;

        // One test run with time, next with desorptions
        {
            Initializer::initTimeLimit(model, stepSizeTime);


            EXPECT_NO_THROW(simManager.StartSimulation());

            // Stop and copy results
            simManager.StopSimulation();

            EXPECT_LT(0, globState.globalHits.globalHits.nbDesorbed);
            EXPECT_LT(0, globState.globalHits.globalHits.nbMCHit);

            auto[diff_glob, diff_loc, diff_fine] = GlobalSimuState::Compare(oldState, globState, 0.009, 0.07);
            size_t runNb = 0;
            if ((diff_glob != 0 || diff_loc != 0)) {
                Log::console_msg(1, "[{}] Diff glob {} / loc {}\n", runNb, diff_glob, diff_loc);
                nCorrect = 0;
            } else if (diff_glob == 0 && diff_loc == 0) {
                nCorrect++;
                Log::console_msg(1, "[{}] Correct run #{}\n", runNb, nCorrect);
            }
        }

        size_t desPerTimestep = globState.globalHits.globalHits.nbDesorbed;
        for (size_t runNb = 1; runNb < nRuns; ++runNb) {
            // Modify argv with new duration
            /*auto newDur = std::ceil(Settings::simDuration + stepSizeTime);
            auto newDur_str = std::to_string((int)(newDur));
            argv[4] = newDur_str;*/

            Initializer::initTimeLimit(model, stepSizeTime * 3);
            size_t newDesLimit = globState.globalHits.globalHits.nbDesorbed + desPerTimestep;
            Settings::desLimit.clear();
            Settings::desLimit.emplace_back(newDesLimit);
            Initializer::initDesLimit(model, globState);

            EXPECT_NO_THROW(simManager.StartSimulation());

            // Stop and copy results
            simManager.StopSimulation();

            EXPECT_LT(0, globState.globalHits.globalHits.nbDesorbed);
            EXPECT_LT(0, globState.globalHits.globalHits.nbMCHit);

            auto[diff_glob, diff_loc, diff_fine] = GlobalSimuState::Compare(oldState, globState, 0.007, 0.06);
            if (runNb < nRuns - 1 && (diff_glob != 0 || diff_loc != 0)) {
                Log::console_msg(1, "[{}] Diff glob {} / loc {}\n", runNb, diff_glob, diff_loc);
                nCorrect = 0;
                continue; // try with more desorptions
            } else if (diff_glob == 0 && diff_loc == 0 && nCorrect < correctStreak - 1) {
                nCorrect++;
                Log::console_msg(1, "[{}] Correct run #{}\n", runNb, nCorrect);

                continue; // try with more desorptions
            } else if (diff_glob == 0 && diff_loc == 0 && nCorrect < correctStreak) {
                nCorrect++;
                Log::console_msg(1, "[{}] Correct run #{} -- Done\n", runNb, nCorrect);
            }

            EXPECT_EQ(0, diff_glob);
            EXPECT_EQ(0, diff_loc);

            if (diff_loc > 0)
                fmt::print(stderr, "[Warning] {} local differences found!\n", diff_loc);
            if (diff_fine > 0)
                fmt::print(stderr, "[Warning] {} differences on fine counters found!\n", diff_fine);
            break;
        }

        simManager.KillAllSimUnits();
        simManager.ResetSimulations();
    }

    TEST_P(ValidationFixture, ResultsWrong) {
        std::string testFile = GetParam();
        std::string outPath = "TPath_RW_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        Log::console_msg(1, "Filename: {}\n", testFile);
        size_t nbSuccess = 0;
        const size_t nRuns = 15;

        std::shared_ptr<SimulationManager> simManager = std::make_shared<SimulationManager>();
        simManager->interactiveMode = false;
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        std::vector<std::string> argv = {"tester", "--verbosity", "0", "-t", "120",
                                         "--file", testFile,
                                         "--outputPath", outPath};
        CharPVec argc_v(argv);
        char **args = argc_v.data();
        {
            if (-1 < Initializer::initFromArgv(argv.size(), (args), simManager.get(), model)) {
                exit(41);
            }
            if (Initializer::initFromFile(simManager.get(), model, &globState)) {
                exit(42);
            }
        }

        // Keep oldstate from file for compari
        GlobalSimuState oldState = globState;
        size_t oldHitsNb = globState.globalHits.globalHits.nbMCHit;
        size_t oldDesNb = globState.globalHits.globalHits.nbDesorbed;

        // First check for valid initial states
        // - old state with results, new state without
        // - after simulation, new state with results
        // Next, check for errors due to short run time
        // - this will prevent false positives for ResultsOkay tests
        for (size_t runNb = 0; runNb < nRuns; ++runNb) {
            if (runNb != 0) {
                // Reset simulation for a fresh start
                simManager = std::make_shared<SimulationManager>();
                model = std::make_shared<MolflowSimulationModel>();
                simManager->interactiveMode = false;
                if (-1 < Initializer::initFromArgv(argv.size(), (args), simManager.get(), model)) {
                    exit(41);
                }
                if (Initializer::initFromFile(simManager.get(), model, &globState)) {
                    exit(42);
                }
            }
            // clear old results from a previous attempt and define a new desorption limit (to prevent early termination as the input file will already have reached this limit)
            globState.Reset();
            Settings::desLimit.clear();
            Settings::desLimit.emplace_back(300);
            Initializer::initDesLimit(model, globState);

            //simManager.RefreshRNGSeed(false);
            simManager->ResetHits();
            simManager->InitSimulation(model, &globState);

            EXPECT_NE(0, oldDesNb);
            EXPECT_NE(0, oldHitsNb);
            EXPECT_EQ(0, globState.globalHits.globalHits.nbDesorbed);
            EXPECT_EQ(0, globState.globalHits.globalHits.nbMCHit);

            EXPECT_NO_THROW(simManager->StartSimulation());

            // Stop and copy results
            simManager->StopSimulation();

            EXPECT_LT(0, globState.globalHits.globalHits.nbDesorbed);
            EXPECT_LT(0, globState.globalHits.globalHits.nbMCHit);

            if (globState.globalHits.globalHits.nbMCHit == globState.globalHits.globalHits.nbDesorbed) {
                nbSuccess = nRuns;
                fmt::print(stderr,
                        "[{}][Warning] Results for this testcase are not comparable, due to equal amount of desorptions!\n",
                        runNb);
                fmt::print(stderr, "[{}][Warning] Results will only differ on finer counters, which demand more hits!\n",
                        runNb);
                break;
            }
            auto[diff_glob, diff_loc, diff_fine] = GlobalSimuState::Compare(oldState, globState, 0.006, 0.05);
            if (diff_glob || diff_loc)
                nbSuccess++;

            //simManager.KillAllSimUnits();
            //simManager.ResetSimulations();

            /*if(diff_glob > 0)
                EXPECT_NE(0, diff_glob);
            else
                EXPECT_NE(0, diff_loc);*/

            if (diff_glob <= 0)
                fmt::print(stderr, "[{}][Warning] No global differences found!\n", runNb);
            else if (diff_loc <= 0)
                fmt::print(stderr, "[{}][Warning] No local differences found!\n", runNb);
            else if (diff_fine <= 0)
                fmt::print(stderr, "[{}][Warning] No differences on fine counters found!\n", runNb);
        }
        if ((double) nbSuccess / nRuns < 0.66) {
            EXPECT_FALSE((double) nbSuccess / nRuns < 0.66);
            fmt::print(stderr, "[FAIL] Threshold for results of a low sample run was not crossed!\n"
                            "{} out of {} runs were correct!\n"
                            "This could be due to random nature of a MC simulation or a programmatic error leading to wrong conclusions.\n",
                    nRuns - nbSuccess, nRuns);
        }
        else {
            fmt::print("[SUCCESS] Necessary threshold for results of a low sample run was crossed!\n"
                               "{} out of {} runs were correct!\n",
                       nRuns - nbSuccess, nRuns);
        }
    }

    // Tests factorial of positive numbers.
    TEST(SubProcessInit, Zero) {

        {
            SimulationManager simMan(0);
            EXPECT_EQ(0, simMan.InitSimUnits());
        }

        {
            SimulationManager simMan(0);
            simMan.useCPU = true;
            simMan.nbThreads = 0;
            simMan.InitSimUnits();
            EXPECT_NE(0, simMan.nbThreads);
        }

        {
            SimulationManager simMan(0);
            simMan.useCPU = true;
            simMan.nbThreads = 1; // more not possible,
            simMan.InitSimUnits();
            EXPECT_EQ(1, simMan.nbThreads);
        }
    }

    TEST(SubProcessCreateAndKill, CPU) {
        {
            SimulationManager simMan(0);
            simMan.useCPU = true;
            simMan.nbThreads = 1;
            simMan.InitSimUnits();
            EXPECT_EQ(1, simMan.nbThreads);
            simMan.KillAllSimUnits();
            EXPECT_EQ(0, simMan.nbThreads);
        }
    }

    TEST(InputOutput, DefaultInput) {

        SimulationManager simManager{0};
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        std::vector<std::string> argv = {"tester", "-t", "1", "--reset", "--file", "TestCases/B01-lr1000_pipe.zip"};
        {
            CharPVec argc_v(argv);
            char **args = argc_v.data();
            Initializer::initFromArgv(argv.size(), (args), &simManager, model);
            Initializer::initFromFile(&simManager, model, &globState);
        }
        FlowIO::WriterXML writer;
        pugi::xml_document newDoc;
        std::string fullFileName = (std::filesystem::path(SettingsIO::outputPath) / SettingsIO::outputFile).string();
        EXPECT_FALSE(std::filesystem::exists(fullFileName));
        auto testPath1 = std::filesystem::path(SettingsIO::workPath) / "autosave_B01-lr1000_pipe.xml";
        auto testPath2 = std::filesystem::path(Initializer::getAutosaveFile());
        EXPECT_TRUE(testPath1.string() == testPath2.string());
        EXPECT_TRUE(Initializer::getAutosaveFile().find("autosave_B01-lr1000_pipe.xml") != std::string::npos);
        newDoc.load_file(fullFileName.c_str());
        writer.SaveGeometry(newDoc, model);
        writer.SaveSimulationState(fullFileName, model, globState);
        EXPECT_TRUE(SettingsIO::outputPath.find("Results_") != std::string::npos);
        EXPECT_TRUE(std::filesystem::exists(SettingsIO::outputPath));
        EXPECT_TRUE(std::filesystem::exists(fullFileName));
        EXPECT_TRUE(SettingsIO::outputFile == "out_B01-lr1000_pipe.zip");

        // CSV: check first whether files don't exist yet before generating and comparing them
        auto f_details = std::filesystem::path(SettingsIO::workPath).append("facet_details.csv");
        auto f_physics = std::filesystem::path(SettingsIO::workPath).append("facet_physics.csv");
        EXPECT_FALSE(std::filesystem::exists(f_details));
        EXPECT_FALSE(std::filesystem::exists(f_physics));
        FlowIO::Exporter::export_facet_details(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_details));
        FlowIO::Exporter::export_facet_quantities(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_physics));
        if(std::filesystem::exists(f_details)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_details.string()) );
        }
        if(std::filesystem::exists(f_physics)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_physics.string()) );
        }

        if (!SettingsIO::workPath.empty() && (SettingsIO::workPath != "." || SettingsIO::workPath != "./"))
            std::filesystem::remove_all(SettingsIO::workPath);
    }

    TEST(InputOutput, Outputpath) {

        SimulationManager simManager{0};
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        // generate hash name for tmp working file
        std::string outPath = "TPath_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        std::vector<std::string> argv = {"tester", "-t", "1", "--reset", "--file", "TestCases/B01-lr1000_pipe.zip",
                                         "--outputPath", outPath};
        {
            CharPVec argc_v(argv);
            char **args = argc_v.data();
            Initializer::initFromArgv(argv.size(), (args), &simManager, model);
            Initializer::initFromFile(&simManager, model, &globState);
        }
        FlowIO::WriterXML writer;
        pugi::xml_document newDoc;
        std::string fullFileName = (std::filesystem::path(SettingsIO::outputPath) / SettingsIO::outputFile).string();
        EXPECT_FALSE(std::filesystem::exists(fullFileName));
        auto testPath1 = std::filesystem::path(SettingsIO::workPath) / "autosave_B01-lr1000_pipe.xml";
        auto testPath2 = std::filesystem::path(Initializer::getAutosaveFile());
        EXPECT_TRUE(testPath1.string() == testPath2.string());
        EXPECT_TRUE(Initializer::getAutosaveFile().find("autosave_B01-lr1000_pipe.xml") != std::string::npos);
        newDoc.load_file(fullFileName.c_str());
        writer.SaveGeometry(newDoc, model);
        writer.SaveSimulationState(fullFileName, model, globState);
        EXPECT_FALSE(SettingsIO::outputPath.find("Results_") != std::string::npos);
        EXPECT_TRUE(SettingsIO::outputPath == outPath);
        EXPECT_TRUE(std::filesystem::exists(SettingsIO::outputPath));
        EXPECT_TRUE(std::filesystem::exists(fullFileName));
        EXPECT_TRUE(SettingsIO::outputFile == "out_B01-lr1000_pipe.zip");

// CSV: check first whether files don't exist yet before generating and comparing them
        auto f_details = std::filesystem::path(SettingsIO::workPath).append("facet_details.csv");
        auto f_physics = std::filesystem::path(SettingsIO::workPath).append("facet_physics.csv");
        EXPECT_FALSE(std::filesystem::exists(f_details));
        EXPECT_FALSE(std::filesystem::exists(f_physics));
        FlowIO::Exporter::export_facet_details(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_details));
        FlowIO::Exporter::export_facet_quantities(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_physics));
        if(std::filesystem::exists(f_details)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_details.string()) );
        }
        if(std::filesystem::exists(f_physics)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_physics.string()) );
        }

        if (!SettingsIO::workPath.empty() && (SettingsIO::workPath != "." || SettingsIO::workPath != "./"))
            std::filesystem::remove_all(SettingsIO::workPath);
    }

    TEST(InputOutput, OutputpathAndFile) {

        SimulationManager simManager{0};
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        std::string outPath = "TPath_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        std::string outFile = "tFile_" + std::to_string(std::hash<time_t>()(time(nullptr))) + ".xml";
        std::vector<std::string> argv = {"tester", "-t", "1", "--reset", "--file", "TestCases/B01-lr1000_pipe.zip",
                                         "--outputPath", outPath, "-o", outFile};
        {
            CharPVec argc_v(argv);
            char **args = argc_v.data();
            Initializer::initFromArgv(argv.size(), (args), &simManager, model);
            Initializer::initFromFile(&simManager, model, &globState);
        }

        FlowIO::WriterXML writer;
        pugi::xml_document newDoc;
        std::string fullFileName = (std::filesystem::path(SettingsIO::outputPath) / SettingsIO::outputFile).string();
        EXPECT_FALSE(std::filesystem::exists(fullFileName));
        auto testPath1 = std::filesystem::path(SettingsIO::workPath) / "autosave_B01-lr1000_pipe.xml";
        auto testPath2 = std::filesystem::path(Initializer::getAutosaveFile());
        EXPECT_TRUE(testPath1.string() == testPath2.string());
        EXPECT_TRUE(Initializer::getAutosaveFile().find("autosave_B01-lr1000_pipe.xml") != std::string::npos);
        newDoc.load_file(fullFileName.c_str());
        writer.SaveGeometry(newDoc, model);
        writer.SaveSimulationState(fullFileName, model, globState);
        EXPECT_FALSE(SettingsIO::outputPath.find("Results_") != std::string::npos);
        EXPECT_TRUE(SettingsIO::outputPath == outPath);
        EXPECT_TRUE(SettingsIO::outputFile == outFile);
        EXPECT_TRUE(std::filesystem::exists(SettingsIO::outputPath));
        EXPECT_TRUE(std::filesystem::exists(fullFileName));

        // CSV: check first whether files don't exist yet before generating and comparing them
        auto f_details = std::filesystem::path(SettingsIO::workPath).append("facet_details.csv");
        auto f_physics = std::filesystem::path(SettingsIO::workPath).append("facet_physics.csv");
        EXPECT_FALSE(std::filesystem::exists(f_details));
        EXPECT_FALSE(std::filesystem::exists(f_physics));
        FlowIO::Exporter::export_facet_details(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_details));
        FlowIO::Exporter::export_facet_quantities(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_physics));
        if(std::filesystem::exists(f_details)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_details.string()) );
        }
        if(std::filesystem::exists(f_physics)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_physics.string()) );
        }

        if (!SettingsIO::workPath.empty() && (SettingsIO::workPath != "." || SettingsIO::workPath != "./"))
            std::filesystem::remove_all(SettingsIO::workPath);
    }

    TEST(InputOutput, Outputfile) {

        SimulationManager simManager{0};
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        std::string outFile = "tFile_" + std::to_string(std::hash<time_t>()(time(nullptr))) + ".xml";
        std::vector<std::string> argv = {"tester", "-t", "1", "--reset", "--file", "TestCases/B01-lr1000_pipe.zip",
                                         "-o", outFile};
        {
            CharPVec argc_v(argv);
            char **args = argc_v.data();
            Initializer::initFromArgv(argv.size(), (args), &simManager, model);
            Initializer::initFromFile(&simManager, model, &globState);
        }
        FlowIO::WriterXML writer;
        pugi::xml_document newDoc;
        std::string fullFileName = (std::filesystem::path(SettingsIO::outputPath) / SettingsIO::outputFile).string();
        EXPECT_FALSE(std::filesystem::exists(fullFileName));
        auto testPath1 = std::filesystem::path(SettingsIO::workPath) / "autosave_B01-lr1000_pipe.xml";
        auto testPath2 = std::filesystem::path(Initializer::getAutosaveFile());
        EXPECT_TRUE(testPath1.string() == testPath2.string());
        EXPECT_TRUE(Initializer::getAutosaveFile().find("autosave_B01-lr1000_pipe.xml") != std::string::npos);
        newDoc.load_file(fullFileName.c_str());
        writer.SaveGeometry(newDoc, model);
        writer.SaveSimulationState(fullFileName, model, globState);
        EXPECT_TRUE(SettingsIO::outputPath.find("Results_") != std::string::npos);
        EXPECT_TRUE(SettingsIO::outputFile == outFile);
        EXPECT_TRUE(std::filesystem::exists(SettingsIO::outputPath));
        EXPECT_TRUE(std::filesystem::exists(fullFileName));

        // CSV: check first whether files don't exist yet before generating and comparing them
        auto f_details = std::filesystem::path(SettingsIO::workPath).append("facet_details.csv");
        auto f_physics = std::filesystem::path(SettingsIO::workPath).append("facet_physics.csv");
        EXPECT_FALSE(std::filesystem::exists(f_details));
        EXPECT_FALSE(std::filesystem::exists(f_physics));
        FlowIO::Exporter::export_facet_details(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_details));
        FlowIO::Exporter::export_facet_quantities(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_physics));
        if(std::filesystem::exists(f_details)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_details.string()) );
        }
        if(std::filesystem::exists(f_physics)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_physics.string()) );
        }

        // cleanup
        if (!SettingsIO::workPath.empty() && (SettingsIO::workPath != "." || SettingsIO::workPath != "./"))
            std::filesystem::remove_all(SettingsIO::workPath);
    }

    TEST(InputOutput, OutputfileWithPath) {

        SimulationManager simManager{0};
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        EXPECT_FALSE(SettingsIO::workPath.find("gtest_relpath") != std::string::npos);
        EXPECT_FALSE(std::filesystem::exists("gtest_relpath/gtest_out.xml"));

        std::string outPath = "TPath_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        std::string outFile = "tFile_" + std::to_string(std::hash<time_t>()(time(nullptr))) + ".xml";
        std::vector<std::string> argv = {"tester", "-t", "1", "--reset", "--file", "TestCases/B01-lr1000_pipe.zip",
                                         "-o", outPath + "/" + outFile};
        {
            CharPVec argc_v(argv);
            char **args = argc_v.data();
            Initializer::initFromArgv(argv.size(), (args), &simManager, model);
            Initializer::initFromFile(&simManager, model, &globState);
        }
        FlowIO::WriterXML writer;
        pugi::xml_document newDoc;
        std::string fullFileName = SettingsIO::outputFile;
        EXPECT_FALSE(std::filesystem::exists(fullFileName));
        auto testPath1 = std::filesystem::path(SettingsIO::workPath) / "autosave_B01-lr1000_pipe.xml";
        auto testPath2 = std::filesystem::path(Initializer::getAutosaveFile());
        EXPECT_TRUE(testPath1.string() == testPath2.string());
        EXPECT_TRUE(Initializer::getAutosaveFile().find("autosave_B01-lr1000_pipe.xml") != std::string::npos);
        EXPECT_TRUE(std::filesystem::exists(SettingsIO::workPath));
        EXPECT_TRUE(SettingsIO::outputPath.empty());
        newDoc.load_file(fullFileName.c_str());
        writer.SaveGeometry(newDoc, model);
        writer.SaveSimulationState(fullFileName, model, globState);
        EXPECT_TRUE(std::filesystem::exists(fullFileName));
        EXPECT_TRUE(SettingsIO::workPath.find(outPath) != std::string::npos);
        EXPECT_TRUE(SettingsIO::outputFile.find(outFile) != std::string::npos);

        // CSV: check first whether files don't exist yet before generating and comparing them
        auto f_details = std::filesystem::path(SettingsIO::workPath).append("facet_details.csv");
        auto f_physics = std::filesystem::path(SettingsIO::workPath).append("facet_physics.csv");
        EXPECT_FALSE(std::filesystem::exists(f_details));
        EXPECT_FALSE(std::filesystem::exists(f_physics));
        FlowIO::Exporter::export_facet_details(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_details));
        FlowIO::Exporter::export_facet_quantities(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_physics));
        if(std::filesystem::exists(f_details)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_details.string()) );
        }
        if(std::filesystem::exists(f_physics)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_physics.string()) );
        }

        if (!SettingsIO::workPath.empty() && (SettingsIO::workPath != "." || SettingsIO::workPath != "./"))
            std::filesystem::remove_all(SettingsIO::workPath);
    }

    TEST(InputOutput, OutputpathAndOutputfileWithPath) {

        SimulationManager simManager{0};
        std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
        GlobalSimuState globState{};

        EXPECT_FALSE(SettingsIO::workPath.find("gtest_relpath") != std::string::npos);
        EXPECT_FALSE(std::filesystem::exists("gtest_relpath/gtest_out.xml"));

        std::string outPath = "TPath_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        std::string outPathF = "TFPath_" + std::to_string(std::hash<time_t>()(time(nullptr)));
        std::string outFile = "tFile_" + std::to_string(std::hash<time_t>()(time(nullptr))) + ".xml";
        std::vector<std::string> argv = {"tester", "-t", "1", "--reset", "--file", "TestCases/B01-lr1000_pipe.zip",
                                         "--outputPath", outPath, "-o", outPathF + "/" + outFile};
        {
            CharPVec argc_v(argv);
            char **args = argc_v.data();
            Initializer::initFromArgv(argv.size(), (args), &simManager, model);
            Initializer::initFromFile(&simManager, model, &globState);
        }
        FlowIO::WriterXML writer;
        pugi::xml_document newDoc;
        std::string fullFileName = SettingsIO::workPath + "/" + SettingsIO::outputFile;
        EXPECT_FALSE(std::filesystem::exists(fullFileName));
        auto testPath1 = std::filesystem::path(SettingsIO::workPath) / "autosave_B01-lr1000_pipe.xml";
        auto testPath2 = std::filesystem::path(Initializer::getAutosaveFile());
        EXPECT_TRUE(testPath1.string() == testPath2.string());
        EXPECT_TRUE(Initializer::getAutosaveFile().find("autosave_B01-lr1000_pipe.xml") != std::string::npos);
        EXPECT_TRUE(std::filesystem::exists(SettingsIO::workPath));
        EXPECT_TRUE(SettingsIO::outputPath == outPath);
        EXPECT_TRUE(SettingsIO::workPath == outPath);
        newDoc.load_file(fullFileName.c_str());
        writer.SaveGeometry(newDoc, model);
        writer.SaveSimulationState(fullFileName, model, globState);
        EXPECT_TRUE(std::filesystem::exists(fullFileName));
        EXPECT_TRUE(SettingsIO::workPath.find(outPath) != std::string::npos);
        EXPECT_TRUE(SettingsIO::outputFile.find(outPathF) != std::string::npos);
        EXPECT_TRUE(SettingsIO::outputFile.find(outFile) != std::string::npos);

        // CSV: check first whether files don't exist yet before generating and comparing them
        auto f_details = std::filesystem::path(SettingsIO::workPath).append("facet_details.csv");
        auto f_physics = std::filesystem::path(SettingsIO::workPath).append("facet_physics.csv");
        EXPECT_FALSE(std::filesystem::exists(f_details));
        EXPECT_FALSE(std::filesystem::exists(f_physics));
        FlowIO::Exporter::export_facet_details(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_details));
        FlowIO::Exporter::export_facet_quantities(&globState, model.get());
        EXPECT_TRUE(std::filesystem::exists(f_physics));
        if(std::filesystem::exists(f_details)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_details.string()) );
        }
        if(std::filesystem::exists(f_physics)){
            EXPECT_LT(0 , FlowIO::CSVExporter::ValidateCSVFile(f_physics.string()) );
        }

        if (!SettingsIO::workPath.empty() && (SettingsIO::workPath != "." || SettingsIO::workPath != "./"))
            std::filesystem::remove_all(SettingsIO::workPath);
    }

    TEST(ParameterParsing, SweepFile) {

        // generate hash name for tmp working file
        std::string paramFile = std::to_string(std::hash<time_t>()(time(nullptr))) + ".cfg";
        std::ofstream outfile(paramFile);

        outfile << "facet.42.opacity=0.5\n"
                   "facet.3.sticking=10.01\n"
                   "facet.50-90.outgassing=42e5\n"
                   "facet.100.temperature=290.92\n"
                   "simulation.mass=42.42\n"
                   "simulation.enableDecay=1\n"
                   "simulation.halfLife=42.42";
        outfile.close();
        ParameterParser::ParseFile(paramFile, std::vector<SelectionGroup>());

        WorkerParams wp;
        ASSERT_FALSE(std::abs(wp.gasMass - 42.42) < 1e-5);
        ASSERT_FALSE(wp.enableDecay);
        ASSERT_FALSE(std::abs(wp.halfLife - 42.42) < 1e-5);
        ParameterParser::ChangeSimuParams(wp);
        ASSERT_TRUE(std::abs(wp.gasMass - 42.42) < 1e-5);
        ASSERT_TRUE(wp.enableDecay);
        ASSERT_TRUE(std::abs(wp.halfLife - 42.42) < 1e-5);


        std::vector<std::shared_ptr<SimulationFacet>> facets(200);
        for (int i = 0; i < facets.size(); ++i) facets[i] = std::make_shared<MolflowSimFacet>();
        ASSERT_FALSE(std::abs(facets[41]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[2]->sh.sticking - 10.01) < 1e-5);
        ASSERT_FALSE(std::abs(facets[49]->sh.outgassing - 42e5) < 1e-5); // first
        ASSERT_FALSE(std::abs(facets[69]->sh.outgassing - 42e5) < 1e-5); // mid
        ASSERT_FALSE(std::abs(facets[89]->sh.outgassing - 42e5) < 1e-5); // last
        ASSERT_FALSE(std::abs(facets[99]->sh.temperature - 290.92) < 1e-5);
        ParameterParser::ChangeFacetParams(facets);
        ASSERT_DOUBLE_EQ(facets[41]->sh.opacity, 0.5);
        ASSERT_DOUBLE_EQ(facets[2]->sh.sticking, 10.01);
        ASSERT_DOUBLE_EQ(facets[49]->sh.outgassing, 42e5); // first
        ASSERT_DOUBLE_EQ(facets[69]->sh.outgassing, 42e5); // mid
        ASSERT_DOUBLE_EQ(facets[89]->sh.outgassing, 42e5); // last
        ASSERT_DOUBLE_EQ(facets[99]->sh.temperature, 290.92);

        std::filesystem::remove(paramFile);
    }

    TEST(ParameterParsing, SweepVec) {

        // generate hash name for tmp working file
        std::vector<std::string> params;

        params.emplace_back("facet.42.opacity=0.5");
        params.emplace_back("facet.3.sticking=10.01");
        params.emplace_back("facet.50-90.outgassing=42e5");
        params.emplace_back("facet.100.temperature=290.92");
        params.emplace_back("simulation.mass=42.42");
        ParameterParser::ParseInput(params, std::vector<SelectionGroup>());

        WorkerParams wp;
        ASSERT_FALSE(std::abs(wp.gasMass - 42.42) < 1e-5);
        ParameterParser::ChangeSimuParams(wp);
        ASSERT_TRUE(std::abs(wp.gasMass - 42.42) < 1e-5);


        std::vector<std::shared_ptr<SimulationFacet>> facets(200);
        for (int i = 0; i < facets.size(); ++i) facets[i] = std::make_shared<MolflowSimFacet>();
        ASSERT_FALSE(std::abs(facets[41]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[2]->sh.sticking - 10.01) < 1e-5);
        ASSERT_FALSE(std::abs(facets[49]->sh.outgassing - 42e5) < 1e-5); // first
        ASSERT_FALSE(std::abs(facets[69]->sh.outgassing - 42e5) < 1e-5); // mid
        ASSERT_FALSE(std::abs(facets[89]->sh.outgassing - 42e5) < 1e-5); // last
        ASSERT_FALSE(std::abs(facets[99]->sh.temperature - 290.92) < 1e-5);
        ParameterParser::ChangeFacetParams(facets);
        ASSERT_DOUBLE_EQ(facets[41]->sh.opacity, 0.5);
        ASSERT_DOUBLE_EQ(facets[2]->sh.sticking, 10.01);
        ASSERT_DOUBLE_EQ(facets[49]->sh.outgassing, 42e5); // first
        ASSERT_DOUBLE_EQ(facets[69]->sh.outgassing, 42e5); // mid
        ASSERT_DOUBLE_EQ(facets[89]->sh.outgassing, 42e5); // last
        ASSERT_DOUBLE_EQ(facets[99]->sh.temperature, 290.92);
    }

    TEST(ParameterParsing, Group) {

        // generate hash name for tmp working file
        std::string paramFile = std::to_string(std::hash<time_t>()(time(nullptr))) + ".cfg";
        std::ofstream outfile(paramFile);

        std::vector<SelectionGroup> selections;
        SelectionGroup group;
        group.name = "ValidSelection";
        group.selection = {5, 8, 9};
        selections.push_back(group);
        group.name = "InvalidSelection";
        group.selection = {1, 2, 3, 4};
        selections.push_back(group);
        group.name = "NotValid";
        group.selection = {10, 11, 12};
        selections.push_back(group);
        outfile << "facet.\"NotValid\".opacity=0.3\n"
                   "facet.\"ValidSelection\".opacity=0.5\n"
                   "facet.\"InvalidSelection\".opacity=0.8\n";
        outfile.close();
        ParameterParser::ParseFile(paramFile, selections);

        std::vector<std::shared_ptr<SimulationFacet>> facets(200);
        for (int i = 0; i < facets.size(); ++i) facets[i] = std::make_shared<MolflowSimFacet>();
        ASSERT_FALSE(std::abs(facets[4]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[5]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[6]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[7]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[8]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[9]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[10]->sh.opacity - 0.5) < 1e-5);
        ParameterParser::ChangeFacetParams(facets);
        ASSERT_FALSE(std::abs(facets[4]->sh.opacity - 0.5) < 1e-5);
        ASSERT_DOUBLE_EQ(facets[5]->sh.opacity, 0.5);
        ASSERT_FALSE(std::abs(facets[6]->sh.opacity - 0.5) < 1e-5);
        ASSERT_FALSE(std::abs(facets[7]->sh.opacity - 0.5) < 1e-5);
        ASSERT_DOUBLE_EQ(facets[8]->sh.opacity, 0.5);
        ASSERT_DOUBLE_EQ(facets[9]->sh.opacity, 0.5);
        ASSERT_FALSE(std::abs(facets[10]->sh.opacity - 0.5) < 1e-5);

        std::filesystem::remove(paramFile);
    }
}  // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);


    return RUN_ALL_TESTS();
}