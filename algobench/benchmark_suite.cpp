//
// Created by pascal on 8/8/19.
//

#include "../src_shared/SimulationManager.h"
#include "../src/Initializer.h"
#include "../src/ParameterParser.h"

#include <filesystem>
#include <fstream>

// hash time to create random file name
#include <ctime>
#include <Helper/Chronometer.h>
#include <memory>
#include <cmath>
#include <IO/CSVExporter.h>
#include <SettingsIO.h>
#include <fmt/core.h>
#include <CLI11/CLI11.hpp>

// helper class for flexible argument initialization
class CharPVec {
public:
    CharPVec() = default;

    [[maybe_unused]] explicit CharPVec(size_t size) : vec(size, nullptr) {};

    explicit CharPVec(const std::vector<std::string> &str_vec) : vec(str_vec.size(), nullptr) {
        cpy(str_vec);
    };

    ~CharPVec() { clear(); };

    char **data() { return vec.data(); }

    void cpy(const std::vector<std::string> &str_vec) {
        if (vec.size() != str_vec.size()) vec.resize(str_vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = new char[str_vec[i].size() + 1];
            std::snprintf(vec[i], std::size(str_vec[i]) + 1, "%s", str_vec[i].c_str());
        }
    }

    void clear() {
        for (auto &c: vec) delete[] c;
    }

    // member
    std::vector<char *> vec;
};


enum class BenchAlgo {
    ALGO_BVH_SAH,
    ALGO_BVH_Prob,
    ALGO_BVH_RDH,
    ALGO_KD_SAH,
    ALGO_KD_SAH_ROPE,
    ALGO_KD_SAH_ROPERESTART,
    ALGO_KD_Prob,
    ALGO_KD_Prob_ROPE,
    ALGO_KD_Prob_ROPERESTART,
    ALGO_KD_Hybrid,
    ALGO_KD_Hybrid_ROPE,
    ALGO_KD_Hybrid_ROPERESTART,
    ALGO_KD_HybridBin,
    ALGO_KD_HybridBin_ROPE,
    ALGO_KD_HybridBin_ROPERESTART
};

void SetAlgo(BenchAlgo current_algo, std::shared_ptr<MolflowSimulationModel> &model, double hybrid_weight = 1.0);

int main(int argc, char **argv) {

    //CLI args
    double hybrid_weight{1.0};  // = 3.14;
    std::string test_case_dir = "./AlgoCases";
    std::string test_case_file;
    {
        CLI::App app("Molflow ADS Algorithm benchmark");
        // add version output
        //app.set_version_flag("--version", std::string(CLI11_VERSION));
        /*std::string file;
        CLI::Option *opt = app.add_option("-f,--file,file", file, "File name");

        int count{0};
        CLI::Option *copt = app.add_option("-c,--count", count, "Counter");

        int v{0};
        CLI::Option *flag = app.add_flag("--flag", v, "Some flag that can be passed multiple times");
        */
        app.add_option("-w,--weight", hybrid_weight, "Hybrid split weight (w*split_main+(1-w)*split_bak)");
        app.add_option("-d,--dir", test_case_dir, "Test case directory (default: ./AlgoCases)");
        app.add_option("-f,--file", test_case_file, "Test case file (default: none)");

        CLI11_PARSE(app, argc, argv);
    }
    fmt::print("Hybrid weight for : {}\n", hybrid_weight);

    if(!test_case_file.empty()){
        test_case_dir = std::filesystem::path(test_case_file).parent_path().string();
    }
    if(!test_case_file.empty() && !std::filesystem::exists(std::filesystem::path(test_case_file))){
        fmt::print(stderr, "Test case file not found : {}\n", test_case_file);
        exit(43);
    }
    if(!std::filesystem::exists(test_case_dir)){
        fmt::print(stderr, "Test case folder not found : {}\n", test_case_dir);
        exit(42);
    }
    fmt::print("using Test case folder : {}\n", test_case_dir);

    static std::unordered_map<BenchAlgo, std::string> const tableDetail = {
            {BenchAlgo::ALGO_BVH_SAH,               "BVHxSAH"},
            {BenchAlgo::ALGO_BVH_Prob,              "BVHxProb"},
            {BenchAlgo::ALGO_BVH_RDH,               "BVHxRDH"},
            {BenchAlgo::ALGO_KD_SAH,                "KDxSAH"},
            {BenchAlgo::ALGO_KD_SAH_ROPE,           "KDxSAHxRope"},
            {BenchAlgo::ALGO_KD_SAH_ROPERESTART,    "KDxSAHxRopeRestart"},
            {BenchAlgo::ALGO_KD_Prob,               "KDxProb"},
            {BenchAlgo::ALGO_KD_Prob_ROPE,          "KDxProbxRope"},
            {BenchAlgo::ALGO_KD_Prob_ROPERESTART,   "KDxProbxRopeRestart"},
            {BenchAlgo::ALGO_KD_Hybrid,             "KDxHybrid"},
            {BenchAlgo::ALGO_KD_Hybrid_ROPE,        "KDxHybridxRope"},
            {BenchAlgo::ALGO_KD_Hybrid_ROPERESTART, "KDxHybridxRopeRestart"},
            {BenchAlgo::ALGO_KD_HybridBin,          "KDxHybridBin"},
            {BenchAlgo::ALGO_KD_HybridBin_ROPE,        "KDxHybridBinxRope"},
            {BenchAlgo::ALGO_KD_HybridBin_ROPERESTART, "KDxHybridBinxRopeRestart"}
    };

    std::string outPath = "AlgoPath_" + std::to_string(std::hash<time_t>()(time(nullptr)));

    size_t nbFails = 0;
    bool fastEnough = false;
    const size_t nRuns = 5;
    const size_t keepNEntries = 20;
    const size_t runForTSec = 60;

    std::map<std::string, std::vector<std::pair<int, double>>> perfTimes;
    std::filesystem::create_directory(outPath);
    std::string timeRecFile = std::filesystem::path(outPath).append("time_bench.txt").string();
    //std::string timeRecFile = std::filesystem::path("time_bench.txt").string();
    fmt::print("Writing to {}\n", timeRecFile);
    std::ofstream ofs(timeRecFile);
    ofs << fmt::format("[filename] algorithm time\n");
    ofs.close();

    for (auto const &dir_entry: std::filesystem::directory_iterator{test_case_dir}) {
        if (!(dir_entry.path().extension().string() == ".zip" || dir_entry.path().extension().string() == ".xml")) {
            continue;
        }
        if(!test_case_file.empty()){
            if(std::filesystem::path(test_case_file).filename().string() != std::filesystem::path(dir_entry).filename().string()){
                continue;
            }
        }

        std::string testFile = dir_entry.path().string();
        fmt::print("Filename: {}\n", testFile.c_str());
        perfTimes.emplace(testFile, std::vector<std::pair<int, double>>());
        std::vector<BenchAlgo> run_algos {
                BenchAlgo::ALGO_BVH_SAH,
                BenchAlgo::ALGO_KD_SAH,
                BenchAlgo::ALGO_KD_SAH_ROPE,
                BenchAlgo::ALGO_KD_SAH_ROPERESTART,
                BenchAlgo::ALGO_BVH_Prob,
                BenchAlgo::ALGO_KD_Prob,
                BenchAlgo::ALGO_KD_Prob_ROPE,
                BenchAlgo::ALGO_KD_Prob_ROPERESTART,
                BenchAlgo::ALGO_BVH_RDH,
                BenchAlgo::ALGO_KD_Hybrid,
                BenchAlgo::ALGO_KD_Hybrid_ROPE,
                BenchAlgo::ALGO_KD_Hybrid_ROPERESTART,
                BenchAlgo::ALGO_KD_HybridBin,
                BenchAlgo::ALGO_KD_HybridBin_ROPE,
                BenchAlgo::ALGO_KD_HybridBin_ROPERESTART
        };

        bool benchmark_with_hits = false;
        for (auto current_algo: run_algos) {
            std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();
            SetAlgo(current_algo, model, hybrid_weight);

            if (model->wp.accel_type == 1) {
                if ((model->wp.splitMethod == static_cast<int>(KdTreeAccel::SplitMethod::TestSplit))
                    || (model->wp.splitMethod == static_cast<int>(KdTreeAccel::SplitMethod::HybridSplit))
                    || (model->wp.splitMethod == static_cast<int>(KdTreeAccel::SplitMethod::HybridBin))) {
                    benchmark_with_hits = true;
                    continue;
                }
            } else if (model->wp.splitMethod == static_cast<int>(BVHAccel::SplitMethod::RDH)) {
                benchmark_with_hits = true;
                continue;
            }
        }

        // Get hit statistics in one go for one geometry and all algorithms
        std::vector<TestRay> hits;
        GlobalSimuState globState_old{};
        auto hits_old = globState_old.hitBattery;

        if(benchmark_with_hits){
            std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();

            SimulationManager simManager;
            //std::shared_ptr<SimulationModel> model = std::make_shared<SimulationModel>();

            {
                std::vector<std::string> arg_vec = {"algobench", "-t", fmt::format("{}", runForTSec),
                                                    "--file", testFile,
                                                    "--outputPath", outPath};

                CharPVec argc_v(arg_vec);
                char **args = argc_v.data();
                Initializer::initFromArgv(static_cast<int>(arg_vec.size()), (args), &simManager, model);
                Initializer::initFromFile(&simManager, model, &globState_old);
            }

            size_t oldDesNb = globState_old.globalHits.globalHits.nbDesorbed;
            double oldHitNb = globState_old.globalHits.globalHits.nbHitEquiv;

            model->otfParams.raySampling = true;
            globState_old.hitBattery.maxSamples = 1024 * 128;
            SetAlgo(BenchAlgo::ALGO_BVH_SAH, model);
            model->BuildAccelStructure(&globState_old, BVH, 0, 2, hybrid_weight);

            simManager.StartSimulation();
            ProcessSleep(1000);
            simManager.StopSimulation();
            hits = globState_old.PrepareHitBattery();
            fmt::print("Hits before {}\n", hits.size());
            simManager.StartSimulation();
            ProcessSleep(20000);
            simManager.StopSimulation();
            hits = globState_old.PrepareHitBattery();

            fmt::print("Hits after {}\n", hits.size());
            fmt::print("Hits after bak {}\n", globState_old.PrepareHitBattery().size());
            oldDesNb = globState_old.globalHits.globalHits.nbDesorbed;
            oldHitNb = globState_old.globalHits.globalHits.nbHitEquiv;
        }
        for (auto current_algo: run_algos) {
            std::shared_ptr<MolflowSimulationModel> model = std::make_shared<MolflowSimulationModel>();

            SetAlgo(current_algo, model, hybrid_weight);

            SimulationManager simManager;
            //std::shared_ptr<SimulationModel> model = std::make_shared<SimulationModel>();
            GlobalSimuState globState{};

            {
                std::vector<std::string> arg_vec = {"algobench", "-t", fmt::format("{}", runForTSec),
                                                    "--file", testFile,
                                                    "--outputPath", outPath};

                CharPVec argc_v(arg_vec);
                char **args = argc_v.data();
                Initializer::initFromArgv(static_cast<int>(arg_vec.size()), (args), &simManager, model);
                Initializer::initFromFile(&simManager, model, &globState);
            }

            size_t oldDesNb = globState.globalHits.globalHits.nbDesorbed;
            double oldHitNb = globState.globalHits.globalHits.nbHitEquiv;

            SetAlgo(current_algo, model, hybrid_weight);
            model->otfParams.raySampling = false;
            globState.hitBattery.maxSamples = 0;

            try {
                simManager.StartSimulation();
                simManager.StopSimulation();
                oldDesNb = globState.globalHits.globalHits.nbDesorbed;
                oldHitNb = globState.globalHits.globalHits.nbHitEquiv;
                if(benchmark_with_hits)
                    model->BuildAccelStructure(&globState_old, model->wp.accel_type, model->wp.splitMethod, 2, hybrid_weight);
                else
                    model->BuildAccelStructure(&globState, model->wp.accel_type, model->wp.splitMethod, 2, hybrid_weight);
                simManager.StartSimulation();
                fmt::print("--- COMMENCE: BENCHMARK {} ---\n", tableDetail.at((BenchAlgo)current_algo));
            }
            catch (...){
                perfTimes[testFile].emplace_back(std::make_pair((int) current_algo, 0.0));
                std::string out_str = fmt::format("\"{}\" {} \"{}\" {:.4e}\n", testFile,
                                                  perfTimes[testFile].back().first,
                                                  tableDetail.at((BenchAlgo) perfTimes[testFile].back().first),
                                                  perfTimes[testFile].back().second);
                fmt::print(out_str);

                ofs.open(timeRecFile, std::ios_base::app);
                ofs << out_str;
                ofs.close();
                continue;
            }
            Chronometer simTimer(false);
            simTimer.Start();
            double elapsedTime;

            bool endCondition = false;
            do {
                ProcessSleep(1000);
                elapsedTime = simTimer.Elapsed();
                if (model->otfParams.desorptionLimit != 0)
                    endCondition =
                            globState.globalHits.globalHits.nbDesorbed >= model->otfParams.desorptionLimit;
                // Check for potential time end
                if (Settings::simDuration > 0) {
                    endCondition |= elapsedTime >= static_cast<double>(Settings::simDuration);
                }
            } while (!endCondition);
            simTimer.Stop();

            // Stop and copy results
            try {
                simManager.StopSimulation();
                simManager.KillAllSimUnits();
            }
            catch (...) {
                ProcessSleep(10000);
                try { simManager.KillAllSimUnits(); }
                catch (...) {
                    perfTimes[testFile].emplace_back(std::make_pair((int) current_algo, 0.0));
                    std::string out_str = fmt::format("\"{}\" {} \"{}\" {:.4e}\n", testFile,
                                                      perfTimes[testFile].back().first,
                                                      tableDetail.at((BenchAlgo) perfTimes[testFile].back().first),
                                                      perfTimes[testFile].back().second);
                    fmt::print(out_str);

                    ofs.open(timeRecFile, std::ios_base::app);
                    ofs << out_str;
                    ofs.close();
                    continue;
                }
            }

            perfTimes[testFile].emplace_back(std::make_pair((int) current_algo, (double) (
                    globState.globalHits.globalHits.nbHitEquiv - oldHitNb) / (elapsedTime)));
            std::string out_str = fmt::format("\"{}\" {} \"{}\" {:.4e}\n", testFile,
                                              perfTimes[testFile].back().first,
                                              tableDetail.at((BenchAlgo) perfTimes[testFile].back().first),
                                              perfTimes[testFile].back().second);
            fmt::print(out_str);

            ofs.open(timeRecFile, std::ios_base::app);
            ofs << out_str;
            ofs.close();
        }
    };

    // Cleanup
    SettingsIO::cleanup_files();

    return 0;
}

void SetAlgo(BenchAlgo current_algo, std::shared_ptr<MolflowSimulationModel> &model, const double hybrid_weight) {
    switch (current_algo) {
        case (BenchAlgo::ALGO_BVH_SAH) : {
            model->wp.accel_type = BVH;
            model->wp.splitMethod = static_cast<int>(BVHAccel::SplitMethod::SAH);
            break;
        }
        case (BenchAlgo::ALGO_KD_SAH) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::SAH);
            model->wp.kd_with_ropes = false;
            break;
        }
        case (BenchAlgo::ALGO_KD_SAH_ROPE) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::SAH);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = false;
            break;
        }
        case (BenchAlgo::ALGO_KD_SAH_ROPERESTART) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::SAH);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = true;
            break;
        }
        case (BenchAlgo::ALGO_BVH_Prob) : {
            model->wp.accel_type = BVH;
            model->wp.splitMethod = static_cast<int>(BVHAccel::SplitMethod::ProbSplit);
            break;
        }
        case (BenchAlgo::ALGO_KD_Prob) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::ProbSplit);
            model->wp.kd_with_ropes = false;
            break;
        }
        case (BenchAlgo::ALGO_KD_Prob_ROPE) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::ProbSplit);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = false;
            break;
        }
        case (BenchAlgo::ALGO_KD_Prob_ROPERESTART) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::ProbSplit);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = true;
            break;
        }
        case (BenchAlgo::ALGO_BVH_RDH) : {
            model->wp.accel_type = BVH;
            model->wp.splitMethod = static_cast<int>(BVHAccel::SplitMethod::RDH);
            model->wp.hybridWeight = hybrid_weight;
            break;
        }
        case (BenchAlgo::ALGO_KD_Hybrid) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::HybridSplit);
            model->wp.kd_with_ropes = false;
            model->wp.hybridWeight = hybrid_weight;
            break;
        }
        case (BenchAlgo::ALGO_KD_Hybrid_ROPE) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::HybridSplit);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = false;
            model->wp.hybridWeight = hybrid_weight;
            break;
        }
        case (BenchAlgo::ALGO_KD_Hybrid_ROPERESTART) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::HybridSplit);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = true;
            model->wp.hybridWeight = hybrid_weight;
            break;
        }
        case (BenchAlgo::ALGO_KD_HybridBin) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::HybridBin);
            model->wp.kd_with_ropes = false;
            model->wp.hybridWeight = hybrid_weight;
            break;
        }
        case (BenchAlgo::ALGO_KD_HybridBin_ROPE) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::HybridBin);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = false;
            model->wp.hybridWeight = hybrid_weight;
            break;
        }
        case (BenchAlgo::ALGO_KD_HybridBin_ROPERESTART) : {
            model->wp.accel_type = KD;
            model->wp.splitMethod = static_cast<int>(KdTreeAccel::SplitMethod::HybridBin);
            model->wp.kd_with_ropes = true;
            model->wp.kd_restart_ropes = true;
            model->wp.hybridWeight = hybrid_weight;
	        break;
        }
        default : {
            fmt::print("Unavailable algorithm {}\n", static_cast<int>(current_algo));
            exit(0);
        }
    }
}
