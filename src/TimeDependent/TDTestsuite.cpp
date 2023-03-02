//
// Created by Pascal Baehr on 28.04.20.
//

#include <iterator>
#include <random>
#include <vector>

#include "../../include/CLI11/CLI11.hpp"
#include "../../src_shared/Helper/Chronometer.h"
#include "../TimeMoments.h"
#include "TDAlgorithms.h"
#include "fmt/core.h"
#include "TDMoments.h"
#include <fmt/color.h>

using namespace MFTD;

const int totalRuns = 5u;
std::vector<std::vector<double>> benchmark_times;

// typedef std::tuple<double,double,double,double> MomentInterval; // begin,
// interval, end, timewindow
namespace DefaultVar {
    constexpr size_t nTimes = 1.0e7;
    constexpr double fromTime = 0.0;
    constexpr double toTime = 100.0;
    constexpr double deltaMin = 0.001;
    constexpr double deltaMax = 10.0;
    constexpr size_t seed = 0; // 0 for random
} // namespace DefaultVar

namespace Settings {
    std::string inputFile;
    std::string outputFile;
    static std::vector<Moment> intervals;
    static std::vector<MomentInterval> uIntervals;
    static std::vector<double> time_points;
    static std::vector<std::vector<double>>
            time_points_wbreak; // use when utilizing startIndex
    static std::vector<size_t> breakPoints;
} // namespace Settings

void generatePoT(size_t nTimes, double fromTime, double toTime, double deltaMin,
                 double deltaMax, std::vector<double> &numbers,
                 size_t seed = 0) {

    if (seed == 0) {
        std::random_device
                rd; // Will be used to obtain a seed for the random number engine
        seed = rd();
    }
    std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(deltaMin, deltaMax);

    numbers.clear();
    numbers.reserve(nTimes);
    double num = fromTime;
    for (size_t n = 0; n < nTimes; ++n) {
        num += dis(gen);
        numbers.push_back(num);

        if (num >= toTime)
            num = fromTime;
    }

    fmt::print("Generated {} numbers\n", numbers.size());
}

void generatePoTW(size_t nTimes, double fromTime, double toTime,
                  double deltaMin, double deltaMax,
                  std::vector<std::vector<double>> &numbers, size_t seed = 0) {

    if (seed == 0) {
        std::random_device
                rd; // Will be used to obtain a seed for the random number engine
        seed = rd();
    }
    std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(deltaMin, deltaMax);

    numbers.clear();
    double num = fromTime;
    numbers.emplace_back();
    for (size_t n = 0; n < nTimes; ++n) {
        num += dis(gen);
        numbers.back().push_back(num);

        if (num >= toTime) {
            numbers.emplace_back();
            num = fromTime;
        }
    }

    size_t totalNumbers = 0;
    for (auto &numb: numbers) {
        totalNumbers += numb.size();
    }
    fmt::print("Generated {} numbers with {} desorptions\n", totalNumbers,
               numbers.size());
}

void generatePoTW2(size_t nTimes, double fromTime, double toTime,
                   double deltaMin, double deltaMax,
                   std::vector<double> &numbers,
                   std::vector<size_t> &resetPoints, size_t seed = 0) {

    if (seed == 0) {
        std::random_device
                rd; // Will be used to obtain a seed for the random number engine
        seed = rd();
    }
    std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(deltaMin, deltaMax);

    numbers.clear();
    double num = fromTime;
    for (size_t n = 0; n < nTimes; ++n) {
        num += dis(gen);
        numbers.push_back(num);

        if (num >= toTime) {
            resetPoints.push_back(numbers.size());
            num = fromTime;
        }
    }

    fmt::print("Generated {} numbers with {} desorptions\n", numbers.size(),
               resetPoints.size() + 1);
}

void parsePoT(const std::string &fileName, std::vector<double> &numbers) {
    std::ifstream timeStream(fileName);
    std::istream_iterator<double> start(timeStream), end;
    numbers.assign(start, end);
    fmt::print("Read {} numbers\n", numbers.size());
}

void parsePoTWStop(const std::string &fileName,
                   std::vector<std::vector<double>> &numbers) {
    std::ifstream timeStream(fileName);
    std::string line;
    while (std::getline(timeStream, line)) {
        if (line.empty())
            continue;
        numbers.emplace_back();
        std::istringstream iss(line);
        double val = 0.0;
        while (iss >> val) {
            numbers.back().push_back(val);
        }
    }

    size_t totalNumbers = 0;
    for (auto &numb: numbers) {
        totalNumbers += numb.size();
    }
    fmt::print("Read {} numbers\n", totalNumbers);
}

void parsePoTWStop2(const std::string &fileName, std::vector<double> &numbers,
                    std::vector<size_t> &resetPoints) {
    std::ifstream timeStream(fileName);
    std::string line;
    while (std::getline(timeStream, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        double val = 0.0;
        while (iss >> val) {
            numbers.push_back(val);
        }
        resetPoints.push_back(numbers.size());
    }

    fmt::print("Read {} numbers with {} desorptions\n", numbers.size(),
               resetPoints.size());
}

int parseCommands(int argc, char **argv) {
    CLI::App app{"Time dependent algorithm test suite"};

    app.add_option("-f,--file", Settings::inputFile,
                   "Required input file (XML only)")
                    /*->required()*/
            ->check(CLI::ExistingFile);
    app.add_option("-o,--output", Settings::outputFile,
                   "Output file if different from input file");
    CLI11_PARSE(app, argc, argv);

    // Save to inputfile
    if (Settings::outputFile.empty())
        Settings::outputFile = "out.txt";

    return 0;
}



void printTimeBins(const std::vector<size_t> &timeBins) {
    if (0) {
        size_t i = 0;
        for (auto &bin: timeBins) {
            if (bin > 0) {
                fmt::print("{}<>{} ", i, bin);
                if ((i % 10) == 9)
                    fmt::print("\n");
                i++;
            }
            if (i >= 20)
                break;
        }
        fmt::print("\n");
    }
}

void initTimeBins(std::vector<size_t> &timeBins, size_t n) {
    std::vector<size_t>(n, 0).swap(timeBins);
}

void diffTimeBins(const std::vector<size_t> &tb_l,
                  const std::vector<size_t> &tb_r) {
    size_t diff_c = 0;

    for (size_t i = 0; i < tb_l.size(); i++) {
        if (tb_l[i] != tb_r[i]) {
            diff_c++;
            if (diff_c <= 10)
                fmt::print(fg(fmt::color::orange), "[{}] {}<>{} ", i, tb_l[i], tb_r[i]);
        }
    }
    if (diff_c > 0)
        fmt::print("\n");
    if (diff_c > 0)
        fmt::print(fg(fmt::color::red), "Total diff: {}\n", diff_c);
}

size_t getBinCount(const std::vector<size_t> &bins) {
    size_t count = 0;

    for (size_t i = 0; i < bins.size(); i++) {
        count += bins[i];
    }

    return count;
}

double runAlgorithm(const std::string& alg_name , int (*alg)(double, const std::vector<Moment> &), const std::vector<Moment> &intervalMoments,
                  std::vector<size_t> &timeBins, size_t totalRuns = 1) {
    // 1. Vector binary search
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Binary search \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        for (auto moment: Settings::time_points) {
            int ind = alg(moment, intervalMoments);
            if (ind >= 0)
                ++timeBins[ind];
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Binary search\n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();

        printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorBinarySearch(const std::vector<Moment> &intervalMoments,
                      std::vector<size_t> &timeBins) {
    // 1. Vector binary search
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Binary search \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        for (auto moment: Settings::time_points) {
            int ind = LookupMomentIndex(moment, intervalMoments);
            if (ind >= 0)
                ++timeBins[ind];
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Binary search\n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();

        printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorQuadSearch(const std::vector<Moment> &intervalMoments,
                    std::vector<size_t> &timeBins) {
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Quad search\n", time.ElapsedMs());
        for (auto moment: Settings::time_points) {
            int ind = quadraticSearch(moment, intervalMoments);
            if (ind >= 0)
                ++timeBins[ind];
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Quad search\n", time.ElapsedMs());

        
avg_time += time.ElapsedMs();printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorInterpSearch(const std::vector<Moment> &intervalMoments,
                      std::vector<size_t> &timeBins) {
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Interp search\n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        for (auto moment: Settings::time_points) {
            int ind = interpolationSearch(moment, intervalMoments);
            if (ind >= 0)
                ++timeBins[ind];
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Interp search\n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();

        printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorInterpSearch_indexed(const std::vector<Moment> &intervalMoments,
                      std::vector<size_t> &timeBins) {
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Interp search -- Index \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        size_t lastIndex = 0;
        for (const auto &particleTrace: Settings::time_points_wbreak) {
            for (const auto moment: particleTrace) {
                int ind = interpolationSearch(moment, intervalMoments, lastIndex);
                if (ind >= 0) {
                    ++timeBins[ind];
                    lastIndex = ind;
                }
            }
            lastIndex = 0;
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Interp search -- Index \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();

        printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorJumpSearch(const std::vector<Moment> &intervalMoments,
                    std::vector<size_t> &timeBins) {
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Jump search\n", time.ElapsedMs());
        int n_err = 0;
        for (auto moment: Settings::time_points) {
            //moment = 99.977320172147103;
            int ind = jumpSearchProg(intervalMoments, moment, intervalMoments.size());
            /*int ind_ref = LookupMomentIndex(moment, intervalMoments);
            if (ind >= (int) timeBins.size()) {
                fmt::print("XJump search error: {} >= {} for {} (should be {})\n", ind,
                           timeBins.size(), moment, ind_ref);
            } else if (n_err < 10 && ind != ind_ref && (ind >= 0 || ind_ref >= 0)) {
                n_err++;
                fmt::print("Jump search error: {} != {} (/{}) for {} (should be in: [{} , {}])\n",
                           ind, ind_ref, timeBins.size(), moment, intervalMoments[ind_ref].first, intervalMoments[ind_ref].second);

            }
            else */if (ind >= 0)
                ++timeBins[ind];
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Jump search \n", time.ElapsedMs());

        
avg_time += time.ElapsedMs();
printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorCalcSearch(const std::vector<Moment> &intervalMoments,
                    std::vector<size_t> &timeBins) {
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Calc search \n", time.ElapsedMs());
        size_t u = 0;
        int n_err = 0;
        for (auto moment: Settings::time_points) {
            int ind = calcSearch(moment, intervalMoments, Settings::uIntervals);
            //int ind_ref = LookupMomentIndex(moment, intervalMoments);
            /*if (ind >= (int) timeBins.size()) {
                fmt::print("XCalc search error: {} >= {} for {} (should be {})\n", ind,
                           timeBins.size(), moment, ind_ref);
            } else if (n_err < 10 && ind != ind_ref && (ind >= 0 || ind_ref >= 0)) {
                n_err++;
                fmt::print("Calc search error: {} != {} (/{}) for {} (should be in: [{} , {}])\n",
                           ind, ind_ref, timeBins.size(), moment, intervalMoments[ind_ref].first, intervalMoments[ind_ref].second);

            }
            else*/ if (ind >= 0) {
                ++timeBins[ind];
            }
            ++u;
            // if(u >= 10000) exit(0);
        }

        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Calc search \n", time.ElapsedMs());
        avg_time += time.ElapsedMs();

        printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorBinarySearch_indexed(const std::vector<Moment> &intervalMoments,
                              std::vector<size_t> &timeBins) {
    // 1. Vector binary search
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Binary search -- Index \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        size_t lastIndex = 0;
        for (const auto &particleTrace: Settings::time_points_wbreak) {
            for (const auto moment: particleTrace) {
                int ind = LookupMomentIndex(moment, intervalMoments, lastIndex);
                if (ind >= 0) {
                    ++timeBins[ind];
                    lastIndex = ind;
                }
            }
            lastIndex = 0;
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Binary search -- Index \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorJumpSearch_indexed(const std::vector<Moment> &intervalMoments,
                            std::vector<size_t> &timeBins) {
    // 1. Vector binary search
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Jump search -- Index \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        size_t lastIndex = 0;
        for (const auto &particleTrace: Settings::time_points_wbreak) {
            for (const auto moment: particleTrace) {
                int ind = jumpSearchProg(intervalMoments, moment, intervalMoments.size(), lastIndex);
                if (ind >= 0) {
                    ++timeBins[ind];
                    lastIndex = ind;
                }
            }
            lastIndex = 0;
        }
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Jump search -- Index \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        printTimeBins(timeBins);
    }

    return avg_time / totalRuns;
}

double runVectorBinarySearch_indexed_v2(const std::vector<Moment> &intervalMoments,
                                 std::vector<size_t> &timeBins) {
    // 1. Vector binary search
    Chronometer time;
    double avg_time = 0.0;
    for (int runNb = 0; runNb < totalRuns; ++runNb) {
        time.ReInit();
        time.Start();
        initTimeBins(timeBins, intervalMoments.size());
        fmt::print(fg(fmt::color::light_golden_rod_yellow), " [START][{:8.2f}ms] Vector -- Binary search -- Index2 \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        size_t lastIndex = 0;
        size_t desNb = 0;
        const size_t total_events = Settings::time_points.size();
        size_t breakAt = Settings::breakPoints[0];
        for (size_t eventNb = 0; eventNb < total_events; ++eventNb) {
            if (eventNb >= breakAt) {
                lastIndex = 0;
                ++desNb;
                breakAt = Settings::breakPoints[desNb];
            }

            int ind = LookupMomentIndex(Settings::time_points[eventNb],
                                        intervalMoments, lastIndex);
            if (ind >= 0) {
                ++timeBins[ind];
                lastIndex = ind;
            }
        }
        lastIndex = 0;
        time.Stop();
        fmt::print(fg(fmt::color::light_salmon), " [ END ][{:8.2f}ms] Vector -- Binary search -- Index2 \n",
                   time.ElapsedMs());
avg_time += time.ElapsedMs();
        printTimeBins(timeBins);

#if defined(DEBUG)
        size_t max = 0;
        size_t max_pos = 0;

        std::sort(timeBins.begin(), timeBins.end());
        double probSumTopQuart = 0.0;
        for (int j = timeBins.size() * 0.75; j < timeBins.size(); ++j) {
            probSumTopQuart += (double) timeBins[j] / DefaultVar::nTimes;
        }
        double probSum = 0.0;
        size_t sum = 0;
        for (size_t j = timeBins.size() * 0.0; j < timeBins.size(); ++j) {
            probSum += (double) timeBins[j] / DefaultVar::nTimes;
            sum += timeBins[j];
        }
        fmt::print("Results for nBins: {} <- {}\n", DefaultVar::nTimes,
                   intervalMoments.size());

        fmt::print("Max_____: {} / {}\n", timeBins.back(),
                   (double) timeBins.back() / DefaultVar::nTimes);
        fmt::print("Med_0.75: {} / {}\n",
                   timeBins[static_cast<size_t>(timeBins.size() * 0.75)],
                   (double) timeBins[static_cast<size_t>(timeBins.size() * 0.75)] /
                   DefaultVar::nTimes);
        fmt::print("Med_0.50: {} / {}\n",
                   timeBins[static_cast<size_t>(timeBins.size() * 0.50)],
                   (double) timeBins[static_cast<size_t>(timeBins.size() * 0.50)] /
                   DefaultVar::nTimes);
        fmt::print("Med_0.25: {} / {}\n",
                   timeBins[static_cast<size_t>(timeBins.size() * 0.25)],
                   (double) timeBins[static_cast<size_t>(timeBins.size() * 0.25)] /
                   DefaultVar::nTimes);
        fmt::print("Top_Prob: {}\n", probSumTopQuart);
        fmt::print("All_Prob: {} / {}\n", sum, probSum);
#endif
    }

    return avg_time / totalRuns;
}

int main(int argc, char **argv) {
    parseCommands(argc, argv);

    std::vector<std::vector<UserMoment>> testCases_um;
    std::vector<UserMoment> uMoments;

    if (Settings::inputFile.empty()) {
        uMoments = std::vector<UserMoment>{{"0.001,0.1,100.0", 0.1}};
        testCases_um.emplace_back(uMoments);
        uMoments = std::vector<UserMoment>{{"0.001,0.001,100.0", 0.001}};
        testCases_um.emplace_back(uMoments);
        uMoments = std::vector<UserMoment>{{"0.001,0.1,100.0", 1e-7}};
        testCases_um.emplace_back(uMoments);
        uMoments = std::vector<UserMoment>{{"0.001,0.001,100.0", 1e-7}};
        testCases_um.emplace_back(uMoments);
        uMoments = std::vector<UserMoment>{
                {"0.001,0.001,1.0",   0.001},
                {"1.01,0.001,10.0",   0.001},
                {"10.01,0.001,20.0",  0.001},
                {"20.01,0.001,30.0",  0.001},
                {"30.01,0.001,40.0",  0.001},
                {"40.01,0.001,50.0",  0.001},
                {"50.01,0.001,60.0",  0.001},
                {"60.01,0.001,70.0",  0.001},
                {"70.01,0.001,80.0",  0.001},
                {"80.01,0.001,90.0",  0.001},
                {"90.01,0.001,100.0", 0.001}};
        testCases_um.emplace_back(uMoments);
        uMoments = std::vector<UserMoment>{
                {"0.001, 0.01,1.0",  0.01},
                {"1.1, 0.01,10.0",   0.01},
                {"10.1, 0.01,20.0",  0.01},
                {"20.1, 0.01,30.0",  0.01},
                {"30.1, 0.01,40.0",  0.01},
                {"40.1, 0.01,50.0",  0.01},
                {"50.1, 0.01,60.0",  0.01},
                {"60.1, 0.01,70.0",  0.01},
                {"70.1, 0.01,80.0",  0.01},
                {"80.1, 0.01,90.0",  0.01},
                {"90.1, 0.01,100.0", 0.01}};
        testCases_um.emplace_back(uMoments);
        uMoments = std::vector<UserMoment>{
                {"0.001, 0.1,1.0",  0.01},
                {"1.1, 0.1,10.0",   0.01},
                {"10.1, 0.1,20.0",  0.01},
                {"20.1, 0.1,30.0",  0.01},
                {"30.1, 0.1,40.0",  0.01},
                {"40.1, 0.1,50.0",  0.01},
                {"50.1, 0.1,60.0",  0.01},
                {"60.1, 0.1,70.0",  0.01},
                {"70.1, 0.1,80.0",  0.01},
                {"80.1, 0.1,90.0",  0.01},
                {"90.1, 0.1,100.0", 0.01}};
        testCases_um.emplace_back(uMoments);
        uMoments = std::vector<UserMoment>{{"0.001,0.01,100.0", 0.01}};
        testCases_um.emplace_back(uMoments);
        // Case X:
        // Y moments, many misses
        // Hit ratio ~= 0.21
        if (1) {
            uMoments = std::vector<UserMoment>{
                    {"0.001, 0.1,1.0",     0.01},
                    {"1.1, 0.001,10.0",    0.001},
                    {"10.1, 0.1,20.0",     0.00001},
                    {"20.1, 0.1,30.0",     0.0001},
                    {"30.1, 0.1,40.0",     0.001},
                    {"40.1, 0.00001,50.0", 0.00001},
                    {"50.1, 0.05,60.0",    0.001},
                    {"60.1, 0.1,70.0",     0.01},
                    {"70.1, 0.7,80.0",     0.07},
                    {"80.1, 0.1,90.0",     0.01},
                    {"90.1, 0.09,100.0",   0.001}};
            testCases_um.emplace_back(uMoments);
        }
        // Case X:
        // Y moments, no spacing in between
        // Hit ratio ~= 1.0
        if (0) {
            uMoments = std::vector<UserMoment>{{"0.005,0.01,109.995", 0.01}};
            testCases_um.emplace_back(uMoments);
        }

        // Case X:
        // Y moments, no spacing in between
        // Hit ratio ~= 1.0
        if (0) {
            uMoments = std::vector<UserMoment>{{"0.05,0.1,109.95", 0.1}};
            testCases_um.emplace_back(uMoments);
        }

        if (1) {
            uMoments = std::vector<UserMoment>{
                    {"0.001, 0.2,1.0",     0.01},
                    {"1.7, 1.0,10.0",    0.01},
                    {"10.7, 1.0,20.0",     0.01}};
            testCases_um.emplace_back(uMoments);
        }
    }
    else{
        // Test cases for loaded input file
        uMoments = std::vector<UserMoment>{
                {"1.0e-6, 1.0e-6,9.1e-6",     1.0e-6},
                {"1.0e-5, 1.0e-6,9.1e-5",     1.0e-6},
                {"1.0e-4, 1.0e-5,9.1e-4",     1.0e-6},
                {"1.0e-3, 1.0e-4,9.1e-3",     1.0e-6}
        };
        testCases_um.emplace_back(uMoments);

        uMoments = std::vector<UserMoment>{
                {"1.0e-6, 1.0e-9, 9.9e-6",     1.0e-9},
                {"1.0e-5, 1.0e-8, 9.9e-5",     1.0e-8},
                {"1.0e-4, 1.0e-7, 9.9e-4",     1.0e-7},
                {"1.0e-3, 1.0e-6, 9.9e-3",     1.0e-6}
        };
        testCases_um.emplace_back(uMoments);

        uMoments = std::vector<UserMoment>{
                {"1.0e-6, 1.0e-11, 9.9e-6",    1.0e-11},
                {"1.0e-5, 1.0e-10, 9.9e-5",    1.0e-10},
                {"1.0e-4, 1.0e-9, 9.9e-4",     1.0e-9},
                {"1.0e-3, 1.0e-8, 9.9e-3",     1.0e-8}
        };
        testCases_um.emplace_back(uMoments);

        uMoments = std::vector<UserMoment>{
                {"1.0e-6, 1.0e-11, 9.9e-6",    1.0e-13},
                {"1.0e-5, 1.0e-10, 9.9e-5",    1.0e-12},
                {"1.0e-4, 1.0e-9, 9.9e-4",     1.0e-11},
                {"1.0e-3, 1.0e-8, 9.9e-3",     1.0e-10}
        };
        testCases_um.emplace_back(uMoments);

        uMoments = std::vector<UserMoment>{
                {"1.0e-6, 1.0e-8, 9.9e-3",    1.0e-8}
        };
        testCases_um.emplace_back(uMoments);

        uMoments = std::vector<UserMoment>{
                {"1.0e-6, 1.0e-8, 9.9e-3",    1.0e-11}
        };
        testCases_um.emplace_back(uMoments);
    }
    /*std::vector<UserMoment> uMoments{
            {"1.0e-13,1.0e-5,0.001", 1.0e-5}
    };*/
    /*std::vector<UserMoment> uMoments{
            {"0.1", 0.5},
            {"1,1,60", 0.5},
            {"70,10,3600", 0.5},
            {"3660,60,30000", 0.5}
    };*/
    /*std::vector<UserMoment> uMoments{
            {"0.001,0.0001,0.08", 0.0001},
            {"0.1", 0.01},
            {"1,0.001,60", 0.001},
            {"70,0.01,3600", 0.01},
            {"3660,0.06,30000", 0.06}
    };*/

    benchmark_times.resize(testCases_um.size());
    int tc = 1;
    for(auto& test_case : testCases_um) {
        fmt::print(fg(fmt::color::alice_blue), "Starting test #{}\n", tc);
        //for (auto &um_test: uMoments) {
        momentsReader(test_case, Settings::intervals);
        momentIntervalReader(test_case, Settings::uIntervals);

        std::vector<Moment> intervalMoments;
        parseMoments(Settings::intervals, intervalMoments);
        if (intervalMoments.empty())
            exit(0);
        size_t totalIntervals = intervalMoments.size();
        std::vector<size_t> timeBins(intervalMoments.size(), 0);

        // exit(0);
        if (!Settings::inputFile.empty()) {
            parsePoT(Settings::inputFile, Settings::time_points);
        } else {
            generatePoT(DefaultVar::nTimes, DefaultVar::fromTime, DefaultVar::toTime,
                        DefaultVar::deltaMin, DefaultVar::deltaMax,
                        Settings::time_points, DefaultVar::seed);
        }

        // 1. Vector binary search
        benchmark_times[tc-1].emplace_back(runVectorBinarySearch(intervalMoments, timeBins));
        std::vector<size_t> timeBins_test1 = timeBins;
        fmt::print("Test case has around: {} / {} bin hits\n", getBinCount(timeBins_test1),  Settings::time_points.size());
        //diffTimeBins(timeBins_test1, timeBins);
        benchmark_times[tc-1].emplace_back(runVectorQuadSearch(intervalMoments, timeBins));
        diffTimeBins(timeBins_test1, timeBins);
        benchmark_times[tc-1].emplace_back(runVectorInterpSearch(intervalMoments, timeBins));
        diffTimeBins(timeBins_test1, timeBins);
        benchmark_times[tc-1].emplace_back(runVectorJumpSearch(intervalMoments, timeBins));
        diffTimeBins(timeBins_test1, timeBins);
        benchmark_times[tc-1].emplace_back(runVectorCalcSearch(intervalMoments, timeBins));
        diffTimeBins(timeBins_test1, timeBins);
        // Free some memory
        Settings::time_points.clear();
        Settings::time_points_wbreak.clear();

        if (!Settings::inputFile.empty()) {
            parsePoTWStop(Settings::inputFile, Settings::time_points_wbreak);
        } else {
            generatePoTW(DefaultVar::nTimes, DefaultVar::fromTime, DefaultVar::toTime,
                         DefaultVar::deltaMin, DefaultVar::deltaMax,
                         Settings::time_points_wbreak, DefaultVar::seed);
        }

        benchmark_times[tc-1].emplace_back(runVectorBinarySearch_indexed(intervalMoments, timeBins));
        std::vector<size_t> timeBins_test6 = timeBins;
        //diffTimeBins(timeBins_test1, timeBins_test6);
        benchmark_times[tc-1].emplace_back(runVectorJumpSearch_indexed(intervalMoments, timeBins));
        diffTimeBins(timeBins_test6, timeBins);
        benchmark_times[tc-1].emplace_back(runVectorInterpSearch_indexed(intervalMoments, timeBins));
        diffTimeBins(timeBins_test6, timeBins);

        // Free some memory
        Settings::time_points_wbreak.clear();
        // initTimeBins(timeBins, intervalMoments.size());
        if (!Settings::inputFile.empty()) {
            parsePoTWStop2(Settings::inputFile, Settings::time_points,
                           Settings::breakPoints);
        } else {
            generatePoTW2(DefaultVar::nTimes, DefaultVar::fromTime,
                          DefaultVar::toTime, DefaultVar::deltaMin,
                          DefaultVar::deltaMax, Settings::time_points,
                          Settings::breakPoints, DefaultVar::seed);
        }
        //fmt::print("Parsed 2\n");

#if defined(DEBUG)
        for(int i = 0; i<20; ++i)
            fmt::print("Break {} : {} : {:e}\n",i,Settings::breakPoints[i],
           Settings::time_points[i]);
#endif
        benchmark_times[tc-1].emplace_back(runVectorBinarySearch_indexed_v2(intervalMoments, timeBins));

        Settings::time_points.clear();
        Settings::breakPoints.clear();
        fmt::print(fg(fmt::color::light_golden_rod_yellow), "Finished test #{}\n", tc);
        tc++;
    }

    for(int i = 0; i < benchmark_times.size(); i++) { // test case
        fmt::print(fg(fmt::color::light_golden_rod_yellow), "TC #{} | ", i);
        for(int j = 0; j < benchmark_times[i].size(); j++) { // algo
            fmt::print("{:03.2f}ms | ", benchmark_times[i][j]);
        }
        fmt::print("\n");

    }


    for(int j = 0; j < benchmark_times[0].size(); j++) { // algofmt::print(fg(fmt::color::light_golden_rod_yellow), "TC #{} | ", i);
        fmt::print(fg(fmt::color::indian_red), "Alg. #{} | ", j);
        for(int i = 0; i < benchmark_times.size(); i++) { // algo
            fmt::print("{:03.2f}ms | ", benchmark_times[i][j]);
        }
        fmt::print("\n");
    }

    return 0;
}