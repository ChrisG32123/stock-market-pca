// MonteCarloSimulation.h
#pragma once
#include <vector>

struct SimulationParameters {
    int numPaths;
    int numSteps;
    double riskFreeRate;
    double volatility;
    double initialPrice;
};

std::vector<double> runMonteCarloSimulation(const SimulationParameters& params);

// MonteCarloSimulation.cpp
#include "MonteCarloSimulation.h"
#include <cmath>
#include <vector>
#include <random>
#include <omp.h>

std::vector<double> runMonteCarloSimulation(const SimulationParameters& params) {
    std::vector<double> endPrices(params.numPaths);
    double dt = 1.0 / params.numSteps;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    #pragma omp parallel for
    for (int i = 0; i < params.numPaths; ++i) {
        double price = params.initialPrice;
        for (int j = 0; j < params.numSteps; ++j) {
            price += params.riskFreeRate * price * dt + params.volatility * price * sqrt(dt) * d(gen);
        }
        endPrices[i] = price;
    }
    return endPrices;
}

// main.cpp
#include "MonteCarloSimulation.h"
#include <iostream>
#include <fstream>

int main() {
    SimulationParameters params = {10000, 100, 0.05, 0.2, 100.0};
    auto results = runMonteCarloSimulation(params);

    std::ofstream outFile("simulation_results.csv");
    outFile << "EndPrice\n";
    for (auto price : results) {
        outFile << price << "\n";
    }
    outFile.close();

    std::cout << "Simulation completed and results are saved to simulation_results.csv" << std::endl;
    return 0;
}

/*
#!/bin/bash
#SBATCH --job-name=MonteCarloSim
#SBATCH --output=montecarlo_%j.out
#SBATCH --error=montecarlo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=standard

module load gcc/9
module load openmpi/4.0.3

mpicxx -o MonteCarloSim main.cpp MonteCarloSimulation.cpp -fopenmp
srun ./MonteCarloSim

*/