// main.cpp
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include "MonteCarloSimulation.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Initialize simulation parameters
    SimulationParameters params = parseParameters(argc, argv);

    if (world_rank == 0) {
        // Only master node handles this part
        std::cout << "Starting Monte Carlo Simulation for Portfolio Risk Analysis" << std::endl;
    }

    // Distributed computation with MPI
    std::vector<double> partial_results = runDistributedMonteCarlo(params, world_rank, world_size);

    // Gather results at the master node
    if (world_rank == 0) {
        std::vector<double> final_results = gatherResults(partial_results, world_size);
        // Output results to CSV for analysis
        outputResultsToCSV(final_results, "simulation_results.csv");
    }

    MPI_Finalize();
    return 0;
}

/*
#!/bin/bash
#SBATCH --job-name=monte-carlo-simulation
#SBATCH --output=monte-carlo-simulation-%j.out
#SBATCH --error=monte-carlo-simulation-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --gres=gpu:2

module load gcc/9.2.0  # Load necessary modules, e.g., GCC, MPI, CUDA
module load mpi/openmpi/4.0.3
module load cuda/10.2

srun ./MonteCarloSimulation

*/