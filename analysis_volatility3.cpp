#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>

double calculateSD(const std::vector<double>& data) {
    double sum = 0.0, mean, standardDeviation = 0.0;
    int i;
    for(i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    mean = sum/data.size();
    for(i = 0; i < data.size(); ++i) {
        standardDeviation += pow(data[i] - mean, 2);
    }
    return sqrt(standardDeviation / data.size());
}

// Assumes 'data' is a vector of doubles, each representing a data point in the dataset.
std::vector<double> calculateVolatility(std::vector<double>& data, int windowSize) {
    std::vector<double> volatilityData;
    #pragma omp parallel for
    for (int i = 0; i <= data.size() - windowSize; i++) {
        std::vector<double> window(data.begin() + i, data.begin() + i + windowSize);
        double sd = calculateSD(window);
        #pragma omp critical
        volatilityData.push_back(sd);
    }
    return volatilityData;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<double> data;
    std::vector<double> localData;
    std::vector<double> globalVolatility;
    int windowSize = atoi(argv[3]); // Assuming window size is passed as the third argument

    // Only the root process will read the data
    if (world_rank == 0) {
        // Load your data into the 'data' vector
        std::ifstream inputFile(argv[1]);
        std::string line;
        while (getline(inputFile, line)) {
            std::stringstream stream(line);
            double price;
            while (stream >> price) {
                data.push_back(price);
            }
        }
        inputFile.close();
    }

    // Distribute the data size
    int dataSize = data.size();
    MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute the data
    localData.resize(dataSize / world_size);
    MPI_Scatter(data.data(), dataSize / world_size, MPI_DOUBLE, localData.data(), dataSize / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate local volatility
    std::vector<double> localVolatility = calculateVolatility(localData, windowSize);

    // Gather the global volatility data at the root process
    if (world_rank == 0) {
        globalVolatility.resize(dataSize - windowSize + 1); // Adjust based on your calculation method
    }
    MPI_Gather(localVolatility.data(), localVolatility.size(), MPI_DOUBLE, globalVolatility.data(), localVolatility.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root process writes the results
    if (world_rank == 0) {
        std::ofstream outputFile(argv[2]);
        for (double vol : globalVolatility) {
            outputFile << vol << std::endl;
        }
        outputFile.close();
    }

    MPI_Finalize();
    return 0;
}


/*

// MPI Initialization
MPI_Init(&argc, &argv);
int world_size, world_rank;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

// Data Distribution with MPI (root process reads and scatters data)

// Local Processing with OpenMP
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < local_data_size; ++i) {
    // Calculate volatility for each data point in local chunk
}

// Gather Results with MPI (gather results to root process)

// MPI Finalization
MPI_Finalize();

*/

/*
#!/bin/bash
#SBATCH --job-name=volatility_analysis
#SBATCH --output=volatility_analysis_%j.out
#SBATCH --error=volatility_analysis_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=01:00:00

module load mpi/openmpi-x.y.z
module load gcc/x.y.z

# Assuming the environment is already set up for MPI and OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Replace with the actual paths to your input and output files and the window size
srun ./analysis_volatility input_data.csv output_results.csv 10
*/
