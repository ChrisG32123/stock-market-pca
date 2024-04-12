#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>

struct Stock {
    double mean_return;
    double std_dev;
    int shares;
};

double simulate_stock_price(double mean, double std_dev, double initial_price, int days) {
    std::default_random_engine generator(omp_get_thread_num());
    std::normal_distribution<double> distribution(mean, std_dev);
    double price = initial_price;
    for (int i = 0; i < days; ++i) {
        price += distribution(generator);  // Simulating daily returns
    }
    return price;
}

void calculate_statistics(const std::vector<double>& results) {
    const size_t num_results = results.size();
    std::vector<double> sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end());

    // Calculate VaR at 95%
    size_t var_index = static_cast<size_t>(num_results * 0.95);
    double var_95 = sorted_results[var_index];

    // Calculate CVaR at 95%
    double cvar_95 = std::accumulate(sorted_results.begin(), sorted_results.begin() + var_index, 0.0) / var_index;

    std::cout << "VaR (95%): " << var_95 << std::endl;
    std::cout << "CVaR (95%): " << cvar_95 << std::endl;
}

void run_simulation(std::vector<Stock>& stocks, int days, int simulations) {
    int rank, size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int per_process = simulations / size;
    std::vector<double> results(per_process);

    #pragma omp parallel for
    for (int i = 0; i < per_process; ++i) {
        double portfolio_value = 0;
        for (const auto& stock : stocks) {
            double final_price = simulate_stock_price(stock.mean_return, stock.std_dev, 100, days);
            portfolio_value += final_price * stock.shares;
        }
        results[i] = portfolio_value;
    }

    // Gather all results here in rank 0
    std::vector<double> all_results;
    if (rank == 0) {
        all_results.resize(simulations);
    }
    MPI_Gather(results.data(), per_process, MPI_DOUBLE, all_results.data(), per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Perform statistical calculations and print
        calculate_statistics(all_results);
        // Optionally, save results to file for further processing in Python
        std::ofstream outfile("simulation_results.csv");
        for (const auto& value : all_results) {
            outfile << value << std::endl;
        }
        outfile.close();
    }

    MPI_Finalize();
}

int main() {
    std::vector<Stock> stocks = {{0.0005, 0.01, 100}, {0.0006, 0.012, 150}};
    run_simulation(stocks, 365, 10000);
    return 0;
}

/*

__global__ void simulate_stock_prices_cuda(double* mean, double* std_dev, double* prices, int days, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(1234, idx, 0, &state);
        double price = prices[idx];
        for (int day = 0; day < days; ++day) {
            double increment = curand_normal_double(&state) * std_dev[idx] + mean[idx];
            price += increment;
        }
        prices[idx] = price;
    }
}

// Host function to setup and launch the CUDA kernel
void run_cuda_simulation(double* h_mean, double* h_std_dev, double* h_prices, int days, int n) {
    double *d_mean, *d_std_dev, *d_prices;
    cudaMalloc(&d_mean, n * sizeof(double));
    cudaMalloc(&d_std_dev, n * sizeof(double));
    cudaMalloc(&d_prices, n * sizeof(double));
    
    cudaMemcpy(d_mean, h_mean, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_dev, h_std_dev, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prices, h_prices, n * sizeof(double), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    simulate_stock_prices_cuda<<<numBlocks, blockSize>>>(d_mean, d_std_dev, d_prices, days, n);
    
    cudaMemcpy(h_prices, d_prices, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_mean);
    cudaFree(d_std_dev);
    cudaFree(d_prices);
}

#include <mpi.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaSetDevice(rank % nDevices);  // Simple modulo to assign devices

    // Proceed with the simulation setup including calling CUDA functions
    MPI_Finalize();
    return 0;
}

*/