#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <curand_kernel.h>

// Error checking wrapper
inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

struct Stock {
    double mean_return;
    double std_dev;
    int shares;
};

__global__ void simulate_stock_prices(double* means, double* std_devs, double* results, int days, int num_stocks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_stocks) {
        curandState state;
        curand_init(1234, idx, 0, &state);
        double price = 100.0;  // Assuming starting price is 100
        for (int day = 0; day < days; ++day) {
            price += curand_normal_double(&state) * std_devs[idx] + means[idx];
        }
        results[idx] = price * means[idx];  // Simplified calculation
    }
}

void run_simulation(std::vector<Stock>& stocks, int days) {
    int rank, size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_stocks_per_proc = stocks.size() / size;
    int start_idx = rank * num_stocks_per_proc;
    int end_idx = start_idx + num_stocks_per_proc;

    std::vector<double> means(num_stocks_per_proc), std_devs(num_stocks_per_proc);
    for (int i = start_idx; i < end_idx; ++i) {
        means[i - start_idx] = stocks[i].mean_return;
        std_devs[i - start_idx] = stocks[i].std_dev;
    }

    double *d_means, *d_std_devs, *d_results;
    checkCuda(cudaMalloc(&d_means, num_stocks_per_proc * sizeof(double)));
    checkCuda(cudaMalloc(&d_std_devs, num_stocks_per_proc * sizeof(double)));
    checkCuda(cudaMalloc(&d_results, num_stocks_per_proc * sizeof(double)));

    checkCuda(cudaMemcpy(d_means, means.data(), num_stocks_per_proc * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_std_devs, std_devs.data(), num_stocks_per_proc * sizeof(double), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_stocks_per_proc + threadsPerBlock - 1) / threadsPerBlock;
    simulate_stock_prices<<<blocksPerGrid, threadsPerBlock>>>(d_means, d_std_devs, d_results, days, num_stocks_per_proc);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());

    std::vector<double> results(num_stocks_per_proc);
    checkCuda(cudaMemcpy(results.data(), d_results, num_stocks_per_proc * sizeof(double), cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(d_means));
    checkCuda(cudaFree(d_std_devs));
    checkCuda(cudaFree(d_results));

    for (int i = 0; i < num_stocks_per_proc; ++i) {
        std::cout << "Processed stock " << i + start_idx << " final price estimate: " << results[i] << std::endl;
    }

    MPI_Finalize();
}

int main() {
    std::vector<Stock> stocks = {{0.0005, 0.01, 100}, {0.0006, 0.012, 150}, /* more stocks */};
    run_simulation(stocks, 365);  // Run the simulation for one year
    return 0;
}
