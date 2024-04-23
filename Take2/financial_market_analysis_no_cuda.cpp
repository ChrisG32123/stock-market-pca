#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <sys/time.h>

void get_walltime_(double* wcTime) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (double)(tp.tv_sec + tp.tv_usec / 1000000.0);
}

void get_walltime(double* wcTime) {
    get_walltime_(wcTime);
}

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
        price += distribution(generator);
    }
    return price;
}

void run_simulation(std::vector<Stock>& stocks, int days, int simulations) {
    int rank, size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int per_process = simulations / size;
    std::vector<double> results(per_process);
    double start, finish;

    get_walltime(&start); // Start timing before the parallel section

    #pragma omp parallel for
    for (int i = 0; i < per_process; ++i) {
        double portfolio_value = 0;
        for (const auto& stock : stocks) {
            double final_price = simulate_stock_price(stock.mean_return, stock.std_dev, 100, days);
            portfolio_value += final_price * stock.shares;
        }
        results[i] = portfolio_value;
    }

    get_walltime(&finish); // End timing after the parallel section

    // Collect results in rank 0 and compute execution time
    double local_execution_time = finish - start;
    double global_execution_time;
    MPI_Reduce(&local_execution_time, &global_execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Execution Time: " << global_execution_time << " seconds." << std::endl;
    }

    MPI_Finalize();
}

int main() {
    std::vector<Stock> stocks = {{0.0005, 0.01, 100}, {0.0006, 0.012, 150}};
    run_simulation(stocks, 365, 10000);
    return 0;
}
