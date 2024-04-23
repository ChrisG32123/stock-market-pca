#include <vector>
#include <iostream>
#include <random>

struct Stock {
    double mean_return;
    double std_dev;
    int shares;
};

// Simulate stock prices using the CPU
void simulate_stock_prices_cpu(const std::vector<Stock>& stocks, int days, std::vector<double>& results) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    for (size_t i = 0; i < stocks.size(); ++i) {
        double price = 100.0;  // Assuming starting price is 100
        for (int day = 0; day < days; ++day) {
            distribution = std::normal_distribution<double>(stocks[i].mean_return, stocks[i].std_dev);
            price += distribution(generator);
        }
        results[i] = price * stocks[i].shares;
    }
}

void run_simulation_cpu(const std::vector<Stock>& stocks, int days) {
    std::vector<double> results(stocks.size());
    simulate_stock_prices_cpu(stocks, days, results);

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "CPU Processed stock " << i << " final price estimate: " << results[i] << std::endl;
    }
}

int main() {
    std::vector<Stock> stocks = {{0.0005, 0.01, 100}, {0.0006, 0.012, 150}, /* more stocks */};
    run_simulation_cpu(stocks, 365);  // Run the simulation for one year
    return 0;
}

/*
To run:
g++ -o financial_market_analysis_cpu.out financial_market_analysis_cpu.cpp
./financial_market_analysis_cpu.out
*/

