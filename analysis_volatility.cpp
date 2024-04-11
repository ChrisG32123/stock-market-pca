// Includes
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <omp.h>

// Use standard namespace for simplicity in this example
using namespace std;

// Function to simulate portfolio paths
vector<double> simulatePortfolioPaths(int numPaths, int numSteps, double dt, double initialPrice, double mu, double sigma) {
    vector<double> endPrices(numPaths);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);

    #pragma omp parallel for
    for (int path = 0; path < numPaths; ++path) {
        double price = initialPrice;
        for (int step = 0; step < numSteps; ++step) {
            double dW = dist(gen) * sqrt(dt);
            price += mu * price * dt + sigma * price * dW;
        }
        endPrices[path] = price;
    }
    return endPrices;
}

// Main function
int main() {
    // Simulation parameters
    int numPaths = 10000;
    int numSteps = 365;
    double dt = 1.0 / numSteps;
    double initialPrice = 100.0; // Example initial price
    double mu = 0.05; // Expected return
    double sigma = 0.2; // Volatility

    // Run simulation
    vector<double> endPrices = simulatePortfolioPaths(numPaths, numSteps, dt, initialPrice, mu, sigma);

    // Output results to a CSV file
    ofstream outFile("simulation_results.csv");
    outFile << "EndPrice\n";
    for (double price : endPrices) {
        outFile << price << "\n";
    }
    outFile.close();

    cout << "Simulation completed. Results are saved to simulation_results.csv." << endl;

    return 0;
}
