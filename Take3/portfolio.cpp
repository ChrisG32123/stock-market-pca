#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <string>

using namespace Eigen;
using namespace std;

MatrixXd loadCSV(const string& path) {
    ifstream inFile(path);
    string line;
    vector<vector<double>> data;

    // Skip the header row
    getline(inFile, line);

    while (getline(inFile, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;
        while (getline(ss, value, ',')) {
            try {
                row.push_back(stod(value));
            } catch (const invalid_argument& e) {
                cerr << "Error converting string to double: '" << value << "'" << endl;
                exit(EXIT_FAILURE);  // Exit if any non-convertible values are found
            }
        }
        data.push_back(row);
    }

    MatrixXd mat(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            mat(i, j) = data[i][j];
        }
    }
    return mat;
}

MatrixXd calculateLogReturns(const MatrixXd& prices) {
    MatrixXd logReturns(prices.rows() - 1, prices.cols());
    for (int i = 1; i < prices.rows(); ++i) {
        for (int j = 0; j < prices.cols(); ++j) {
            logReturns(i - 1, j) = log(prices(i, j) / prices(i - 1, j));
        }
    }
    return logReturns;
}

MatrixXd calculateCovarianceMatrix(const MatrixXd& returns) {
    MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    MatrixXd cov = centered.adjoint() * centered / double(returns.rows() - 1);
    return cov * 252;
}

double standardDeviation(const VectorXd& weights, const MatrixXd& covMatrix) {
    return sqrt((weights.transpose() * covMatrix * weights)(0, 0));
}

double expectedReturn(const VectorXd& weights, const MatrixXd& returns) {
    VectorXd meanReturns = returns.colwise().mean();
    return weights.dot(meanReturns) * 252;
}

double sharpeRatio(const VectorXd& weights, const MatrixXd& returns, const MatrixXd& covMatrix, double riskFreeRate) {
    double expRet = expectedReturn(weights, returns);
    double stdDev = standardDeviation(weights, covMatrix);
    return (expRet - riskFreeRate) / stdDev;
}

int main() {
    string csvPath = "adj_close_df.csv";
    MatrixXd adjClosePrices = loadCSV(csvPath);
    MatrixXd logReturns = calculateLogReturns(adjClosePrices);
    MatrixXd covMatrix = calculateCovarianceMatrix(logReturns);

    const int numAssets = adjClosePrices.cols();
    const int numPortfolios = 10000;
    const double riskFreeRate = 0.02;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    double maxSharpeRatio = -numeric_limits<double>::infinity();
    VectorXd optimalWeights(numAssets);

    for (int i = 0; i < numPortfolios; ++i) {
        VectorXd weights(numAssets);
        double sum = 0.0;
        for (int j = 0; j < numAssets; ++j) {
            weights(j) = dis(gen);
            sum += weights(j);
        }
        weights /= sum; // Normalize the weights

        double sharpe = sharpeRatio(weights, logReturns, covMatrix, riskFreeRate);
        if (sharpe > maxSharpeRatio) {
            maxSharpeRatio = sharpe;
            optimalWeights = weights;
        }
    }

    cout << "Optimal Weights:" << endl;
    for (int i = 0; i < numAssets; ++i) {
        cout << "Asset " << i << ": " << optimalWeights(i) << endl;
    }
    return 0;
}

/*
g++ -I /path/to/eigen -o portfolio_optimization portfolio_optimization.cpp
./portfolio_optimization
*/