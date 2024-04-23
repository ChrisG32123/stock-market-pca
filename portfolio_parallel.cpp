#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <cstdlib>  // For atoi and atof
#include <sys/stat.h>  // Include this for struct stat and the stat() function

using namespace Eigen;
using namespace std;

MatrixXd loadCSV(const string& path) {
    ifstream inFile(path);
    string line;
    vector<vector<double>> data;
    getline(inFile, line);  // Skip the header
    while (getline(inFile, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
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

bool fileExistsAndNotEmpty(const string& filename) {
    struct stat buffer;  // Using struct stat to get file attributes
    if (stat(filename.c_str(), &buffer) != 0) {
        return false;  // File does not exist or cannot be accessed
    }
    return buffer.st_size > 0;  // Check if the file is not empty by checking its size
}

MatrixXd calculateLogReturns(const MatrixXd& prices) {
    MatrixXd logReturns(prices.rows() - 1, prices.cols());
    for (int i = 1; i < prices.rows(); ++i) {
        logReturns.row(i - 1) = (prices.row(i).array() / prices.row(i - 1).array()).log();
    }
    return logReturns;
}

MatrixXd calculateCovarianceMatrix(const MatrixXd& returns) {
    MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    return (centered.adjoint() * centered) / (returns.rows() - 1) * 252;  // Annualize covariance
}

tuple<double, VectorXd, double> calculatePortfolioDetails(const VectorXd& weights, const MatrixXd& prices, double riskFreeRate) {
    MatrixXd logReturns = calculateLogReturns(prices);
    MatrixXd covMatrix = calculateCovarianceMatrix(logReturns);
    VectorXd meanReturns = logReturns.colwise().mean() * 252;  // Annualize returns
    double portfolioReturn = weights.dot(meanReturns);
    double portfolioVolatility = sqrt((weights.transpose() * covMatrix * weights)(0, 0));
    double sharpeRatio = (portfolioReturn - riskFreeRate) / portfolioVolatility;
    double confidenceInterval = 1.96 * portfolioVolatility / sqrt(logReturns.rows()); // 95% confidence interval
    return make_tuple(sharpeRatio, weights, confidenceInterval);
}

void broadcastMatrix(MatrixXd &matrix, MPI_Comm comm) {
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);
    int rows = matrix.rows();
    int cols = matrix.cols();
    MPI_Bcast(&rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cols, 1, MPI_INT, 0, comm);
    if (world_rank != 0) {
        matrix.resize(rows, cols);
    }
    MPI_Bcast(matrix.data(), rows * cols, MPI_DOUBLE, 0, comm);
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Pass In Command Line Arguments
    if (argc != 6) {
        if (world_rank == 0) {
            cerr << "Usage: " << argv[0] << " <num_openmp_threads> <num_portfolios> <risk_free_rate> <trading_days> <output_file>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int openMPThreads = atoi(argv[1]);
    int numPortfolios = atoi(argv[2]);
    double riskFreeRate = atof(argv[3]);
    int tradingDays = atoi(argv[4]);
    string outputFilename = argv[5];

    omp_set_num_threads(openMPThreads);
    double mpi_start_time = MPI_Wtime();

    // Load S&P 500 Data
    MatrixXd prices;
    if (world_rank == 0) {
    prices = loadCSV("adj_close_df.csv");
        if (prices.size() == 0) {
            cerr << "Error: No data loaded from CSV." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // Broadcast Data To MPI Ranks
    broadcastMatrix(prices, MPI_COMM_WORLD);

    vector<double> localSharpeRatios(numPortfolios / world_size);
    vector<VectorXd> localWeights(numPortfolios / world_size);
    vector<double> localConfidenceIntervals(numPortfolios / world_size);

    double omp_start_time = 0, omp_end_time = 0;

    // Run Monte Carlo Simulation
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Load Balancing Through Dynamic Scheduling
        #pragma omp for reduction(+:omp_start_time, omp_end_time) schedule(dynamic)
        for (int i = 0; i < numPortfolios / world_size; ++i) {
            VectorXd weights = VectorXd::Random(prices.cols()).array().abs();
            weights /= weights.sum();  // Normalize weights to sum to 1
            
            // Calculate OpenMP Time by Adding Walltimes for all Threads and dividing by numPortfolios
            omp_start_time = omp_get_wtime();

            auto details = calculatePortfolioDetails(weights, prices, riskFreeRate);
            localSharpeRatios[i] = get<0>(details);
            localWeights[i] = get<1>(details);
            localConfidenceIntervals[i] = get<2>(details);

            omp_end_time += omp_get_wtime() - omp_start_time;
        }
    }

    // Collect Data From All MPI Ranks
    vector<double> globalSharpeRatios;
    vector<double> globalConfidenceIntervals;
    vector<double> allWeights;
    if (world_rank == 0) {
        globalSharpeRatios.resize(numPortfolios);
        globalConfidenceIntervals.resize(numPortfolios);
        allWeights.resize(numPortfolios * prices.cols());
    }

    // Gather all local Sharpe Ratios
    MPI_Gather(localSharpeRatios.data(), numPortfolios / world_size, MPI_DOUBLE,
               globalSharpeRatios.data(), numPortfolios / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gather all local Confidence Intervals
    MPI_Gather(localConfidenceIntervals.data(), numPortfolios / world_size, MPI_DOUBLE,
               globalConfidenceIntervals.data(), numPortfolios / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gather all weights
    for (int i = 0; i < numPortfolios / world_size; ++i) {
        MPI_Gather(localWeights[i].data(), prices.cols(), MPI_DOUBLE,
                   &allWeights[i * prices.cols()], prices.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double mpi_end_time = MPI_Wtime();
    
    // Save Data To CSV
    if (world_rank == 0) {
        bool fileExists = fileExistsAndNotEmpty(outputFilename);
        ofstream resultsFile(outputFilename, ios::out | ios::app); // Open file in append mode

        cout << "Checkpoint 3 - Starting to write to file." << endl;
    
        if (!fileExists) {
            resultsFile << "Max Sharpe Ratio,Optimal Weights,Confidence Interval,MPI Wall Time,OMP Wall Time,OMP Threads,MPI Ranks\n";
        }

        for (int i = 0; i < numPortfolios; ++i) {
            resultsFile << globalSharpeRatios[i] << ",";
            for (int j = 0; j < prices.cols(); ++j) {
                resultsFile << allWeights[i * prices.cols() + j] << (j < prices.cols() - 1 ? " " : "");
            }
            resultsFile << "," << globalConfidenceIntervals[i] << ",";
            resultsFile << (mpi_end_time - mpi_start_time) << ",";
            resultsFile << (omp_end_time / numPortfolios) << ",";
            resultsFile << openMPThreads << ",";
            resultsFile << world_size << "\n";
        }

        resultsFile.close();
    }
    
    MPI_Finalize();
    return 0;
}

