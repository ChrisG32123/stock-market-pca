double simulatePath(const SimulationParameters& params, int pathIndex) {
    double portfolio_return = 0.0;
    // Assuming asset_returns is a pre-computed matrix of simulated returns for each asset
    for (int asset = 0; asset < params.numAssets; ++asset) {
        // Calculate the weighted return of each asset and sum them up
        portfolio_return += asset_weights[asset] * asset_returns[asset][pathIndex];
    }
    return portfolio_return; // Return the total portfolio return for this path
}

void calculateRiskMetrics(const std::vector<double>& portfolio_returns) {
    std::sort(portfolio_returns.begin(), portfolio_returns.end());
    int VaR_index = static_cast<int>(portfolio_returns.size() * 0.05); // For 95% VaR
    double VaR = portfolio_returns[VaR_index];
    double CVaR = std::accumulate(portfolio_returns.begin(), portfolio_returns.begin() + VaR_index, 0.0) / VaR_index;
    std::cout << "VaR: " << VaR << ", CVaR: " << CVaR << std::endl;
}
