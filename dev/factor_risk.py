# factor_risk_model.py

import numpy as np


def generate_dummy_data(n_assets=5, n_factors=3):
    """
    Generates dummy data for the factor risk model.
    Args:
        n_assets (int): Number of assets in the portfolio.
        n_factors (int): Number of factors in the model.

    Returns:
        dict: Dictionary containing dummy data for:
              - asset_returns: n_assets x time_periods matrix of returns
              - factor_exposures: n_assets x n_factors matrix
              - factor_covariance: n_factors x n_factors matrix
              - specific_risk_matrix: n_assets x n_assets diagonal matrix
              - portfolio_weights: n_assets x 1 vector
    """
    np.random.seed(42)  # For reproducibility

    # Random asset returns matrix (n_assets x time_periods)
    asset_returns = np.random.randn(n_assets, 100) * 0.01

    # Random factor exposures matrix (n_assets x n_factors)
    factor_exposures = np.random.randn(n_assets, n_factors)

    # Random factor covariance matrix (n_factors x n_factors)
    factor_covariance = np.random.randn(n_factors, n_factors)
    # Ensure it's positive semi-definite
    factor_covariance = np.dot(factor_covariance, factor_covariance.T)

    # Random specific risk matrix (n_assets x n_assets) - diagonal matrix
    specific_risk_variances = np.random.rand(n_assets) * 0.02
    specific_risk_matrix = np.diag(specific_risk_variances)

    # Random portfolio weights (n_assets x 1) - normalized to sum to 1
    portfolio_weights = np.random.rand(n_assets, 1)
    portfolio_weights /= np.sum(portfolio_weights)

    return {
        'asset_returns': asset_returns,
        'factor_exposures': factor_exposures,
        'factor_covariance': factor_covariance,
        'specific_risk_matrix': specific_risk_matrix,
        'portfolio_weights': portfolio_weights
    }


def calculate_systematic_risk(factor_exposures, factor_covariance, portfolio_weights):
    """
    Calculates the systematic risk of the portfolio.
    Args:
        factor_exposures (ndarray): n_assets x n_factors matrix of factor exposures.
        factor_covariance (ndarray): n_factors x n_factors factor covariance matrix.
        portfolio_weights (ndarray): n_assets x 1 vector of portfolio weights.

    Returns:
        float: Systematic risk of the portfolio.
    """
    # Calculate the portfolio factor exposure
    portfolio_factor_exposure = np.dot(factor_exposures.T, portfolio_weights)

    # Calculate the systematic risk
    systematic_risk = np.dot(np.dot(
        portfolio_factor_exposure.T, factor_covariance), portfolio_factor_exposure)
    return np.sqrt(systematic_risk)


def calculate_specific_risk(specific_risk_matrix, portfolio_weights):
    """
    Calculates the specific risk of the portfolio.
    Args:
        specific_risk_matrix (ndarray): n_assets x n_assets diagonal matrix of specific risks.
        portfolio_weights (ndarray): n_assets x 1 vector of portfolio weights.

    Returns:
        float: Specific risk of the portfolio.
    """
    specific_risk = np.dot(
        np.dot(portfolio_weights.T, specific_risk_matrix), portfolio_weights)
    return np.sqrt(specific_risk)


def calculate_total_risk(factor_exposures, factor_covariance, specific_risk_matrix, portfolio_weights):
    """
    Calculates the total risk of the portfolio by combining systematic and specific risks.
    Args:
        factor_exposures (ndarray): n_assets x n_factors matrix of factor exposures.
        factor_covariance (ndarray): n_factors x n_factors factor covariance matrix.
        specific_risk_matrix (ndarray): n_assets x n_assets diagonal matrix of specific risks.
        portfolio_weights (ndarray): n_assets x 1 vector of portfolio weights.

    Returns:
        float: Total risk of the portfolio.
    """
    systematic_risk = calculate_systematic_risk(
        factor_exposures, factor_covariance, portfolio_weights)
    specific_risk = calculate_specific_risk(
        specific_risk_matrix, portfolio_weights)
    total_risk = np.sqrt(systematic_risk**2 + specific_risk**2)
    return total_risk


def run_factor_risk_model():
    """
    Demonstrates the factor risk model using dummy data.
    """
    data = generate_dummy_data()

    asset_returns = data['asset_returns']
    factor_exposures = data['factor_exposures']
    factor_covariance = data['factor_covariance']
    specific_risk_matrix = data['specific_risk_matrix']
    portfolio_weights = data['portfolio_weights']

    systematic_risk = calculate_systematic_risk(
        factor_exposures, factor_covariance, portfolio_weights)
    specific_risk = calculate_specific_risk(
        specific_risk_matrix, portfolio_weights)
    total_risk = calculate_total_risk(
        factor_exposures, factor_covariance, specific_risk_matrix, portfolio_weights)

    print(f"Systematic Risk: {systematic_risk:.4f}")
    print(f"Specific Risk: {specific_risk:.4f}")
    print(f"Total Portfolio Risk: {total_risk:.4f}")


# Run the model with dummy data
if __name__ == "__main__":
    run_factor_risk_model()
