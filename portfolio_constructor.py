"""
Portfolio Constructor
Implements various portfolio construction methods including:
- Equal Weight
- Minimum Variance
- Maximum Sharpe Ratio
- Mean-Variance Optimization (Efficient Frontier)
"""
import datetime
import os

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
#import matplotlib
#matplotlib.use('Agg')  # Non-blocking backend for PyCharm
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, BlackLittermanModel, HRPOpt
from pypfopt import risk_models, expected_returns


class PortfolioConstructor:
    """
    Portfolio construction using various optimization methods.

    Parameters:
    -----------
    prices : pandas.DataFrame
        DataFrame with dates as index and tickers as columns (from data loader)
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe ratio calculations (default: 0.02 = 2%)
    """

    def __init__(self, prices, risk_free_rate=0.02):
        self.prices = prices
        self.risk_free_rate = risk_free_rate

        # Calculate returns
        self.returns = prices.pct_change().dropna()

        # Calculate expected returns (annualized mean)
        self.expected_returns = self.returns.mean() * 252

        # Calculate covariance matrix (annualized)
        self.cov_matrix = self.returns.cov() * 252

        # Number of assets
        self.n_assets = len(prices.columns)
        self.tickers = list(prices.columns)

        print(f"Portfolio Constructor initialized with {self.n_assets} assets")
        print(f"Data period: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Number of observations: {len(self.returns)}")


    def equal_weight(self):
        """
        Equal weight portfolio - simplest approach.

        Returns:
        --------
        dict : Dictionary with 'weights' and 'method' keys
        """
        weights = np.array([1.0 / self.n_assets] * self.n_assets)

        return {
            'weights': pd.Series(weights, index=self.tickers),
            'method': 'Equal Weight',
            'expected_return': self._portfolio_return(weights),
            'volatility': self._portfolio_volatility(weights),
            'sharpe_ratio': self._portfolio_sharpe(weights)
        }

    def custom_weights(self, weights, name="Custom Portfolio"):
        """
        Use custom weights provided by user.

        Parameters:
        -----------
        weights : dict or pandas.Series
            Custom weights for each ticker.
        name : str, optional
            Name for this portfolio (default: "Custom Portfolio")
        """
        # Convert to Series if dict
        if isinstance(weights, dict):
            weights = pd.Series(weights)

        # Ensure weights match our tickers
        weights = weights.reindex(self.tickers, fill_value=0)

        # Normalize to sum to 1
        total_weight = weights.sum()
        if total_weight == 0:
            raise ValueError("All weights are zero!")
        weights = weights / total_weight

        # Convert to numpy array for calculations
        weights_array = weights.values

        return {
            'weights': weights,
            'method': name,
            'expected_return': self._portfolio_return(weights_array),
            'volatility': self._portfolio_volatility(weights_array),
            'sharpe_ratio': self._portfolio_sharpe(weights_array)
        }

    def minimum_variance(self, allow_short=False):
        """
        Minimum variance portfolio - minimizes risk.

        Parameters:
        -----------
        allow_short : bool, optional
            If True, allows short selling (negative weights). Default: False

        Returns:
        --------
        dict : Dictionary with portfolio details
        """
        # Objective: minimize portfolio variance
        def objective(weights):
            return self._portfolio_volatility(weights)

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds: weights between 0 and 1 (or -1 and 1 if shorting allowed)
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            print(f"Warning: Optimization did not converge - {result.message}")

        weights = result.x

        return {
            'weights': pd.Series(weights, index=self.tickers),
            'method': 'Minimum Variance',
            'expected_return': self._portfolio_return(weights),
            'volatility': self._portfolio_volatility(weights),
            'sharpe_ratio': self._portfolio_sharpe(weights)
        }


    def maximum_sharpe(self, allow_short=False):
        """
        Maximum Sharpe ratio portfolio - best risk-adjusted returns.

        Parameters:
        -----------
        allow_short : bool, optional
            If True, allows short selling (negative weights). Default: False

        Returns:
        --------
        dict : Dictionary with portfolio details
        """
        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        def objective(weights):
            return -self._portfolio_sharpe(weights)

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            print(f"Warning: Optimization did not converge - {result.message}")

        weights = result.x

        return {
            'weights': pd.Series(weights, index=self.tickers),
            'method': 'Maximum Sharpe Ratio',
            'expected_return': self._portfolio_return(weights),
            'volatility': self._portfolio_volatility(weights),
            'sharpe_ratio': self._portfolio_sharpe(weights)
        }

    def black_litterman(self, market_caps=None, views=None, view_confidences=None,
                        risk_aversion=1, allow_short=False):
        """
        Black-Litterman portfolio optimization.

        Combines market equilibrium with your own views about expected returns.

        Parameters:
        -----------
        market_caps : dict or pandas.Series, optional
            Market capitalizations for each asset. If None, uses equal weights.
            Example: {'AAPL': 3000000000000, 'MSFT': 2500000000000}
        views : dict, optional
            Your views on expected returns.
            Example: {'AAPL': 0.15} means you expect AAPL to return 15%
        view_confidences : dict or list, optional
            Confidence in each view (0 to 1). Higher = more confident.
            If dict: {'AAPL': 0.8} means 80% confident
            If list: [0.8, 0.5] for views in order
        risk_aversion : float, optional
            Risk aversion parameter (default: 1)
        allow_short : bool, optional
            Allow short selling (default: False)

        Returns:
        --------
        dict : Portfolio with weights and metrics

        Example:
        --------
        # Express views that tech will outperform
        views = {'AAPL': 0.20, 'MSFT': 0.18}  # Expected 20% and 18% returns
        confidences = {'AAPL': 0.8, 'MSFT': 0.7}  # 80% and 70% confident

        portfolio = constructor.black_litterman(views=views, view_confidences=confidences)
        """
        # Calculate expected returns using CAPM (if no views provided)
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()

        # Market-cap weights (or equal if not provided)
        if market_caps is None:
            market_prior = np.array([1 / self.n_assets] * self.n_assets)
        else:
            if isinstance(market_caps, dict):
                market_caps = pd.Series(market_caps)
            market_caps = market_caps.reindex(self.tickers, fill_value=0)
            market_prior = market_caps / market_caps.sum()

        omega = None
        if view_confidences is not None:
            if isinstance(view_confidences, dict):
                tau = 0.05
                omega_dict = {}
                for asset, confidence in view_confidences.items():
                    if views is None or asset not in views:
                        continue
                    asset_idx = self.tickers.index(asset)
                    asset_var = S.iloc[asset_idx, asset_idx]
                    conf = max(confidence, 0.01)
                    omega_dict[asset] = tau * asset_var * (1 / conf - 1)

                # Build diagonal omega in view order, not ticker order
                if views is not None:
                    omega_vals = []
                    for asset in views.keys():
                        asset_idx = self.tickers.index(asset)
                        asset_var = S.iloc[asset_idx, asset_idx]
                        omega_vals.append(omega_dict.get(asset, tau * asset_var))
                    omega = np.diag(omega_vals)
            else:
                # scalar or list – let PyPortfolioOpt handle via Idzorek mapping
                omega = "idzorek"
        else:
            omega = "idzorek"

        # Black-Litterman model
        bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views,
                                omega=omega, risk_aversion=risk_aversion)

        # Get posterior expected returns
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        # Optimize
        ef = EfficientFrontier(bl_returns, bl_cov)
        if allow_short:
            ef.min_volatility()
        else:
            ef.add_constraint(lambda w: w >= 0)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)

        weights = ef.clean_weights()
        weights_series = pd.Series(weights)

        # Calculate metrics
        perf = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)

        return {
            'weights': weights_series,
            'method': 'Black-Litterman',
            'expected_return': perf[0],
            'volatility': perf[1],
            'sharpe_ratio': perf[2]
        }

    def hierarchical_risk_parity(self,show_dendrogram=False,ticker_names=None,output_folder='output'):
        """
      Hierarchical Risk Parity (HRP) portfolio.

        Uses hierarchical clustering to build a diversified portfolio.
        Works well when correlations are unstable.

        Parameters:
        -----------
        show_dendrogram : bool, optional
            If True, displays the hierarchical clustering dendrogram (default: False)
        ticker_names : dict, optional
            Dictionary mapping ticker symbols to friendly names for dendrogram labels.
            Example: {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft'}

        Returns:
        --------
        dict : Portfolio with weights and metrics

        Example:
        --------
        # Run HRP on subset of assets and show dendrogram with friendly names
        TICKERS = {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google'}

        hrp_portfolio = constructor.hierarchical_risk_parity(
            assets_subset=['AAPL', 'MSFT', 'GOOGL'],
            show_dendrogram=True,
            ticker_names=TICKERS
        )
        """
        # Calculate returns
        returns = self.returns
        S = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()

        # HRP optimization
        hrp = HRPOpt(returns = returns, cov_matrix = S)
        weights = hrp.optimize()

        weights_series = pd.Series(weights)
        weights_array = weights_series.values

        if show_dendrogram:

            dendrogram_tickers = self.tickers
            dendrogram_returns = returns
            # Calculate correlation matrix
            corr = dendrogram_returns.corr()

            # Convert correlation to distance: distance = sqrt(0.5 * (1 - correlation))
            dist = np.sqrt(0.5 * (1 - corr))

            # Convert to condensed distance matrix (required by linkage)
            dist_condensed = squareform(dist, checks=False)

            # Perform hierarchical clustering
            linkage_matrix = linkage(dist_condensed, method='single')

            if ticker_names:
                labels = [ticker_names.get(ticker, ticker) for ticker in dendrogram_tickers]
            else:
                labels = dendrogram_tickers

            # Plot dendrogram
            plt.figure(figsize=(12, 6))
            dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
            plt.title('HRP Clustering Dendrogram', fontsize=14, fontweight='bold')
            plt.xlabel('Assets', fontsize=12)
            plt.ylabel('Distance', fontsize=12)
            plt.tight_layout()

            os.makedirs(output_folder, exist_ok=True)
            figure_filename = os.path.join(output_folder, f'dendrogram.png')
            plt.savefig(figure_filename, dpi=300, bbox_inches='tight')

            plt.show(block=False)

        return {
            'weights': weights_series,
            'method': 'Hierarchical Risk Parity',
            'expected_return': self._portfolio_return(weights_array),
            'volatility': self._portfolio_volatility(weights_array),
            'sharpe_ratio': self._portfolio_sharpe(weights_array)
        }

    def robust_mean_variance(self, uncertainty_penalty=0.1, allow_short=False):
        """
        Robust Mean-Variance Optimization.

        Accounts for uncertainty in expected returns using robust optimization.
        More conservative than standard mean-variance.

        Parameters:
        -----------
        uncertainty_penalty : float, optional
            How much to penalize uncertainty (default: 0.1)
            Higher values = more conservative
        allow_short : bool, optional
            Allow short selling (default: False)

        Returns:
        --------
        dict : Portfolio with weights and metrics
        """

        # Expected returns and covariance
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()

        # Robust optimization with uncertainty sets
        ef = EfficientFrontier(mu, S)

        # Add robustness by adjusting expected returns
        # Subtract uncertainty penalty from expected returns
        robust_mu = mu - uncertainty_penalty * np.sqrt(np.diag(S))

        ef = EfficientFrontier(robust_mu, S)

        if allow_short:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            ef.add_constraint(lambda w: w >= 0)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)

        weights = ef.clean_weights()
        weights_series = pd.Series(weights)

        # Calculate metrics
        perf = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)

        return {
            'weights': weights_series,
            'method': 'Robust Mean-Variance',
            'expected_return': perf[0],
            'volatility': perf[1],
            'sharpe_ratio': perf[2]
        }


    def efficient_frontier(self, n_portfolios=50, allow_short=False):
        """
        Generate efficient frontier portfolios.

        Parameters:
        -----------
        n_portfolios : int, optional
            Number of portfolios to generate along the frontier (default: 50)
        allow_short : bool, optional
            If True, allows short selling (default: False)

        Returns:
        --------
        pandas.DataFrame : DataFrame with return, volatility, and weights for each portfolio
        """
        # Find min and max returns for the frontier
        min_var_portfolio = self.minimum_variance(allow_short=allow_short)
        max_sharpe_portfolio = self.maximum_sharpe(allow_short=allow_short)

        min_return = min_var_portfolio['expected_return']
        max_return = self.expected_returns.max()

        # Target returns along the frontier
        target_returns = np.linspace(min_return, max_return, n_portfolios)

        frontier_portfolios = []

        print(f"Generating efficient frontier with {n_portfolios} portfolios...")

        for target_return in target_returns:
            portfolio = self._efficient_return(target_return, allow_short)
            if portfolio is not None:
                frontier_portfolios.append(portfolio)

        # Create DataFrame
        df = pd.DataFrame(frontier_portfolios)

        print(f"✓ Generated {len(df)} efficient portfolios")

        return df


    def _efficient_return(self, target_return, allow_short=False):
        """
        Find minimum variance portfolio for a target return.
        """
        # Objective: minimize variance
        def objective(weights):
            return self._portfolio_volatility(weights)

        # Constraints:
        # 1. Weights sum to 1
        # 2. Portfolio return equals target
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self._portfolio_return(w) - target_return}
        ]

        # Bounds
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            return None

        weights = result.x

        # Create result dictionary with weights as separate columns
        result_dict = {
            'expected_return': self._portfolio_return(weights),
            'volatility': self._portfolio_volatility(weights),
            'sharpe_ratio': self._portfolio_sharpe(weights)
        }

        # Add individual weights
        for i, ticker in enumerate(self.tickers):
            result_dict[ticker] = weights[i]

        return result_dict

    def plot_efficient_frontier(self, frontier_df=None, show_assets=True, custom_portfolios=None):
        """
        Plot efficient frontier with multiple custom portfolios.

        Parameters:
        -----------
        custom_portfolios : list of dicts, optional
            List of custom portfolios from self.custom_weights().
            Each dict should have 'expected_return', 'volatility', 'weights', 'method' keys.
        """
        if frontier_df is None:
            frontier_df = self.efficient_frontier()

        if custom_portfolios is None:
            custom_portfolios = []

        plt.figure(figsize=(14, 9))

        # Efficient frontier
        plt.plot(frontier_df['volatility'], frontier_df['expected_return'],
                 'b-', linewidth=3, label='Efficient Frontier')

        # Individual assets
        if show_assets:
            asset_returns = self.expected_returns
            asset_vols = np.sqrt(np.diag(self.cov_matrix))
            plt.scatter(asset_vols, asset_returns.values, c='red', s=60, alpha=0.7,
                        label='Individual Assets', zorder=3)

            for i, ticker in enumerate(self.tickers):
                plt.annotate(ticker, (asset_vols[i], asset_returns.iloc[i]),
                             xytext=(8, 8), textcoords='offset points', fontsize=9)

        # Special portfolios
        portfolios = [
            ('Equal Weight', self.equal_weight(), 'green', 's'),
            ('Min Variance', self.minimum_variance(), 'orange', '^'),
            ('Max Sharpe', self.maximum_sharpe(), 'gold', '*')
        ]

        for name, port, color, marker in portfolios:
            plt.scatter(port['volatility'], port['expected_return'],
                        c=color, s=150, marker=marker, label=name, zorder=5,
                        edgecolors='black', linewidth=1.5)

        # MULTIPLE CUSTOM PORTFOLIOS
        for i, custom_port in enumerate(custom_portfolios):
            label = custom_port.get('method', f'Custom {i + 1}')
            plt.scatter(custom_port['volatility'], custom_port['expected_return'],
                        c=f'C{i}', s=200, marker='D', label=label, zorder=6,
                        edgecolors='black', linewidth=2, alpha=0.9)

        plt.xlabel('Volatility (Annualized Std Dev)', fontsize=12)
        plt.ylabel('Expected Return (Annualized)', fontsize=12)
        plt.title('Efficient Frontier', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compare_portfolios(self, custom_portfolios=None):
        """
        Compare portfolios including multiple custom portfolios.

        Parameters:
        -----------
        custom_portfolios : list of dict or dict, optional
            Single dict or list of custom portfolio dicts/weight dicts.
        """
        if custom_portfolios is None:
            custom_portfolios = []
        elif isinstance(custom_portfolios, dict):
            custom_portfolios = [custom_portfolios]

        portfolios = [
            self.equal_weight(),
            self.minimum_variance(),
            self.maximum_sharpe()
        ]

        # Add custom portfolios
        for i, item in enumerate(custom_portfolios):
            if 'expected_return' in item:  # Already computed portfolio
                portfolios.append(item)
            elif isinstance(item, dict) and not ('expected_return' in item):  # Raw weights dict
                port = self.custom_weights(item)
                port['method'] = f'Custom {i + 1}'
                portfolios.append(port)
            else:
                print(f"Skipping invalid portfolio {i}: {type(item)}")

        # Create comparison table
        comparison = pd.DataFrame([
            {
                'Method': p['method'],
                'Expected Return': f"{p['expected_return']:.2%}",
                'Volatility': f"{p['volatility']:.2%}",
                'Sharpe Ratio': f"{p['sharpe_ratio']:.3f}"
            }
            for p in portfolios
        ])

        print("\n" + "=" * 70)
        print("PORTFOLIO COMPARISON (incl. Custom Portfolios)")
        print("=" * 70)
        print(comparison.to_string(index=False))
        print("=" * 70)

        # Show weights
        print("\nPORTFOLIO WEIGHTS:")
        print("-" * 70)
        for p in portfolios:
            print(f"\n{p['method']}:")
            weights_pct = p['weights'].map(lambda x: f"{x:.1%}")
            print(weights_pct.to_string())

        return portfolios


    # Helper functions for portfolio metrics
    def _portfolio_return(self, weights):
        """Calculate portfolio expected return."""
        return np.dot(weights, self.expected_returns)

    def _portfolio_volatility(self, weights):
        """Calculate portfolio volatility (standard deviation)."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_sharpe(self, weights):
        """Calculate portfolio Sharpe ratio."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # This assumes you have price data from the data loader
    # Example: prices = get_market_close_data(...)

    # For demonstration, create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Simulate prices
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )

    # Initialize constructor
    constructor = PortfolioConstructor(prices, risk_free_rate=0.02)

    # Compare all methods
    portfolios = constructor.compare_portfolios()

    # Generate and plot efficient frontier
    frontier = constructor.efficient_frontier(n_portfolios=50)
    constructor.plot_efficient_frontier(frontier)

