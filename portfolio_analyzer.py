"""
Portfolio Analyzer
Comprehensive portfolio performance analysis including:
- Returns analysis (daily, cumulative, annualized)
- Risk metrics (volatility, drawdown, VaR, CVaR)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Attribution analysis
- Correlation and diversification metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PortfolioAnalyzer:
    """
    Analyze portfolio performance with comprehensive metrics.

    Parameters:
    -----------
    prices : pandas.DataFrame
        DataFrame with dates as index and tickers as columns
    weights : pandas.Series or dict
        Portfolio weights for each ticker
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 = 2%)
    """

    def __init__(self, prices, weights, risk_free_rate=0.02):
        self.prices = prices
        self.risk_free_rate = risk_free_rate

        # Convert weights to Series if dict
        if isinstance(weights, dict):
            weights = pd.Series(weights)

        # Ensure weights match price columns
        self.weights = weights.reindex(prices.columns, fill_value=0)

        # Normalize weights to sum to 1
        self.weights = self.weights / self.weights.sum()

        # Calculate returns
        self.returns = prices.pct_change().dropna()

        # Calculate portfolio returns
        self.portfolio_returns = (self.returns * self.weights).sum(axis=1)

        # Calculate portfolio value over time (starting at 100)
        self.portfolio_value = (1 + self.portfolio_returns).cumprod() * 100

        print(f"Portfolio Analyzer initialized")
        print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Assets: {len(self.weights)} tickers")
        print(f"Total observations: {len(self.portfolio_returns)}")

    def summary_statistics(self):
        """
        Calculate comprehensive summary statistics.

        Returns:
        --------
        dict : Dictionary of performance metrics
        """
        stats = {}

        # Return metrics
        stats['Total Return'] = self._total_return()
        stats['Annualized Return'] = self._annualized_return()
        stats['Daily Mean Return'] = self.portfolio_returns.mean()

        # Risk metrics
        stats['Annualized Volatility'] = self._annualized_volatility()
        stats['Max Drawdown'] = self._max_drawdown()
        stats['VaR (95%)'] = self._value_at_risk(0.95)
        stats['CVaR (95%)'] = self._conditional_var(0.95)

        # Risk-adjusted metrics
        stats['Sharpe Ratio'] = self._sharpe_ratio()
        stats['Sortino Ratio'] = self._sortino_ratio()
        stats['Calmar Ratio'] = self._calmar_ratio()

        # Other metrics
        stats['Winning Days %'] = (self.portfolio_returns > 0).sum() / len(self.portfolio_returns) * 100
        stats['Best Day'] = self.portfolio_returns.max()
        stats['Worst Day'] = self.portfolio_returns.min()

        return stats

    def print_summary(self):
        """Print formatted summary statistics."""
        stats = self.summary_statistics()

        print("\n" + "=" * 60)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("=" * 60)

        print("\nRETURN METRICS:")
        print("-" * 60)
        print(f"Total Return:           {stats['Total Return']:>12.2%}")
        print(f"Annualized Return:      {stats['Annualized Return']:>12.2%}")
        print(f"Daily Mean Return:      {stats['Daily Mean Return']:>12.4%}")

        print("\nRISK METRICS:")
        print("-" * 60)
        print(f"Annualized Volatility:  {stats['Annualized Volatility']:>12.2%}")
        print(f"Max Drawdown:           {stats['Max Drawdown']:>12.2%}")
        print(f"VaR (95%):              {stats['VaR (95%)']:>12.4%}")
        print(f"CVaR (95%):             {stats['CVaR (95%)']:>12.4%}")

        print("\nRISK-ADJUSTED METRICS:")
        print("-" * 60)
        print(f"Sharpe Ratio:           {stats['Sharpe Ratio']:>12.3f}")
        print(f"Sortino Ratio:          {stats['Sortino Ratio']:>12.3f}")
        print(f"Calmar Ratio:           {stats['Calmar Ratio']:>12.3f}")

        print("\nOTHER METRICS:")
        print("-" * 60)
        print(f"Winning Days:           {stats['Winning Days %']:>12.1f}%")
        print(f"Best Day:               {stats['Best Day']:>12.2%}")
        print(f"Worst Day:              {stats['Worst Day']:>12.2%}")

        print("=" * 60)

    def attribution_analysis(self):
        """
        Analyze contribution of each asset to portfolio performance.

        Returns:
        --------
        pandas.DataFrame : Attribution metrics for each asset
        """
        # Calculate weighted returns for each asset
        weighted_returns = self.returns * self.weights

        # Total contribution to return
        total_contribution = weighted_returns.sum()
        contribution_pct = (total_contribution / total_contribution.sum()) * 100

        # Create attribution dataframe
        attribution = pd.DataFrame({
            'Weight': self.weights,
            'Asset Return': self.returns.mean() * 252,  # Annualized
            'Contribution to Return': total_contribution,
            'Contribution %': contribution_pct
        })

        attribution = attribution.sort_values('Contribution to Return', ascending=False)

        print("\n" + "=" * 60)
        print("ATTRIBUTION ANALYSIS")
        print("=" * 60)
        print(attribution.to_string())
        print("=" * 60)

        return attribution

    def correlation_analysis(self):
        """
        Analyze correlations between assets.

        Returns:
        --------
        pandas.DataFrame : Correlation matrix
        """
        corr_matrix = self.returns.corr()

        print("\n" + "=" * 60)
        print("CORRELATION MATRIX")
        print("=" * 60)
        print(corr_matrix.to_string())
        print("=" * 60)

        # Calculate average correlation
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        avg_corr = corr_matrix.where(mask).stack().mean()
        print(f"\nAverage Pairwise Correlation: {avg_corr:.3f}")

        return corr_matrix

    def rolling_metrics(self, window=252):
        """
        Calculate rolling performance metrics.

        Parameters:
        -----------
        window : int, optional
            Rolling window in days (default: 252 = 1 year)

        Returns:
        --------
        pandas.DataFrame : Rolling metrics over time
        """
        rolling_return = self.portfolio_returns.rolling(window).mean() * 252
        rolling_vol = self.portfolio_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return - self.risk_free_rate) / rolling_vol

        rolling_df = pd.DataFrame({
            'Rolling Return': rolling_return,
            'Rolling Volatility': rolling_vol,
            'Rolling Sharpe': rolling_sharpe
        })

        return rolling_df

    def plot_performance(self):
        """Plot comprehensive performance charts."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Portfolio value over time
        ax1 = axes[0, 0]
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, linewidth=2)
        ax1.set_title('Portfolio Value Over Time', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value (Base 100)')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown chart
        ax2 = axes[0, 1]
        drawdown = self._drawdown_series()
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_title('Drawdown Over Time', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Returns distribution
        ax3 = axes[1, 0]
        ax3.hist(self.portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(self.portfolio_returns.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {self.portfolio_returns.mean():.4f}')
        ax3.set_title('Daily Returns Distribution', fontweight='bold')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Rolling Sharpe ratio
        ax4 = axes[1, 1]
        rolling = self.rolling_metrics(window=252)
        ax4.plot(rolling.index, rolling['Rolling Sharpe'], linewidth=2)
        ax4.axhline(0, color='black', linestyle='--', linewidth=1)
        ax4.set_title('Rolling Sharpe Ratio (1 Year)', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap."""
        corr_matrix = self.returns.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Asset Correlation Heatmap', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_attribution(self):
        """Plot attribution analysis."""
        attribution = self.attribution_analysis()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Weights pie chart
        ax1 = axes[0]
        ax1.pie(attribution['Weight'], labels=attribution.index, autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Portfolio Weights', fontweight='bold')

        # Contribution bar chart
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in attribution['Contribution to Return']]
        ax2.barh(attribution.index, attribution['Contribution to Return'], color=colors)
        ax2.set_title('Contribution to Total Return', fontweight='bold')
        ax2.set_xlabel('Contribution')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.show()

    # Helper methods for calculations
    def _total_return(self):
        """Calculate total return."""
        return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1

    def _annualized_return(self):
        """Calculate annualized return."""
        total_return = self._total_return()
        n_years = len(self.portfolio_returns) / 252
        return (1 + total_return) ** (1 / n_years) - 1

    def _annualized_volatility(self):
        """Calculate annualized volatility."""
        return self.portfolio_returns.std() * np.sqrt(252)

    def _drawdown_series(self):
        """Calculate drawdown series."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def _max_drawdown(self):
        """Calculate maximum drawdown."""
        return self._drawdown_series().min()

    def _value_at_risk(self, confidence=0.95):
        """Calculate Value at Risk."""
        return np.percentile(self.portfolio_returns, (1 - confidence) * 100)

    def _conditional_var(self, confidence=0.95):
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self._value_at_risk(confidence)
        return self.portfolio_returns[self.portfolio_returns <= var].mean()

    def _sharpe_ratio(self):
        """Calculate Sharpe ratio."""
        excess_return = self._annualized_return() - self.risk_free_rate
        return excess_return / self._annualized_volatility()

    def _sortino_ratio(self):
        """Calculate Sortino ratio (uses downside deviation)."""
        excess_return = self._annualized_return() - self.risk_free_rate
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        return excess_return / downside_std if downside_std > 0 else 0

    def _calmar_ratio(self):
        """Calculate Calmar ratio."""
        annual_return = self._annualized_return()
        max_dd = abs(self._max_drawdown())
        return annual_return / max_dd if max_dd > 0 else 0


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # This assumes you have price data and weights
    # Example: prices = get_market_close_data(...)
    #          portfolio = constructor.maximum_sharpe()
    #          weights = portfolio['weights']

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

    # Example weights
    weights = pd.Series([0.4, 0.35, 0.25], index=tickers)

    # Initialize analyzer
    analyzer = PortfolioAnalyzer(prices, weights, risk_free_rate=0.02)

    # Print summary
    analyzer.print_summary()

    # Attribution analysis
    analyzer.attribution_analysis()

    # Correlation analysis
    analyzer.correlation_analysis()

    # Plot performance
    analyzer.plot_performance()

    # Plot correlation heatmap
    analyzer.plot_correlation_heatmap()

    # Plot attribution
    analyzer.plot_attribution()