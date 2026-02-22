"""
Portfolio Backtester
Backtest portfolio strategies with:
- Multiple rebalancing frequencies
- Transaction costs
- Strategy comparison
- Rolling optimization
- Performance tracking over time
"""
import datetime
import os

import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Agg')  # Non-blocking backend for PyCharm
import matplotlib.pyplot as plt
from portfolio_constructor import PortfolioConstructor


class PortfolioBacktester:
    """
    Backtest portfolio strategies over time with rebalancing.

    Parameters:
    -----------
    prices : pandas.DataFrame
        DataFrame with dates as index and tickers as columns
    strategy : str or callable
        Strategy name ('equal_weight', 'min_variance', 'max_sharpe')
        or custom function that returns weights
    rebalance_method : str, optional
        Rebalancing method: 'scheduled', 'threshold', or 'both'
        - 'scheduled': Only rebalance on fixed schedule (e.g., monthly)
        - 'threshold': Only rebalance when weights drift beyond threshold
        - 'both': Rebalance on schedule OR when threshold is breached
        (default: 'scheduled')
    rebalance_freq : str, optional
        Rebalancing frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
        Only used if rebalance_method is 'scheduled' or 'both'
        (default: 'monthly')
    min_history : int, optional
        Minimum number of days needed before starting backtest (default: 252 = 1 year)
    use_expanding_window : bool, optional
        If True, uses all data from start (expanding window).
        If False, uses only last min_history days (rolling window).
        (default: True)
    rebalance_threshold : float, optional
        Rebalance if any weight deviates more than this from target (in percentage points)
        For example, 0.02 = rebalance if weight differs by more than 2 percentage points
        Only used if rebalance_method is 'threshold' or 'both'
        (default: 0.02)
    transaction_cost : float, optional
        Transaction cost as percentage of trade value (default: 0.001 = 0.1%)
    initial_capital : float, optional
        Starting portfolio value (default: 100)
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 = 2%)
    """

    def __init__(self, prices, strategy='equal_weight', rebalance_method='scheduled',
                 rebalance_freq='monthly', min_history=252, use_expanding_window=True,
                 rebalance_threshold=0.02, transaction_cost=0.001,
                 initial_capital=100, risk_free_rate=0.02, custom_weights=None,ticker_names=None):

        self.prices = prices
        self.strategy = strategy
        self.rebalance_method = rebalance_method
        self.rebalance_freq = rebalance_freq
        self.min_history = min_history
        self.use_expanding_window = use_expanding_window
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.custom_weights = custom_weights
        self.ticker_names = ticker_names or {}

        # Validate strategy and custom_weights
        if strategy == 'custom' and custom_weights is None:
            raise ValueError("custom_weights must be provided when strategy='custom'")

        # Validate rebalance method
        valid_methods = ['scheduled', 'threshold', 'both']
        if rebalance_method not in valid_methods:
            raise ValueError(f"rebalance_method must be one of {valid_methods}")

        # Results will be stored here
        self.portfolio_value = None
        self.weights_history = None
        self.rebalance_dates = None
        self.transaction_costs = None

        print(f"Backtester initialized")
        print(f"Strategy: {strategy}")
        if strategy == 'custom':
            print(f"  Custom weights: {len(custom_weights)} assets")
        print(f"Rebalance method: {rebalance_method}")
        if rebalance_method in ['scheduled', 'both']:
            print(f"  - Frequency: {rebalance_freq}")
        if rebalance_method in ['threshold', 'both']:
            print(f"  - Threshold: {rebalance_threshold:.1%}")
        print(f"Window type: {'Expanding' if use_expanding_window else 'Rolling'}")
        print(f"Min history: {min_history} days")
        print(f"Transaction cost: {transaction_cost:.2%}")


    def run(self):
        """
        Run the backtest.

        Returns:
        --------
        dict : Dictionary with backtest results
        """
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)

        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates()

        # Check if we have enough data
        if self.min_history >= len(self.prices):
            print(f"ERROR: min_history ({self.min_history}) is >= data length ({len(self.prices)})")
            print(f"Backtest cannot run. Reduce min_history or use more data.")
            return None

        print(f"Total data points: {len(self.prices)}")
        print(f"Starting backtest from index {self.min_history} (date: {self.prices.index[self.min_history].date()})")
        #print(f"Potential scheduled rebalance dates: {len(rebalance_dates)}")

        # Initialize tracking variables
        backtest_index = self.prices.index[self.min_history:]  # Only backtest dates
        portfolio_value = pd.Series(index=backtest_index, dtype=float)
        portfolio_value.iloc[0] = self.initial_capital

        #portfolio_value = pd.Series(index=self.prices.index, dtype=float)
        #portfolio_value.iloc[0] = self.initial_capital

        weights_history = []
        transaction_costs_total = 0
        current_weights = None

        # INITIALIZE portfolio at start of backtest period
        # Get initial weights using first available historical data
        initial_hist_prices = self.prices.iloc[0:self.min_history]
        initial_weights = self._calculate_weights(initial_hist_prices)

        # Set up initial portfolio
        current_weights = initial_weights

        weights_history.append({
            'date': self.prices.index[self.min_history],
            'weights': current_weights.copy(),
            'reason': 'initial'
        })

        print(f"  Initial portfolio set at index {self.min_history} ({self.prices.index[self.min_history].date()})")

       # Main backtest loop - start after minimum history period
        for i in range(1, len(portfolio_value)):
            current_date = portfolio_value.index[i]

            # Calculate daily return based on current weights
            daily_returns = self.prices.iloc[self.min_history+i] / self.prices.iloc[self.min_history+i-1] - 1
            portfolio_return = (daily_returns * current_weights).sum()

            # Update portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)

            # Calculate actual weights after drift (for threshold checking)
            # Weights drift due to different asset returns
            asset_values = current_weights * portfolio_value.iloc[i-1] * (1 + daily_returns)
            actual_weights = asset_values / asset_values.sum()

            # Determine if rebalancing is needed
            should_rebalance = False
            rebalance_reason = None

            # Check based on rebalancing method
            if self.rebalance_method == 'scheduled':
                # Only rebalance on scheduled dates
                if current_date in rebalance_dates:
                    should_rebalance = True
                    rebalance_reason = "scheduled"

            elif self.rebalance_method == 'threshold':
                # Only rebalance when threshold is breached
                weight_deviation = abs(actual_weights - current_weights).max()
                if weight_deviation > self.rebalance_threshold:
                    should_rebalance = True
                    rebalance_reason = f"threshold ({weight_deviation:.2%} deviation)"

            elif self.rebalance_method == 'both':
                # Rebalance on schedule OR when threshold is breached
                if current_date in rebalance_dates:
                    should_rebalance = True
                    rebalance_reason = "scheduled"
                else:
                    weight_deviation = abs(actual_weights - current_weights).max()
                    if weight_deviation > self.rebalance_threshold:
                        should_rebalance = True
                        rebalance_reason = f"threshold ({weight_deviation:.2%} deviation)"

            # Rebalance if needed
            if should_rebalance:
                hist_end = self.prices.index.get_loc(current_date)
                # Get historical data for optimization
                if self.use_expanding_window:
                    # Expanding window: use all data from start to current
                    hist_prices = self.prices.iloc[0:hist_end]
                else:
                    # Rolling window: use last min_history days
                    lookback_start = max(0, hist_end  - self.min_history)
                    hist_prices = self.prices.iloc[lookback_start:(self.min_history+i)]

                # Calculate new target weights
                new_weights = self._calculate_weights(hist_prices)

                # Calculate costs BEFORE growth
                weight_change = abs(new_weights - actual_weights).sum()
                txn_cost = weight_change * self.transaction_cost * portfolio_value.iloc[i - 1]  # Use previous value!
                portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + portfolio_return)  # First growth
                portfolio_value.iloc[i] -= txn_cost  # Then deduct costs
                transaction_costs_total += txn_cost

                # Update weights
                current_weights = new_weights

                weights_history.append({
                    'date': current_date,
                    'weights': current_weights.copy(),
                    'reason': rebalance_reason
                })

                print(f"  Rebalanced on {current_date.date()} - Reason: {rebalance_reason}")

        # Store results
        self.portfolio_value = portfolio_value

        # Handle weights history - check if any rebalances occurred
        if weights_history:
            self.weights_history = pd.DataFrame([
                {'date': w['date'], **w['weights'].to_dict()}
                for w in weights_history
            ]).set_index('date')
        else:
            # No rebalances occurred - create empty DataFrame with proper structure
            print("Warning: No rebalances occurred during backtest period")
            self.weights_history = pd.DataFrame(columns=['date'] + list(self.prices.columns))
            self.weights_history = self.weights_history.set_index('date')

        self.rebalance_dates = rebalance_dates
        self.transaction_costs = transaction_costs_total

        print(f"✓ Backtest complete")
        print(f"Total rebalances executed: {len(weights_history)}")
        print(f"Total transaction costs: ${transaction_costs_total:.2f}")

        # Calculate performance metrics
        if len(weights_history) > 0:
            results = self._calculate_results()
        else:
            print("\nWarning: No rebalances occurred. Cannot calculate full results.")
            print("Possible reasons:")
            print("  - Threshold never breached (try lower threshold)")
            print("  - No scheduled rebalance dates in range (try longer date range)")
            print("  - min_history too large relative to data length")
            results = None

        return results


    def _get_rebalance_dates(self):
        """Determine rebalancing dates based on frequency."""
        dates = self.prices.index

        if self.rebalance_freq == 'daily':
            return dates
        elif self.rebalance_freq == 'weekly':
            return dates[dates.to_series().dt.dayofweek == 0]
        elif self.rebalance_freq == 'monthly':
            return dates[dates.to_series().dt.is_month_end]
        elif self.rebalance_freq == 'quarterly':
            return dates[dates.to_series().dt.is_quarter_end]
        elif self.rebalance_freq == 'yearly':
            return dates[dates.to_series().dt.is_year_end]
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_freq}")


    def _calculate_weights(self, hist_prices):
        """Calculate portfolio weights based on strategy."""
        constructor = PortfolioConstructor(hist_prices, self.risk_free_rate)

        if self.strategy == 'equal_weight':
            result = constructor.equal_weight()
        elif self.strategy == 'min_variance':
            result = constructor.minimum_variance()
        elif self.strategy == 'max_sharpe':
            result = constructor.maximum_sharpe()
        elif self.strategy == 'black_litterman':
            # Can pass views via custom_weights if needed
            result = constructor.black_litterman()
        elif self.strategy == 'hrp':
            result = constructor.hierarchical_risk_parity()
        elif self.strategy == 'robust_mv':
            result = constructor.robust_mean_variance()
        elif self.strategy == 'custom':
            result = constructor.custom_weights(self.custom_weights)
        elif callable(self.strategy):
            result = self.strategy(constructor)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return result['weights']


    def _calculate_results(self):
        """Calculate comprehensive backtest results."""
        returns = self.portfolio_value.pct_change().dropna()


        if hasattr(self.weights_history, 'empty') and not self.weights_history.empty:
            actual_rebalances = len(self.weights_history) - 1  # Subtract initial portfolio setup
        else:
            actual_rebalances = 0

            # Calculate yearly returns
        yearly_values = self.portfolio_value.resample('YE').last().dropna()
        yearly_returns = yearly_values.pct_change().dropna()
        yearly_returns.index = yearly_returns.index.year  # Extract year as int
        yearly_returns_dict = yearly_returns.to_dict()

        yearly_stats = {
            'years': list(yearly_returns.index),  # [2020, 2021, 2022, ...] - now integers!
            'avg_yearly_return': yearly_returns.mean() if len(yearly_returns) > 0 else np.nan,
            'best_year_return': yearly_returns.max() if len(yearly_returns) > 0 else np.nan,
            'best_year_date': int(yearly_returns.idxmax()) if len(yearly_returns) > 0 else np.nan,
            'worst_year_return': yearly_returns.min() if len(yearly_returns) > 0 else np.nan,
            'worst_year_date': int(yearly_returns.idxmin()) if len(yearly_returns) > 0 else np.nan,
            'win_years': int((yearly_returns > 0).sum()) if len(yearly_returns) > 0 else 0,
            'total_years': len(yearly_returns),
            'win_rate': (yearly_returns > 0).mean() if len(yearly_returns) > 0 else 0,
            'yearly_returns': yearly_returns_dict  # {2020: 0.1234, 2021: -0.0567, ...}
        }

        results = {
            'total_return': (self.portfolio_value.iloc[-1] / self.initial_capital) - 1,
            'annualized_return': ((self.portfolio_value.iloc[-1] / self.initial_capital) **
                                 (252 / len(returns))) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': ((returns.mean() * 252 - self.risk_free_rate) /
                           (returns.std() * np.sqrt(252))),
            'max_drawdown': self._calculate_max_drawdown(),
            'num_rebalances': actual_rebalances,
            'transaction_costs': self.transaction_costs,
            'final_value': self.portfolio_value.iloc[-1],
            'years': yearly_stats['years'],
            'avg_yearly_return': yearly_stats['avg_yearly_return'],
            'best_year_return': yearly_stats['best_year_return'],
            'best_year_date': yearly_stats['best_year_date'],
            'worst_year_return': yearly_stats['worst_year_return'],
            'worst_year_date': yearly_stats['worst_year_date'],
            'win_years': yearly_stats['win_years'],
            'total_years': yearly_stats['total_years'],
            'win_rate': yearly_stats['win_rate'],
            'yearly_returns': yearly_stats['yearly_returns']  # Full yearly breakdown
        }

        return results


    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown."""
        cumulative = self.portfolio_value / self.portfolio_value.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def print_results(self):
        """Print formatted backtest results with full yearly stats."""
        results = self._calculate_results()

        if results is None:
            print("No results to display")
            return

        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Strategy:              {self.strategy}")
        print(f"Rebalance Method:      {self.rebalance_method}")
        print("-" * 70)
        print(f"Total Return:          {results['total_return']:>12.2%}")
        print(f"Annualized Return:     {results['annualized_return']:>12.2%}")
        print(f"Avg Yearly Return:     {results['avg_yearly_return']:>12.2%}")
        print(f"Best Year:             {results['best_year_return']:>12.2%} ({int(results['best_year_date'])})")
        print(f"Worst Year:            {results['worst_year_return']:>12.2%} ({int(results['worst_year_date'])})")
        print(
            f"Win Rate:              {results['win_rate']:>12.1%} ({results['win_years']}/{results['total_years']} years)")
        print(f"Volatility:            {results['volatility']:>12.2%}")
        print(f"Sharpe Ratio:          {results['sharpe_ratio']:>12.3f}")
        print(f"Max Drawdown:          {results['max_drawdown']:>12.2%}")
        print("-" * 70)

        # Show actual yearly returns
        print("YEARLY RETURNS:")
        for year, ret in results['yearly_returns'].items():
            print(f"  {int(year)}: {ret:>8.2%}")

        print("=" * 70)


    def plot_results(self,output_folder='output'):
        """Plot backtest results."""
        if len(self.weights_history) == 1:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        else:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 1. Portfolio value over timeplot_results
        ax1 = axes[0]
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, linewidth=2)

        # # Mark rebalance dates
        # rebal_values = self.weights_history.index if hasattr(self.weights_history, 'index') else []
        # ax1.scatter(rebal_values.index, rebal_values.values,
        #            color='red', s=30, alpha=0.5, label='Rebalance')

        ax1.set_title(f'Portfolio Value - {self.strategy} ({self.rebalance_freq})',
                     fontweight='bold', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = axes[1]
        cumulative = self.portfolio_value / self.portfolio_value.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Weight evolution (stacked area)
        ax3 = axes[2]
        if self.weights_history is not None and len(self.weights_history) > 0:
            if len(self.weights_history) == 1:
                # SINGLE entry → Bar chart
                weights = self.weights_history.iloc[0].values #self.weights_history.values
                asset_names = self.weights_history.columns[0:].tolist()
                friendly_names = [self.ticker_names.get(t, t) for t in asset_names]
                x_pos = np.arange(len(weights))

                ax3.bar(x_pos, weights, alpha=0.7, color=plt.cm.Set3(np.linspace(0, 1, len(weights))))
                ax3.set_title('Portfolio Weights (Final Allocation)', fontweight='bold', fontsize=12)
                ax3.set_ylabel('Weight')
                ax3.set_ylim([0, 1])
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(friendly_names, rotation=45, ha='right')
                ax3.grid(True, alpha=0.3)
            else:
                weights_renamed = self.weights_history.rename(columns=self.ticker_names)
                weights_renamed.plot.area(ax=ax3, stacked=True, alpha=0.7)
                ax3.set_title('Portfolio Weights Over Time', fontweight='bold', fontsize=12)
                ax3.set_ylabel('Weight')
                ax3.set_ylim([0, 1])
                ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        os.makedirs(output_folder, exist_ok=True)
        strategy_name = (getattr(self.strategy, '__name__', str(self.strategy))
                         if callable(self.strategy) else self.strategy)
        figure_filename = os.path.join(output_folder, f'{strategy_name} ({self.rebalance_freq}).png')
        plt.savefig(figure_filename, dpi=300, bbox_inches='tight')

        plt.show(block=False)


    def compare_strategies(prices=None, strategies=None, backtesters=None,
                           rebalance_method='scheduled', rebalance_freq='monthly',
                           min_history=252, use_expanding_window=True,
                           rebalance_threshold=0.02, transaction_cost=0.001,
                           custom_labels=None, save_to_excel=False, output_folder='output'):

        full_results = []
        portfolio_values = {}
        name_mapping = {}  # original_name -> display_name

        print("\n" + "=" * 60)
        print("COMPARING STRATEGIES")
        print("=" * 60)

        custom_count = 1

        # PRIORITY 1: Use pre-configured backtesters
        if backtesters:
            for i, backtester in enumerate(backtesters):
                print(f"\nRunning pre-configured backtester {i + 1}...")
                result = backtester.run()

                if result is None:
                    print(f"Warning: Backtester {i + 1} failed to run. Skipping.")
                    continue

                orig_name = f"Custom_{custom_count}" if backtester.strategy == 'custom' else backtester.strategy
                if backtester.strategy == 'custom':
                    custom_count += 1

                # Apply custom label
                display_name = orig_name
                if custom_labels:
                    if i in custom_labels:
                        display_name = custom_labels[i]
                    elif orig_name in custom_labels:
                        display_name = custom_labels[orig_name]

                name_mapping[orig_name] = display_name
                portfolio_values[display_name] = backtester.portfolio_value
                full_results.append((orig_name, result))

        # PRIORITY 2: Fallback to strategies mode
        elif strategies and prices is not None:
            for strategy in strategies:
                print(f"\nRunning {strategy}...")
                backtester = PortfolioBacktester(
                    prices, strategy=strategy,
                    rebalance_method=rebalance_method, rebalance_freq=rebalance_freq,
                    min_history=min_history, use_expanding_window=use_expanding_window,
                    rebalance_threshold=rebalance_threshold, transaction_cost=transaction_cost
                )
                result = backtester.run()

                if result is None:
                    print(f"Warning: Strategy {strategy} failed to run. Skipping.")
                    continue

                display_name = custom_labels[strategy] if custom_labels and strategy in custom_labels else strategy
                name_mapping[strategy] = display_name
                portfolio_values[display_name] = backtester.portfolio_value
                full_results.append((strategy, result))

        else:
            raise ValueError("Must provide either 'backtesters' OR 'prices+strategies'")

        # *** SIMPLE, BULLETPROOF comparison_df building ***
        summary_data = []
        yearly_matrix = {}

        for orig_name, res in full_results:
            display_name = name_mapping[orig_name]

            # Store yearly returns with ORIGINAL name
            yearly_matrix[orig_name] = res['yearly_returns']

            # Build summary row with DISPLAY name
            summary_data.append({
                'Strategy': display_name,
                'Total Return': res['total_return'],
                'Annual Return': res['annualized_return'],
                'Avg Yearly': res['avg_yearly_return'],
                'Best Year': f"{res['best_year_return']:.1%} ({int(res['best_year_date'])})",
                'Worst Year': f"{res['worst_year_return']:.1%} ({int(res['worst_year_date'])})",
                'Win Rate': f"{res['win_rate']:.0%}",
                'Years': res['total_years'],
                'Volatility': res['volatility'],
                'Sharpe Ratio': res['sharpe_ratio'],
                'Max Drawdown': res['max_drawdown'],
                'Num Rebalances': res['num_rebalances'],
                'Transaction Costs': res['transaction_costs']
            })

        # Base summary DataFrame
        comparison_df = pd.DataFrame(summary_data)

        # *** SIMPLE yearly columns addition - NO COMPLEX MELT/PIVOT ***
        all_years = set()
        for yearly_returns in yearly_matrix.values():
            all_years.update(yearly_returns.keys())

        for year in sorted(all_years):
            year_col = f"{int(year)}"
            comparison_df[year_col] = None

            for orig_name in yearly_matrix:
                display_name = name_mapping[orig_name]
                if year in yearly_matrix[orig_name]:
                    idx = comparison_df[comparison_df['Strategy'] == display_name].index[0]
                    comparison_df.loc[idx, year_col] = yearly_matrix[orig_name][year]

        # Print PERFECT table
        print("\n" + "=" * 120)
        print("COMPREHENSIVE STRATEGY COMPARISON")
        print("=" * 120)
        print(comparison_df.round(4).astype(str).to_string(index=False))
        print("=" * 120)

        # Excel export (custom names work perfectly)
        if save_to_excel:
            import os
            from datetime import datetime

            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_filename = os.path.join(output_folder, f'strategy_comparison_{timestamp}.xlsx')

            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                comparison_df.to_excel(writer, sheet_name='Summary', index=False)
                pd.DataFrame(portfolio_values).to_excel(writer, sheet_name='Portfolio Values')

            print(f"\n✓ Results saved to: {excel_filename}")

        # Plot
        plt.figure(figsize=(12, 6))
        for strategy, values in portfolio_values.items():
            plt.plot(values.index, values.values, label=strategy, linewidth=2)
        plt.title('Strategy Comparison', fontweight='bold', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_to_excel:
            import os
            from datetime import datetime

            # Use same timestamp as Excel file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            figure_filename = os.path.join(output_folder, f'strategy_comparison_{timestamp}.png')
            plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to: {figure_filename}")

        plt.show(block=False)

        return comparison_df


    @staticmethod
    def compare_individual_assets(prices, tickers_dict, rebalance_method='scheduled',
                              rebalance_freq='yearly', min_history=252,
                              transaction_cost=0.0, save_to_excel=False,
                              output_folder='output'):
        """
        Compare performance of individual assets (100% allocation to each).

        Useful for comparing how individual stocks/ETFs perform versus portfolios.

        Parameters:
        -----------
        prices : pandas.DataFrame
            Price data with tickers as columns
        tickers_dict : dict
            Dictionary mapping ticker symbols to descriptive names
            Example: {'AAPL': 'Apple', 'MSFT': 'Microsoft'}
        rebalance_method : str, optional
            Rebalancing method (default: 'scheduled')
        rebalance_freq : str, optional
            Rebalancing frequency (default: 'yearly')
        min_history : int, optional
            Minimum history period (default: 252)
        transaction_cost : float, optional
            Transaction cost (default: 0.0 for buy-and-hold)
        save_to_excel : bool, optional
            If True, saves comparison to Excel (default: False)
        output_folder : str, optional
            Folder path for saving Excel file (default: 'output')

        Returns:
        --------
        pandas.DataFrame : Comparison of all individual assets

        Example:
        --------
        TICKERS = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google'
        }

        comparison = PortfolioBacktester.compare_individual_assets(
            prices=data,
            tickers_dict=TICKERS,
            rebalance_freq='yearly'
        )
        """
        backtesters = []
        custom_labels = {}

        print("\n" + "=" * 60)
        print("CREATING INDIVIDUAL ASSET PORTFOLIOS")
        print("=" * 60)

        for i, (ticker, name) in enumerate(tickers_dict.items()):
            # Check if ticker exists in price data
            if ticker not in prices.columns:
                print(f"⚠ Warning: {ticker} ({name}) not found in data. Skipping.")
                continue

            # Create 100% weight for this ticker
            weights = {t: 1.0 if t == ticker else 0.0 for t in prices.columns}

            # Create backtester
            bt = PortfolioBacktester(
                prices,
                strategy='custom',
                custom_weights=weights,
                rebalance_method=rebalance_method,
                rebalance_freq=rebalance_freq,
                min_history=min_history,
                transaction_cost=transaction_cost
            )

            backtesters.append(bt)
            custom_labels[i] = name  # Use descriptive name

            print(f"  ✓ Created portfolio for {name} ({ticker})")

        print(f"\n✓ Created {len(backtesters)} individual asset portfolios")

        # Compare all individual assets
        comparison = PortfolioBacktester.compare_strategies(
            backtesters=backtesters,
            custom_labels=custom_labels,
            save_to_excel=save_to_excel,
            output_folder=output_folder
        )

        return comparison

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # For demonstration, create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Simulate prices
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )

    # Example 1: Scheduled rebalancing only (monthly)
    print("\n" + "="*60)
    print("EXAMPLE 1: SCHEDULED REBALANCING")
    print("="*60)
    backtester1 = PortfolioBacktester(
        prices,
        strategy='max_sharpe',
        rebalance_method='scheduled',  # Only on schedule
        rebalance_freq='monthly',
        min_history=252,
        use_expanding_window=True,
        transaction_cost=0.001
    )
    results1 = backtester1.run()
    backtester1.print_results()
    backtester1.plot_results()

    # Example 2: Threshold rebalancing only (2% drift)
    print("\n" + "="*60)
    print("EXAMPLE 2: THRESHOLD REBALANCING")
    print("="*60)
    backtester2 = PortfolioBacktester(
        prices,
        strategy='max_sharpe',
        rebalance_method='threshold',  # Only on threshold breach
        rebalance_threshold=0.02,      # Rebalance if weights drift > 2%
        min_history=252,
        use_expanding_window=True,
        transaction_cost=0.001
    )
    results2 = backtester2.run()
    backtester2.print_results()
    backtester2.plot_results()

    # Example 3: Both methods (schedule OR threshold)
    print("\n" + "="*60)
    print("EXAMPLE 3: BOTH METHODS")
    print("="*60)
    backtester3 = PortfolioBacktester(
        prices,
        strategy='max_sharpe',
        rebalance_method='both',  # Rebalance on schedule OR threshold
        rebalance_freq='quarterly',
        rebalance_threshold=0.05,  # 5% drift threshold
        min_history=252,
        use_expanding_window=True,
        transaction_cost=0.001
    )
    results3 = backtester3.run()
    backtester3.print_results()
    backtester3.plot_results()

    # Example 4: Compare multiple strategies with threshold rebalancing
    print("\n" + "="*60)
    print("EXAMPLE 4: STRATEGY COMPARISON")
    print("="*60)
    strategies = ['equal_weight', 'min_variance', 'max_sharpe']
    comparison = PortfolioBacktester.compare_strategies(
        prices,
        strategies,
        rebalance_method='threshold',
        rebalance_threshold=0.03
    )