

from download_data import get_market_close_data
from portfolio_constructor import PortfolioConstructor
from portfolio_backtester import PortfolioBacktester
import pandas as pd
import os


# Configuration
LOAD_FROM_EXCEL = False  # Set to True to load from Excel, False to fetch fresh data
EXCEL_FILE = 'data/close_prices_XAMS_2005-01-01_to_2025-12-31.csv'

# Ticker definitions with descriptions for clarity

TICKERS =   {
    'L0CK.DE' : 'Digital Security' ,
    'MVOL.MI' : 'World Minimum Volatility' ,
    'IS3R.DE' : 'World Momentum' ,
    'IMIE.PA' : 'All Country World Investable Market' ,
    'DFEN.DE' : 'Defense VanEck' ,
    'XAIX.DE' : 'Artificial Intelligence & Big Data' ,
    'XDEQ.DE' : 'World Quality' ,
    'IS3N.DE' : 'Emerging Market' ,
    'IS3S.DE' : 'World Value' ,
    'IEVL.MI' : 'Europe Value' ,
    'IEMO.MI' : 'Europe Momentum' ,
    'VPN.MI' : 'Data Center REITs & Digital Infrastructure ' ,
    'GRID.DE' : 'Clean Edge Smart Grid Infrastructure'
}

# Load or fetch data
if LOAD_FROM_EXCEL and os.path.exists(EXCEL_FILE):
    print(f"Loading data from {EXCEL_FILE}...")
    data = pd.read_csv(EXCEL_FILE, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(data)} rows, {len(data.columns)} tickers")
else:
    if LOAD_FROM_EXCEL:
        print(f"Warning: {EXCEL_FILE} not found. Fetching fresh data...")
    else:
        print("Fetching fresh market data...")

    # Fetch market data
    data = get_market_close_data(
        start_date='2005-01-01',
        exchange='XAMS',
        tickers=list(TICKERS.keys()),
        save_to_csv=True
    )

#get a subset of data
final_portfolio_assets = ['IMIE.PA', 'IS3N.DE', 'XDEQ.DE', 'IEVL.MI', 'IEMO.MI', 'DFEN.DE','XAIX.DE', 'VPN.MI','GRID.DE']

filtered_tickers = {k: v for k, v in TICKERS.items() if k in
                    final_portfolio_assets}

data_subset = data[final_portfolio_assets]
portfolio = PortfolioConstructor(data_subset)

hrp_portfolio = portfolio.hierarchical_risk_parity(show_dendrogram=True,ticker_names=filtered_tickers)

bl_2026_corrected_views = {
    'IMIE.PA': 0.10,
    'IS3N.DE': 0.14,
    'IEVL.MI': 0.13,  # Europe Value (per your implementation)
    'XDEQ.DE': 0.09,
    'IEMO.MI': 0.11,
    'XAIX.DE': 0.12,
    'DFEN.DE': 0.18,
    'VPN.MI': 0.16,
}

bl_2026_corrected_confidences = {
    'IMIE.PA': 0.85,
    'IS3N.DE': 0.85,
    'IEVL.MI': 0.85,  # ↑ Higher due to proven track record
    'XDEQ.DE': 0.85,
    'IEMO.MI': 0.75,  # ↑ Higher due to accelerating trend
    'XAIX.DE': 0.60,  # ↓ Lower due to valuation concerns
    'DFEN.DE': 0.95,
    'VPN.MI': 0.90,
}


def bl_2026_document_strategy(constructor):
    """
    Black-Litterman implementation of 2026 allocation document

    """
    return constructor.black_litterman(
        views=bl_2026_corrected_views,
        view_confidences=bl_2026_corrected_confidences,
        risk_aversion=1.5
    )


comparison1 = PortfolioBacktester.compare_individual_assets(
       prices=data,
       tickers_dict=TICKERS,
       save_to_excel=True,              # ← Enable Excel export
       output_folder='individual_assets' # ← Custom folder
)

comparison2 = PortfolioBacktester.compare_individual_assets(
       prices=data_subset,
       tickers_dict=filtered_tickers,
       save_to_excel=True,              # ← Enable Excel export
       output_folder='individual_assets' # ← Custom folder
)


# Create backtesters
bl_document_bt = PortfolioBacktester(
    data_subset,
    strategy=bl_2026_document_strategy,
    rebalance_freq='yearly',
    min_history=252,
    ticker_names=filtered_tickers
)
results = bl_document_bt.run()
bl_document_bt.print_results()
bl_document_bt.plot_results()

hrp_bt = PortfolioBacktester(
    data_subset,
    strategy='hrp',
    rebalance_freq='yearly',
    min_history=252,
    ticker_names=filtered_tickers
)
results2 = hrp_bt.run()
hrp_bt.print_results()
hrp_bt.plot_results()

rubust_bt = PortfolioBacktester(
    data_subset,
    strategy='robust_mv',
    rebalance_freq='yearly',
    min_history=252,
    ticker_names=filtered_tickers
)
results3 = rubust_bt.run()
rubust_bt.print_results()
rubust_bt.plot_results()





# # Define all portfolios in a structured way
PORTFOLIOS = {
    'Old Portfolio': {
        'IMIE.PA': 0.4,
        'MVOL.MI': 0.4*1/3,
        'IS3R.DE': 0.4*1/3,
        'XDEQ.DE': 0.4*1/3,
        'L0CK.DE': 0.2*1/3,
        'XAIX.DE': 0.2*1/3,
        'DFEN.DE': 0.2*1/3,
     },
     'Market Portfolio': {
         'IMIE.PA': 1,
     },
     'Emerging Market Portfolio': {
        'IS3N.DE': 1,
    },
     'Old Factor Portfolio': {
         'MVOL.MI': 1/3,
         'IS3R.DE': 1/3,
         'XDEQ.DE': 1/3,
     },
     'Old Thema Portfolio': {
         'L0CK.DE': 1/3,
         'XAIX.DE': 1/3,
         'DFEN.DE': 1/3,
     },
    'New Factor Portfolio': {
        'XDEQ.DE': 1 / 3 ,
        'IEVL.MI': 1 / 3 ,
        'IEMO.MI': 1 / 3 ,
     },
    'New Thema Portfolio': {
        'DFEN.DE': 1 / 4 ,
        'XAIX.DE': 1 / 4 ,
        'VPN.MI': 1 / 4 ,
        'GRID.DE': 1 / 4,
    },
     'Sleeve Equal Weight': {
         'IMIE.PA': 0.88*1/3,
         'IS3N.DE': 0.12*1/3,
         'XDEQ.DE': 1/3*1/3,
         'IEVL.MI': 1/3*1/3,
         'IEMO.MI': 1/3*1/3,
         'DFEN.DE': 1/4*1/3,
         'XAIX.DE': 1/4*1/3,
         'VPN.MI': 1/4*1/3,
         'GRID.DE': 1/4*1/3,
     },
}

#
 # Common backtest settings
BACKTEST_CONFIG = {
     'rebalance_method': 'threshold',
     'rebalance_threshold': 0.02,
     'min_history': 252,
     'use_expanding_window': True,
     'transaction_cost': 0.01
}

BACKTEST_CONFIG2 = {
     'rebalance_freq' : 'yearly',
     'min_history': 252,
     'use_expanding_window': True,
     'transaction_cost': 0.01
}

# # Create backtesters for custom portfolios
backtesters = []
labels = {}
#
for i, (name, weights) in enumerate(PORTFOLIOS.items()):
     bt = PortfolioBacktester(
         data,
         strategy='custom',
         custom_weights=weights,
         **BACKTEST_CONFIG  # Unpack common settings
     )
     backtesters.append(bt)
     labels[i] = name

 # Add optimized portfolios
backtesters.extend([
     PortfolioBacktester(data_subset, strategy='equal_weight', **BACKTEST_CONFIG2),
     PortfolioBacktester(data_subset, strategy='hrp', **BACKTEST_CONFIG2),
     PortfolioBacktester(data_subset, strategy='robust_mv', **BACKTEST_CONFIG2),
     bl_document_bt
])
#
labels[len(backtesters)-4] = 'Assets Equal Weight'
labels[len(backtesters)-3] = 'Hierarchical Risk Parity'
labels[len(backtesters)-2] = 'Robust Mean Variance / Max Sharpe Portfolio'
labels[len(backtesters)-1] = 'Black Litterman'

# Run comparison
comparison = PortfolioBacktester.compare_strategies(
     backtesters=backtesters,
     custom_labels=labels,
     save_to_excel=True,
     output_folder='portfolio_comparison'
)
#


backtesters2 = [  backtesters[0],
                    backtesters[6],
                  PortfolioBacktester(data_subset, strategy='equal_weight', **BACKTEST_CONFIG2),
     PortfolioBacktester(data_subset, strategy='hrp', **BACKTEST_CONFIG2),
     PortfolioBacktester(data_subset, strategy='robust_mv', **BACKTEST_CONFIG2),
     bl_document_bt]

labels2 = [ 'Current portfolio',
            'Sleeve Equal Weight',
            'Assets Equal Weight',
           'Hierarchical Risk Parity',
           'Robust Mean Variance / Max Sharpe Portfolio',
           'Black Litterman']
labels2_dict = {i: label for i, label in enumerate(labels2)}

comparison3 = PortfolioBacktester.compare_strategies(
     backtesters=backtesters2,
     custom_labels=labels2_dict,
     save_to_excel=True,
     output_folder='portfolio_comparison'
)