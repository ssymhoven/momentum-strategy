import pandas as pd
from matplotlib import pyplot as plt
from source_engine.opus_source import OpusSource
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import timedelta


def get_stocks():
    query = """
        SELECT
            positions.bloomberg_query,
            positions.name,
            positions.gics_industry_sector,
            positions.country_of_domicile,
            positions.percent_nav
        FROM
            reportings
                JOIN
            accountsegments ON (accountsegments.reporting_uuid = reportings.uuid)
                JOIN
            positions ON (reportings.uuid = positions.reporting_uuid)
        WHERE
                positions.account_segment_id = accountsegments.accountsegment_id
                        AND accountsegments.accountsegment_id = '17154631'
                AND reportings.newest = 1
                AND reportings.report = 'positions'
                AND positions.asset_class = 'STOCK'
                AND positions.bloomberg_query is not null
                AND reportings.report_date = (SELECT
                                                MAX(report_date)
                                              FROM
                                                reportings)
    """
    opus = OpusSource()

    df = opus.read_sql(query=query)
    df.set_index("bloomberg_query", inplace=True)

    return df


def get_data():
    benchmark_prices = pd.read_excel("beta.xlsx", sheet_name="Benchmark", header=0, index_col=0, parse_dates=True)
    benchmark_prices['SXXEWR_change'] = benchmark_prices['SXXP Index'].dropna().pct_change().dropna()
    benchmark_prices['SPXEWNTR_change'] = benchmark_prices['SPX Index'].dropna().pct_change().dropna()

    def weighted_combination(row):
        if pd.isna(row['SXXEWR_change']) and pd.isna(row['SPXEWNTR_change']):
            return pd.NA
        elif pd.isna(row['SXXEWR_change']):
            return row['SPXEWNTR_change'] * 100
        elif pd.isna(['SPXEWNTR_change']):
            return row['SXXEWR_change'] * 100
        else:
            return (0.6 * row['SXXEWR_change'] + 0.4 * row['SPXEWNTR_change']) * 100

    benchmark_prices['Benchmark'] = benchmark_prices.apply(weighted_combination, axis=1)
    benchmark_prices = benchmark_prices.dropna(subset=['Benchmark'])

    stock_prices = pd.read_excel("beta.xlsx", sheet_name="Stocks", header=0, index_col=0, parse_dates=True)

    for stock in stock_prices.columns:
        stock_data = stock_prices[stock].dropna()
        stock_data = stock_data.pct_change() * 100
        stock_data = stock_data.dropna()

        stock_prices[stock] = stock_data

    stock_prices = stock_prices.dropna(how='all')

    risk_free_rates = pd.read_excel("beta.xlsx", sheet_name="Risk Free Rates", header=0, index_col=0, parse_dates=True)

    return stock_prices, benchmark_prices, risk_free_rates


def calculate_alpha_beta(stock_returns, benchmark_returns):
    combined_data = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    clean_stock_returns = combined_data.iloc[:, 0]
    clean_benchmark_returns = combined_data.iloc[:, 1]

    X = clean_benchmark_returns.values.reshape(-1, 1)
    y = clean_stock_returns.values

    reg = LinearRegression().fit(X, y)

    beta = reg.coef_[0]
    alpha = reg.intercept_ * 250

    return alpha, beta


def plot_alpha_beta(stocks, timeframe):
    dr_aktien = stocks.loc[stocks.index == 'DRAKTIV GR Equity']

    stocks = stocks.drop(index=['DRAKTIV GR Equity'], errors='ignore')

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 3)

    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2], sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2], sharey=ax_main)

    ax_main.scatter(stocks['Beta'], stocks['Alpha'], s=stocks['percent_nav'] * 100, alpha=0.6)
    ax_main.scatter(dr_aktien['Beta'], dr_aktien['Alpha'], s=300, color='red', label='D&R Aktien',
                    edgecolor='red', alpha=0.6)

    ax_main.text(dr_aktien['Beta'], dr_aktien['Alpha'], "D&R Aktien", fontsize=9, ha='right')

    ax_main.set(xlabel="Beta", ylabel="Alpha")

    sns.kdeplot(stocks['Beta'], ax=ax_xDist, fill=True)
    ax_xDist.set(ylabel='Density')

    sns.kdeplot(y=stocks['Alpha'], ax=ax_yDist, fill=True)
    ax_yDist.set(xlabel='Density')

    weighted_beta = (stocks['Beta'] / 100 * stocks['percent_nav']).sum()
    total_pct_nav = stocks['percent_nav'].sum()
    ax_main.text(0.05, 0.95, f'Total % Equity: {total_pct_nav:.2f}%', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_main.text(0.05, 0.90, f'Weighted Beta: {weighted_beta:.2f}', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')

    ax_main.text(0.05, 0.10, f'D&R Aktien Beta: {dr_aktien["Beta"][0]:.2f}', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_main.text(0.05, 0.05, f'D&R Aktien (p.a): {dr_aktien["Alpha"][0]:.2f}%', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')

    ax_main.axhline(0, linestyle="--", color="black")
    ax_main.axvline(1, linestyle="--", color="black")

    plt.tight_layout()

    plt.savefig(f"images/alpha_vs_beta_{timeframe}_market_weight.png")


def filter_data_by_timeframe(df, years):
    end_date = df.index.max()
    start_date = end_date - timedelta(days=years * 365)
    return df.loc[start_date:end_date]


def plot_sector_correlation(stocks_df, stock_prices_df):
    sectors = stocks_df['gics_industry_sector'].unique()

    for sector in sectors:
        sector_stocks = stocks_df[stocks_df['gics_industry_sector'] == sector].index

        sector_stock_prices = stock_prices_df[sector_stocks]
        correlation_matrix = sector_stock_prices.corr()

        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix - {sector}')
        plt.tight_layout()
        plt.savefig(f'images/sector_correlation_{sector}.png')


if __name__ == '__main__':
    stocks = get_stocks()

    stock_prices, benchmark_prices, risk_free_rates = get_data()

    plot_sector_correlation(stocks, stock_prices)

    timeframes = {'1y': 1, '2y': 2, '3y': 3, '4y': 4}

    for label, years in timeframes.items():
        stock_prices_filtered = filter_data_by_timeframe(stock_prices, years)
        benchmark_prices_filtered = filter_data_by_timeframe(benchmark_prices, years)

        for stock in stock_prices_filtered.columns:
            if stock == "DRAKTIV GR Equity":
                stock_prices_filtered[stock] = stock_prices_filtered[stock].shift(-1)

            alpha, beta = calculate_alpha_beta(stock_prices_filtered[stock], benchmark_prices_filtered['Benchmark'])
            stocks.loc[stock, ["Alpha", "Beta"]] = [alpha, beta]

        plot_alpha_beta(stocks, label)
