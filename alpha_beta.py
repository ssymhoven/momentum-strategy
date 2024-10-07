import pandas as pd
from matplotlib import pyplot as plt
from source_engine.opus_source import OpusSource
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
import seaborn as sns


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
    benchmark_prices = pd.read_excel("beta.xlsx", sheet_name="Benchmark", header=0, index_col=0)
    benchmark_prices['SXXEWR_change'] = benchmark_prices['SXXEWR Index'].dropna().pct_change().dropna()
    benchmark_prices['SPXEWNTR_change'] = benchmark_prices['SPXEWNTR Index'].dropna().pct_change().dropna()

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

    stock_prices = pd.read_excel("beta.xlsx", sheet_name="Stocks", header=0, index_col=0)

    for stock in stock_prices.columns:
        stock_data = stock_prices[stock].dropna()
        stock_data = stock_data.pct_change() * 100
        stock_data = stock_data.dropna()

        stock_prices[stock] = stock_data

    stock_prices = stock_prices.dropna(how='all')

    risk_free_rates = pd.read_excel("beta.xlsx", sheet_name="Risk Free Rates", header=0, index_col=0)

    return stock_prices, benchmark_prices, risk_free_rates


def calculate_alpha_beta(stock_returns, benchmark_returns):
    combined_data = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    clean_stock_returns = combined_data.iloc[:, 0]
    clean_benchmark_returns = combined_data.iloc[:, 1]

    X = clean_benchmark_returns.values.reshape(-1, 1)
    y = clean_stock_returns.values

    reg = LinearRegression().fit(X, y)

    print(len(clean_stock_returns))

    beta = reg.coef_[0]
    alpha = reg.intercept_ * 250

    return alpha, beta


def plot_alpha_beta(stocks):
    dropped_row = stocks.loc[stocks.index == 'DRAKTIV GR Equity']
    stocks = stocks.drop(index='DRAKTIV GR Equity', errors='ignore')

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 3)

    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2], sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2], sharey=ax_main)

    ax_main.scatter(stocks['Beta'], stocks['Alpha'], s=stocks['percent_nav'] * 100, alpha=0.6)
    ax_main.scatter(dropped_row['Beta'], dropped_row['Alpha'], s=300, color='red', label='Portfolio (DRAKTIV GR Equity)',
                    edgecolor='red', alpha=0.6)
    ax_main.text(dropped_row['Beta'], dropped_row['Alpha'], "D&R Aktien", fontsize=9, ha='right')

    ax_main.set(xlabel="Beta", ylabel="Alpha")

    sns.kdeplot(stocks['Beta'], ax=ax_xDist, fill=True)
    ax_xDist.set(ylabel='Density')

    sns.kdeplot(stocks['Alpha'], ax=ax_yDist, fill=True, vertical=True)
    ax_yDist.set(xlabel='Density')

    for i, stock_name in enumerate(stocks.index):
        ax_main.text(stocks['Beta'][i], stocks['Alpha'][i], stock_name, fontsize=9, ha='right')

    # Add weighted beta and total percent_nav as text annotations
    weighted_beta = (stocks['Beta'] / 100 * stocks['percent_nav']).sum()
    total_pct_nav = stocks['percent_nav'].sum()
    ax_main.text(0.05, 0.95, f'Total % Equity: {total_pct_nav:.2f}%', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_main.text(0.05, 0.90, f'Weighted Beta: {weighted_beta:.2f}', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_main.text(0.05, 0.85, f'Portfolio Beta: {dropped_row["Beta"][0]:.2f}', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_main.text(0.05, 0.80, f'Portfolio Alpha annualized: {dropped_row["Alpha"][0]:.2f}%', transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top')

    # Add gridlines and central reference lines
    ax_main.axhline(0, linestyle="--", color="black")
    ax_main.axvline(1, linestyle="--", color="black")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.savefig("images/alpha_vs_beta_4y.png")


if __name__ == '__main__':
    stocks = get_stocks()
    stock_prices, benchmark_prices, risk_free_rates = get_data()

    for stock in stock_prices.columns:
        if stock == "DRAKTIV GR Equity":
            stock_prices[stock] = stock_prices[stock].shift(-1)

        alpha, beta = calculate_alpha_beta(stock_prices[stock], benchmark_prices['Benchmark'])
        stocks.loc[stock, ["Alpha", "Beta"]] = [alpha, beta]

    plot_alpha_beta(stocks)
