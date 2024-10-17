import numpy as np
import pandas as pd
import dataframe_image as dfi
from matplotlib.colors import LinearSegmentedColormap


def calculate_momentum_score(z: float) -> float:
    """
    Calculate the momentum score based on the z-score.

    Args:
        z (float): The z-score of the risk-adjusted momentum value.

    Returns:
        float: The momentum score.
    """
    if z > 0:
        return 1 + z
    elif z < 0:
        return 1 / (1 - z)
    else:
        return 1


def calculate_allocation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the allocation based on momentum values.

    Args:
        df (pd.DataFrame): DataFrame containing the necessary columns.

    Returns:
        pd.DataFrame: DataFrame with calculated momentum values and weights.
    """
    df['momentum_val'] = (df['#1D'] / df['#12M']) - 1
    df['risk_adjusted_momentum_val'] = df['momentum_val'] / df['#std12M']

    mean_val = df['risk_adjusted_momentum_val'].mean()
    std_val = df['risk_adjusted_momentum_val'].std()
    df['z_score'] = (df['risk_adjusted_momentum_val'] - mean_val) / std_val

    df['momentum_score'] = df['z_score'].apply(calculate_momentum_score)

    total_momentum_score = df['momentum_score'].sum()
    df['weight'] = (df['momentum_score'] / total_momentum_score) * 100
    return df


def get_universe_data(universe: str) -> pd.DataFrame:
    df = pd.read_excel('universe.xlsx', sheet_name=universe, header=0, index_col=0)
    df['momentum_val'] = df.apply(
        lambda row: (row['#1D'] / row['#12M']) - 1 if pd.notna(row['#12M']) else
        ((row['#1D'] / row['#9M']) - 1 if pd.notna(row['#9M']) else pd.NA), axis=1
    )

    df['risk_adjusted_momentum_val'] = df.apply(
        lambda row: row['momentum_val'] / row['#std12M'] if pd.notna(row['#12M']) else
        (row['momentum_val'] / row['#std9M'] if pd.notna(row['#9M']) else pd.NA), axis=1
    )

    df.dropna(subset=['momentum_val'], inplace=True)

    mean_val = df['risk_adjusted_momentum_val'].mean()
    std_val = df['risk_adjusted_momentum_val'].std()

    df['z_score'] = df['risk_adjusted_momentum_val'].apply(
        lambda x: (x - mean_val) / std_val
    )
    df['momentum_score'] = df['z_score'].apply(calculate_momentum_score)

    return df


def calculate_weighted_metrics(group: pd.DataFrame) -> pd.Series:
    """
    Calculate weighted metrics for a group based on market capitalization.

    Args:
        group (pd.DataFrame): Grouped DataFrame.

    Returns:
        pd.Series: Series with weighted metrics.
    """
    weight = group['cur_mkt_cap().value'] / group['cur_mkt_cap().value'].sum()
    weighted_1D = (group['#1D'] * weight).sum()
    weighted_12M = (group['#12M'] * weight).sum()
    weighted_std12M = (group['#std12M'] * weight).sum()
    return pd.Series({
        '#1D': weighted_1D,
        '#12M': weighted_12M,
        '#std12M': weighted_std12M,
    })


def get_us_sector_allocation() -> pd.DataFrame:
    """
    Get the US sector allocation based on momentum values.

    Returns:
        pd.DataFrame: DataFrame with US sector allocations.
    """
    df = pd.read_excel('universe.xlsx', sheet_name="US Sector", header=0, index_col=0)
    df = calculate_allocation(df)
    return df


def get_eu_sector_allocation() -> pd.DataFrame:
    """
    Get the EU sector allocation based on weighted momentum values.

    Returns:
        pd.DataFrame: DataFrame with EU sector allocations.
    """
    df = pd.read_excel('universe.xlsx', sheet_name="EU Sector", header=0)
    df_weighted = df.groupby('GICS').apply(calculate_weighted_metrics)
    df_weighted = calculate_allocation(df_weighted)
    return df_weighted


def get_final_sector_allocation() -> pd.DataFrame:
    """
    Combine US and EU sector allocations into a final allocation of 60% EU and 40% US.

    Returns:
        pd.DataFrame: DataFrame with US allocation, EU allocation, and combined allocation.
    """
    us_allocation = get_us_sector_allocation()
    eu_allocation = get_eu_sector_allocation()

    us_weights = us_allocation[['weight']].rename(columns={'weight': 'S&P 500'})
    eu_weights = eu_allocation[['weight']].rename(columns={'weight': 'Stoxx Europe 600'})

    combined_df = pd.merge(eu_weights, us_weights, left_index=True, right_index=True, how='outer')

    combined_df.fillna(0, inplace=True)

    combined_df['60/40 Portfolio'] = 0.6 * combined_df['Stoxx Europe 600'] + 0.4 * combined_df['S&P 500']

    plot_allocation(combined_df)

    return combined_df


def plot_allocation(df: pd.DataFrame):
    max_val = df.abs().max().max()

    cm = LinearSegmentedColormap.from_list("custom_red_green", ["red", "white", "green"], N=len(df))

    styled = (df.style.bar(subset='Stoxx Europe 600', cmap=cm, align=0, vmin=-max_val, vmax=max_val)
    .bar(subset='S&P 500', cmap=cm, align=0, vmin=-max_val, vmax=max_val)
    .bar(subset='60/40 Portfolio', cmap=cm, align=0, vmin=-max_val, vmax=max_val)
    .set_table_styles([
        {'selector': 'th.col0',
         'props': [('border-left', '1px solid black')]},
        {'selector': 'td.col0',
         'props': [('border-left', '1px solid black')]},
        {
            'selector': 'th.index_name',
            'props': [('min-width', '150px'), ('white-space', 'nowrap')]
        }
    ]).format({
        'Stoxx Europe 600': "{:.2f}%",
        'S&P 500': "{:.2f}%",
        '60/40 Portfolio': "{:.2f}%"
    }))

    output_path = f'images/sector_allocation.png'
    dfi.export(styled, output_path, table_conversion="selenium")


if __name__ == '__main__':
    fsector_allocation = get_final_sector_allocation()

    us_universe = get_universe_data(universe="US Underlying")
    eu_universe = get_universe_data(universe="EU Underlying")

    us_universe.to_excel("us_universe.xlsx")
    eu_universe.to_excel("eu_universe.xlsx")