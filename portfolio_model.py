import pandas as pd


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


def get_final_allocation() -> pd.DataFrame:
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

    return combined_df


if __name__ == '__main__':
    final_allocation = get_final_allocation()
    print(final_allocation)
