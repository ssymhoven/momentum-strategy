import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import dataframe_image as dfi


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


def calculate_momentum_score(z):
    if z > 0:
        return 1 + z
    elif z < 0:
        return 1 / (1 - z)
    else:
        return 1


def style_universe_with_bars(df: pd.DataFrame, name: str) -> str:
    df = df[['name', 'gics_sector_name', '#mkt_cap', 'momentum_val', 'momentum_score', 'weight']].copy()

    df = df.rename(columns={
        'name': 'Name',
        'gics_sector_name': 'Sector',
        '#mkt_cap': 'Market Cap (Mio.)',
        'momentum_val': 'Momentum Value',
        'momentum_score': 'Momentum Score',
        'weight': 'Weight'
    })

    df['Momentum Value'] = pd.to_numeric(df['Momentum Value'], errors='coerce') * 100
    df['Market Cap (Mio.)'] = df['Market Cap (Mio.)'] / 1_000_000

    df = df.sort_values(by='Momentum Score', ascending=False)

    momentum_val_max_abs_value = max(abs(df['Momentum Value'].min()), abs(df['Momentum Value'].max()))
    momentum_score_max_abs_value = max(abs(df['Momentum Score'].min()), abs(df['Momentum Score'].max()))

    cm = LinearSegmentedColormap.from_list("custom_red_green", ["red", "white", "green"], N=len(df))

    styled = (df.style
        .bar(subset='Momentum Value', cmap=cm, align=0, vmin=-momentum_val_max_abs_value, vmax=momentum_val_max_abs_value)
        .bar(subset='Momentum Score', cmap=cm, align=0, vmin=-momentum_score_max_abs_value, vmax=momentum_score_max_abs_value)
        .set_table_styles([
        {'selector': 'th.col0',
         'props': [('border-left', '1px solid black')]},
        {'selector': 'td.col0',
         'props': [('border-left', '1px solid black')]},
        {
            'selector': 'th.index_name',
            'props': [('min-width', '150px'), ('white-space', 'nowrap')]
        },
        {
            'selector': 'td.col0',
            'props': [('min-width', '200px'), ('white-space', 'nowrap')]
        },
        {
            'selector': 'td.col1',
            'props': [('min-width', '150px'), ('white-space', 'nowrap')]
        }
    ])
    .format({
        'Momentum Value': "{:.2f}%",
        'Momentum Score': "{:.2f}",
        'Market Cap (Mio.)': "{:,.2f} $" if 'S&P' in name else "{:,.2f} €",
        'Weight': "{:.2f}%",
    }))

    output_path = f'images/dezil_momentum_score_{name}.png'
    dfi.export(styled, output_path, table_conversion="selenium")
    return output_path


def plot_histogram(df, universe_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['momentum_score'], kde=True, bins=30, color='#7EC0C6')
    plt.title(f'Momentum Score Distribution - {universe_name}')
    plt.xlabel('Momentum Score')
    plt.ylabel('Frequency')
    plt.savefig(f'images/boxplot_momentum_score_{universe_name}.png')

