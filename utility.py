import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import dataframe_image as dfi

def calculate_momentum_score(z):
    if z > 0:
        return 1 + z
    elif z < 0:
        return 1 / (1 - z)
    else:
        return 1


def get_first_dezil(df, column):
    first_dezil = df[column].quantile(0.9)
    first_dezil_df = df[df[column] >= first_dezil]
    return first_dezil_df


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


def style_universe_with_bars(df: pd.DataFrame, name: str) -> str:
    df = df[['name', 'gics_sector_name', 'momentum_val', 'momentum_score']].copy()
    df['momentum_val'] = pd.to_numeric(df['momentum_val'], errors='coerce') * 100
    df = df.sort_values(by='momentum_score', ascending=False)

    momentum_val_max_abs_value = max(abs(df['momentum_val'].min()), abs(df['momentum_val'].max()))
    momentum_score_max_abs_value = max(abs(df['momentum_score'].min()), abs(df['momentum_score'].max()))

    cm = LinearSegmentedColormap.from_list("custom_red_green", ["red", "white", "green"], N=len(df))

    styled = (df.style
        .bar(subset='momentum_val', cmap=cm, align=0, vmax=momentum_val_max_abs_value, vmin=-momentum_val_max_abs_value)
        .bar(subset='momentum_score', cmap=cm, align=0, vmax=momentum_score_max_abs_value, vmin=-momentum_score_max_abs_value)
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
        'momentum_val': "{:.2f}%",
        'momentum_score': "{:.2f}"
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

