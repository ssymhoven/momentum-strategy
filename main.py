from utility import get_universe_data, plot_histogram, get_first_dezil, style_universe_with_bars

spx = get_universe_data(universe="S&P 500")
sxxp = get_universe_data(universe="STOXX Europe 600")
dax = get_universe_data(universe="DAX Index")

if __name__ == '__main__':
    plot_histogram(spx, 'S&P 500')
    plot_histogram(sxxp, 'STOXX Europe 600')
    plot_histogram(dax, 'DAX')

    first_dezil_spx_df = get_first_dezil(spx, 'momentum_score')
    style_universe_with_bars(first_dezil_spx_df, "S&P 500")

    first_dezil_sxxp_df = get_first_dezil(sxxp, 'momentum_score')
    style_universe_with_bars(first_dezil_sxxp_df, "STOXX Europe 600")

