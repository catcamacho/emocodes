import pandas as pd
import numpy as np
import itertools

# TODO make feature analysis classes


def pairwise_ips(features, column_names='all'):
    """

    Parameters
    ----------
    features: DataFrame
        The dataframe with signals to be analyzed.
    column_names: list
        List of columns to compare pairwise in the ratings DataFrame. Default is 'all'.

    Returns
    -------
    mean_ips_df: DataFrame
        NxN DataFrame with pairwise feature mean phase synchrony
    ips_series: numpy array
        NxNxT (feature x feature x time) array with the instantaneous phase synchrony at each timepoint, pairwise

    """

    from scipy.signal import hilbert

    if column_names == 'all':
        column_names = features.columns

    ips_series = np.ones((len(column_names), len(column_names), len(features)))
    mean_ips_df = pd.DataFrame(columns=column_names, index=column_names)

    # get unique pairs of column labels
    combs = itertools.combinations(column_names, 2)
    for pair in combs:
        a = pair[0]
        b = pair[1]
        a_idx = features.get_loc(a)
        b_idx = features.get_loc(b)
        aphase = np.angle(hilbert(features[a]), deg=False)
        bphase = np.angle(hilbert(features[b]), deg=False)
        phase_synchrony = 1 - np.sin(np.abs(aphase - bphase) / 2)
        ips_series[a_idx, b_idx, :] = phase_synchrony
        mean_ips_df.loc[a, b] = np.mean(phase_synchrony)
        ips_series[b_idx, a_idx, :] = phase_synchrony
        mean_ips_df.loc[b, a] = np.mean(phase_synchrony)
    return mean_ips_df, ips_series


def pairwise_corr(features, column_names='all', nan_policy='omit'):
    """

    Parameters
    ----------
    features: DataFrame
        DataFrame with signals to be analyzed.
    column_names: list
        List of columns to compare pairwise in the ratings DataFrame. Default is 'all'.
    nan_policy: str
        policy for dealing with NaNs that is passed to scipy.spearmanr.  Default is "omit".

    Returns
    -------
    corr_mat_df: DataFrame
        Pairwise Spearman correlations organized into a Pandas DataFrame.
    """
    from scipy.stats import spearmanr
    if column_names == 'all':
        column_names = features.columns

    corr_mat_df = pd.DataFrame(1, columns=column_names, index=column_names)
    # get unique pairs of column labels
    combs = itertools.combinations(column_names, 2)
    for pair in combs:
        a = pair[0]
        b = pair[1]
        r, p = spearmanr(features[a], features[b], nan_policy=nan_policy)
        corr_mat_df.loc[a, b] = r
        corr_mat_df.loc[b, a] = r

    return corr_mat_df


def vif_collineary(features, column_names='all'):
    """

    Parameters
    ----------
    features: DataFrame
        DataFrame with signals to be analyzed.
    column_names: list
        List of columns to compare pairwise in the ratings DataFrame. Default is 'all'.

    Returns
    -------

    """
    from pliers.diagnostics import variance_inflation_factors

    if column_names == 'all':
        column_names = features.columns
    else:
        try:
            features = features[column_names]
        except:
            raise("column names not found in features dataframe.")





    return vif_scores, vif_plot


def feature_freq_power(features, column_names='all'):
    return power, power_plot


def hrf(time, time_to_peak=6, undershoot_dur=12):
    """

    Parameters
    ----------
    time: numpy array
        a 1D numpy array that makes up the x-axis (time) of our HRF in seconds
    time_to_peak: int
        Time to HRF peak in seconds. Default is 6 seconds.
    undershoot_durL int
        Duration of the post-peak undershoot.  Default is 12 seconds.

    Returns
    -------
    hrf_timeseries: numpy array
        The y-values for the HRF at each time point
    """

    from scipy.stats import gamma

    peak = gamma.pdf(time, time_to_peak)
    undershoot = gamma.pdf(time, undershoot_dur)
    hrf_timeseries = peak - 0.35 * undershoot
    return hrf_timeseries


def hrf_convolve_features(features, column_names='all', time_col='index'):
    """

    Parameters
    ----------
    features: DataFrame
        A Pandas dataframe with the feature signals to convolve.
    column_names: list
        List of columns names to use.  Default is "all"
    time_col: str
        The name of the time column to use if not the index. Must be in seconds. Default is "index".

    Returns
    -------
    convolved_features: DataFrame
        The HRF-convolved feature timeseries
    """
    if column_names == 'all':
        column_names = features.columns

    if time_col == 'index':
        time = features.index.to_numpy()

    convolved_features = pd.DataFrame(columns=column_names, index=time)
    hrf_sig = hrf(time)
    for a in column_names:
        convolved_features[a] = np.convolve(features[a], hrf_sig)[:len(time)]

    return convolved_features



