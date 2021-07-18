import pandas as pd
import numpy as np
import itertools


# TODO make feature analysis classes


def pairwise_ips(ratings, column_names='all'):
    """

    Parameters
    ----------
    ratings: DataFrame
        The dataframe with signals to be analyzed.
    column_names: str or list
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
        column_names = ratings.columns

    ips_series = np.ones((len(column_names), len(column_names), len(ratings)))
    mean_ips_df = pd.DataFrame(columns=column_names, index=column_names)

    # get unique pairs of column labels
    combs = itertools.combinations(column_names, 2)
    for pair in combs:
        a = pair[0]
        b = pair[1]
        a_idx = ratings.get_loc(a)
        b_idx = ratings.get_loc(b)
        aphase = np.angle(hilbert(ratings[a]), deg=False)
        bphase = np.angle(hilbert(ratings[b]), deg=False)
        phase_synchrony = 1 - np.sin(np.abs(aphase - bphase) / 2)
        ips_series[a_idx, b_idx, :] = phase_synchrony
        mean_ips_df.loc[a, b] = np.mean(phase_synchrony)
        ips_series[b_idx, a_idx, :] = phase_synchrony
        mean_ips_df.loc[b, a] = np.mean(phase_synchrony)
    return mean_ips_df, ips_series


def pairwise_corr(ratings, column_names='all', nan_policy='omit'):
    """

    Parameters
    ----------
    ratings: DataFrame
        DataFrame with signals to be analyzed.
    column_names: str or list
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
        column_names = ratings.columns

    corr_mat_df = pd.DataFrame(1, columns=column_names, index=column_names)
    # get unique pairs of column labels
    combs = itertools.combinations(column_names, 2)
    for pair in combs:
        a = pair[0]
        b = pair[1]
        r, p = spearmanr(ratings[a], ratings[b], nan_policy=nan_policy)
        corr_mat_df.loc[a, b] = r
        corr_mat_df.loc[b, a] = r

    return corr_mat_df
