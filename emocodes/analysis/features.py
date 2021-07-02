import pandas as pd
import numpy as np

# TODO make feature analysis classes
# TODO make feature analysis functions



def pairwise_ips(ratings):
    """

    Parameters
    ----------
    ratings: DataFrame


    Returns
    -------
    mean_ips: DataFrame
        NxN DataFrame with pairwise feature mean phase synchrony
    ips_series: numpy array
        NxNxT (feature x feature x time) array with the instantaneous phase synchrony at each timepoint pairwise

    """
    return(mean_ips, ips_series)


