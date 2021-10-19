import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
import os
from emocodes.plotting import plot_heatmap, plot_vif, make_num_plot
import markdown
import weasyprint as wp

class SummarizeVideoFeatures:
    def __init__(self):
        """
        This class produces a summary of video features to help users judge the suitability of each feature for regression
        analysis.
        """
        self.features = None
        self.features_file = None
        self.sampling_rate = None
        self.units = None
        self.time_col = None
        self.columns_names = None
        self.convolve_hrf_first = None
        self.vif_scores = None
        self.vif_plot = None
        self.ips_scores = None
        self.ips_plot = None
        self.corr_scores = None
        self.corr_plot = None
        self.power_spectra = None
        self.power_plot = None
        self.feature_plot = None
        self.hrf_feature_plots = None
        self.fig_dir = None

    def compile(self, features, out_dir, convolve_hrf=True, column_names='all', sampling_rate=10, units='s',
                time_col='index'):
        """
        This function runs the methods to create a features report.
        Parameters
        ----------
        features : filepath
            A CSV containing a dataframe object with timeseries data for each feature you want to include in the report
        out_dir : filepath
            The full or relative path to the folder where you want the report saved to.
        convolve_hrf : bool
            Setting to convolve each feature with a double-gamma hemodynamic response function (HRF) before creating the
             report
        column_names : list
            The columns to include in the feature analysis
        sampling_rate : float
            Sampling rate in Hz (samples per second) of the input data
        units : str
            Must be 's', 'ms', 'm', or 'h' indicating seconds, milliseconds, minutes, or hours respectively. The units
            that the time variable (index) is in.
        time_col : str
            The name of the column to use as time if not the index.

        """
        self.features_file = features
        self.time_col = time_col
        if time_col == 'index':
            self.features = pd.read_csv(features, index_col=0)
        else:
            self.features = pd.read_csv(features, index_col=self.time_col)
        self.sampling_rate = sampling_rate
        self.units = units

        self.column_names = column_names
        self.convolve_hrf_first = convolve_hrf
        self.fig_dir = os.path.join(out_dir, 'figs')
        os.makedirs(self.fig_dir, exist_ok=True)

        if convolve_hrf:
            self.hrf_conv_features = hrf_convolve_features(self.features, column_names=self.column_names,
                                                           time_col=self.time_col)
        else:
            self.hrf_conv_features = self.features
        self.compute_plot_corr()
        self.compute_plot_ips()
        self.compute_plot_vif()
        self.compute_plot_power()
        self.plot_features()

        # set filenames for each report output type
        reportmd = os.path.join(out_dir, 'report.md')
        reporthtml = os.path.join(out_dir, 'report.html')
        reportpdf = os.path.join(out_dir, 'report.pdf')

        # write markdown file
        with open(reportmd, 'w') as f:
            f.write('# EmoCodes Analysis Summary Report\n')
            f.write('in_file: {0} \n'.format(self.features_file))
            f.write('---\n')
            f.write('## Features Included in this Analysis\n')
            f.write()
        return self

    def compute_plot_corr(self):
        self.corr_scores = pairwise_corr(self.hrf_conv_features, column_names=self.column_names, nan_policy='omit')
        f = plot_heatmap(self.corr_scores)
        plt.savefig(f, os.join(self.fig_dir, 'corr_plot.svg'))
        self.corr_plot = os.join(self.fig_dir, 'corr_plot.svg')
        return self

    def compute_plot_ips(self):
        ips_df, ips = pairwise_ips(self.hrf_conv_features, column_names=self.column_names)
        self.ips_scores = ips_df
        f = plot_heatmap(ips_df)
        plt.savefig(f, os.join(self.fig_dir, 'mean_ips_plot.svg'))
        self.ips_plot = os.join(self.fig_dir, 'mean_ips_plot.svg')
        return self

    def compute_plot_vif(self):
        self.vif_scores = vif_collinear(self.hrf_conv_features, column_names=self.column_names)
        f = plot_vif(self.vif_scores)
        plt.savefig(f, os.join(self.fig_dir, 'vif_plot.svg'))
        self.vif_plot = os.join(self.fig_dir, 'vif_plot.svg')
        return self

    def compute_plot_power(self):
        self.power_spectra = feature_freq_power(self.hrf_conv_features, time_col=self.time_col, units=self.units,
                                                column_names=self.column_names, sampling_rate=self.sampling_rate)
        f = make_num_plot(self.power_spectra)
        plt.savefig(f, os.join(self.fig_dir, 'power_plot.svg'))
        self.power_plot = os.join(self.fig_dir, 'power_plot.svg')

        return self

    def plot_features(self):
        if self.convolve_hrf_first:
            f = make_num_plot(self.hrf_conv_features)
            plt.savefig(f, os.join(self.fig_dir, 'hrf_features_plot.svg'))
            self.hrf_feature_plots = os.join(self.fig_dir, 'hrf_features_plot.svg')

        f = make_num_plot(self.features)
        plt.savefig(f, os.join(self.fig_dir, 'features_plot.svg'))
        self.feature_plot = os.join(self.fig_dir, 'features_plot.svg')
        return self


def pairwise_ips(features, column_names='all'):
    """
    This function computes the pair-wise instantaneous phase synchrony (IPS) between columns in a dataframe.  It returns
    both the mean IPS in a NxN matrix as well as a numpy array that is size NxNxT containing the pair-wise IPS at each
    timepoint.
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
    mean_ips_df = pd.DataFrame(1, columns=column_names, index=column_names)

    # get unique pairs of column labels
    combs = itertools.combinations(column_names, 2)
    for pair in combs:
        a = pair[0]
        b = pair[1]
        a_idx = column_names.index(a)
        b_idx = column_names.index(b)
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
    Computes the pair-wise Spearman correlation coefficient for a set of features.
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


def vif_collinear(features, column_names='all'):
    """
    Wraps the pliers variance inflation factor command. Computes the variance inflation factor for the specified
    columns in a set of features.
    Parameters
    ----------
    features: DataFrame
        DataFrame with signals to be analyzed.
    column_names: list
        List of columns to compare pairwise in the ratings DataFrame. Default is 'all'.

    Returns
    -------
    vif_scores : Series
        Pandas Series obect containing the VIF scores for each column in column_names.
    """
    from pliers.diagnostics import variance_inflation_factors

    if column_names != 'all':
        try:
            features = features[column_names]
        except Exception:
            raise ValueError("column names not found in features dataframe.")

    vif_scores = variance_inflation_factors(features)

    return vif_scores


def feature_freq_power(features, time_col='index', units='s', column_names='all', sampling_rate=10):
    """

    Parameters
    ----------
    features: DataFrame
        A Pandas dataframe with the feature signals to convolve.
    time_col: str
        Name of the column containing time information.  Default is to use the DataFrame index.
    units: str ['H', 'M', 's', 'ms']
        units that the time variable is is (if not a datetime index). Default is 's'.
    column_names: list
        List of columns to conduct spectrum analysis on.  Default is to use all the columns.
    sampling_rate: int or float
        input sampling rating in Hz.

    Returns
    -------
    power: DataFrame
        A DataFrame with the power spectrums for each variable in columns_names (index is frequency up to nyquist.)

    """

    if units != 's':
        if time_col == 'index':
            features['time_orig_index'] = features.index
            features.index = range(0, len(features))
            features.index.name = None
            time_col = 'time_orig_index'
        if units == 'm' or units == 'minutes':
            features[time_col] = features[time_col] * 60
        if units == 'm' or units == 'hours':
            features[time_col] = features[time_col] * 3600
        if units == 'ms' or units == 'milliseconds':
            features[time_col] = features[time_col] / 1000

    if column_names != 'all':
        try:
            features = features[[time_col] + column_names]
        except Exception:
            raise ValueError("column names not found in features dataframe.")

    mm = MinMaxScaler((0, 1))
    features[column_names] = mm.fit_transform(features[column_names].to_numpy())

    power = pd.DataFrame(columns=column_names)
    for a in column_names:
        fourier_transform = np.fft.rfft(features[a])
        abs_fourier_transform = np.abs(fourier_transform)
        power[a] = np.square(abs_fourier_transform)

    power.index = np.linspace(0, sampling_rate / 2, len(power))
    power.index.name = 'Frequency'

    return power


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
    else:
        time = features[time_col]

    convolved_features = pd.DataFrame(columns=[time_col] + column_names)
    hrf_sig = hrf(time)
    for a in column_names:
        convolved_features[a] = np.convolve(features[a], hrf_sig)[:len(time)]

    return convolved_features
