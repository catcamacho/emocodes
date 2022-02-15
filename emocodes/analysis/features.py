import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import os
from emocodes.plotting import plot_heatmap, plot_vif
import markdown
import weasyprint as wp

sns.set(context='paper', style='white')

class SummarizeVideoFeatures:
    """
    This class produces a summary report of video features to help users judge the suitability of each feature for
    regression analysis. After running the class, a PDF, markdown, and HTML version of the report are saved in the
    output folder along with a folder of figures.

        >>> import emocodes as ec
        >>> codes = 'video_features.csv' # DataFrame saved as CSV with feature timeseries
        >>> output = './report' # directory to save the report in
        >>> report = ec.SummarizeVideoFeatures()
        >>> report.compile(codes, output)

    """
    def __init__(self):
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
        features: filepath
            A CSV containing a dataframe object with timeseries data for each feature you want to include in the report
        out_dir: filepath
            The full or relative path to the folder where you want the report saved to.
        convolve_hrf: bool
            Setting to convolve each feature with a double-gamma hemodynamic response function (HRF) before reporting
        column_names: list
            The columns to include in the feature analysis
        sampling_rate: float
            Sampling rate in Hz (samples per second) of the input data
        units: str
            Must be 's', 'ms', 'm', or 'h' indicating seconds, milliseconds, minutes, or hours respectively. The units
            that the time variable (index) is in.
        time_col: str
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
                                                           time_col=self.time_col, units=self.units)
        else:
            self.hrf_conv_features = self.features
        self.compute_plot_corr()
        self.compute_plot_ips()
        self.compute_plot_vif()
        self.plot_features()

        # set filenames for each report output type
        reportmd = os.path.join(out_dir, 'report.md')
        reporthtml = os.path.join(out_dir, 'report.html')
        reportpdf = os.path.join(out_dir, 'report.pdf')

        # write markdown file
        with open(reportmd, 'w') as f:
            f.write('# EmoCodes Analysis Summary Report\n\n')
            f.write('**in_file:** {0} \n\n'.format(self.features_file))
            f.write('| Feature | Non-Zero | Min Value | Max Value |\n')
            f.write('| :------ | :------: | :-------: | :-------: |\n')
            for c in self.features.columns:
                nonzero = (sum(self.features[c] != 0)/len(self.features))*100
                f.write('| {0} | {1}% | {2} | {3} |\n'.format(c, round(nonzero, 2), round(self.features[c].min(), 1),
                                                             round(self.features[c].max(), 1)))
            f.write('\n')
            f.write('******\n\n')
            if self.convolve_hrf_first:
                f.write('## Features Included in this Analysis\n\n')
                f.write('### Original Features\n\n')
                f.write('![feature plots]({0})\n\n'.format(self.feature_plot))
                f.write('### After HRF convolution\n\n')
                f.write('![hrf-convolved feature plots]({0})\n\n'.format(self.hrf_feature_plots))
            else:
                f.write('## Features Included in this Analysis\n\n')
                f.write('![feature plots]({0})\n\n'.format(self.feature_plot))
            f.write('******\n\n')
            f.write('## Spearman Correlations\n\n')
            f.write('![correlation plots]({0})\n\n'.format(self.corr_plot))
            f.write('******\n')
            f.write('## Mean Instantaneous Phase Synchrony\n\n')
            f.write('![mean IPS plots]({0})\n\n'.format(self.ips_plot))
            f.write('******\n')
            f.write('## Variance Inflation Factors\n\n')
            f.write('![VIF plots]({0})\n'.format(self.vif_plot))
        f.close()
        # convert the markdown to HTML
        with open(reportmd, 'r') as f:
            text = f.read()
            html = markdown.markdown(text, extensions=['tables'])
        with open(reporthtml, 'w') as f:
            f.write(html)
        # convert the HTML to PDF
        wp.HTML(reporthtml).write_pdf(reportpdf)
        return self

    def compute_plot_corr(self):
        self.corr_scores = pairwise_corr(self.hrf_conv_features, column_names=self.column_names)
        plot_heatmap(self.corr_scores)
        plt.title('Pair-wise Spearman Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'corr_plot.svg'))
        self.corr_plot = os.path.join(self.fig_dir, 'corr_plot.svg')
        plt.close()
        return self

    def compute_plot_ips(self):
        ips_df, ips = pairwise_ips(self.hrf_conv_features, column_names=self.column_names)
        self.ips_scores = ips_df
        plot_heatmap(ips_df)
        plt.title('Pair-wise Mean IPS')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'mean_ips_plot.svg'))
        self.ips_plot = os.path.join(self.fig_dir, 'mean_ips_plot.svg')
        plt.close()
        return self

    def compute_plot_vif(self):
        self.vif_scores = vif_collinear(self.hrf_conv_features, column_names=self.column_names)
        plot_vif(self.vif_scores)
        plt.title('VIF Scores')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'vif_plot.svg'))
        self.vif_plot = os.path.join(self.fig_dir, 'vif_plot.svg')
        plt.close()
        return self

    def plot_features(self):
        if self.convolve_hrf_first:
            plt.figure(figsize=(7, 1.5 * len(self.column_names)))
            self.hrf_conv_features.plot(kind='line', subplots=True, xlim=(0, self.hrf_conv_features.index[-1]))
            plt.xlabel('Time')
            plt.suptitle('HRF-Convolved Features')
            plt.ticklabel_format(style='plain')
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir, 'hrf_features_plot.svg'))
            self.hrf_feature_plots = os.path.join(self.fig_dir, 'hrf_features_plot.svg')
            plt.close()

        plt.figure(figsize=(7, 1.5 * len(self.column_names)))
        self.features.plot(kind='area', subplots=True, xlim=(0, self.features.index[-1]))
        plt.ticklabel_format(style='plain')
        plt.xlabel('Time')
        plt.suptitle('Original Feature Values')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'features_plot.svg'))
        self.feature_plot = os.path.join(self.fig_dir, 'features_plot.svg')
        plt.close()
        return self


def pairwise_ips(features, column_names='all'):
    """
    This function computes the pair-wise instantaneous phase synchrony (IPS) between columns in a dataframe.  It returns
    both the mean IPS in a NxN matrix as well as a numpy array that is size NxNxT containing the pair-wise IPS at each
    time point.

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
        column_names = features.columns.to_list()

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


def pairwise_corr(features, column_names='all'):
    """
    Computes the pair-wise Spearman correlation coefficient for a set of features.

    Parameters
    ----------
    features: DataFrame
        DataFrame with signals to be analyzed.
    column_names: list
        List of columns to compare pairwise in the ratings DataFrame. Default is 'all'.

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
        r, p = spearmanr(features[a], features[b], nan_policy='omit')
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
    vif_scores: Series
        Pandas Series object containing the VIF scores for each column in column_names.
    """
    from pliers.diagnostics import variance_inflation_factors

    if column_names != 'all':
        try:
            features = features[column_names]
        except Exception:
            raise ValueError("column names not found in features dataframe.")

    vif_scores = variance_inflation_factors(features)

    return vif_scores


def hrf(time, time_to_peak=5, undershoot_dur=12):
    """
    This function creates a hemodynamic response function timeseries.

    Parameters
    ----------
    time: numpy array
        a 1D numpy array that makes up the x-axis (time) of our HRF in seconds
    time_to_peak: int
        Time to HRF peak in seconds. Default is 5 seconds.
    undershoot_dur: int
        Duration of the post-peak undershoot. Default is 12 seconds.

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


def hrf_convolve_features(features, column_names='all', time_col='index', units='s', time_to_peak=5, undershoot_dur=12):
    """
    This function convolves a hemodynamic response function with each column in a timeseries dataframe.

    Parameters
    ----------
    features: DataFrame
        A Pandas dataframe with the feature signals to convolve.
    column_names: list
        List of columns names to use.  Default is "all"
    time_col: str
        The name of the time column to use if not the index. Default is "index".
    units: str
        Must be 'ms','s','m', or 'h' to denote milliseconds, seconds, minutes, or hours respectively.
    time_to_peak: int
        Time to peak for HRF model. Default is 5 seconds.
    undershoot_dur: int
        Undershoot duration for HRF model. Default is 12 seconds.

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
        features.index = time

    if units == 'm' or units == 'minutes':
        features.index = features.index * 60
        time = features.index.to_numpy()
    if units == 'h' or units == 'hours':
        features.index = features.index * 3600
        time = features.index.to_numpy()
    if units == 'ms' or units == 'milliseconds':
        features.index = features.index / 1000
        time = features.index.to_numpy()

    convolved_features = pd.DataFrame(index=time)
    hrf_sig = hrf(time, time_to_peak=time_to_peak, undershoot_dur=undershoot_dur)
    for a in column_names:
        convolved_features[a] = np.convolve(features[a], hrf_sig)[:len(time)]

    return convolved_features
