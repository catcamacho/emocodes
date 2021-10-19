# import needed libraries
import pandas as pd
from datetime import datetime
from os.path import abspath, isdir, basename
from os import mkdir
from emocodes.processing.video import get_video_length
import logging as emolog

# TODO: make validator class

class CodeTimeSeries:
    """ This class processes a Datavyu CSV. converting the codes to a time series for bio-behavioral analysis."""
    def __init__(self, interpolate_gaps=True, sampling_rate=5):
        """
        Parameters
        ----------
        interpolate_gaps : bool
            Defaults is 'True'.  To leave gaps blank, set to False.
        sampling_rate : float
            Desired output sampling rate in Hz (samples per second). Default is 5 Hz.
        """

        today = datetime.now()

        self.codes_df = None
        self.labels = None
        self.video_duration = None
        self.sampling_rate = sampling_rate
        self.interpolate_gaps = interpolate_gaps
        self.proc_codes_df = None

        # set up logging
        if not isdir('./logs'):
            mkdir('./logs')
        log_file = './logs/file_{0}.log'.format(today.strftime('%Y%m%d'))
        emolog.basicConfig(filename=log_file, level=emolog.DEBUG)

    def proc_codes_file(self, codes_file, video_file, save_file_name='video_codes_time_series', file_type='csv'):
        self.load_codes_file(codes_file)
        self.find_video_length(video_file)
        self.get_labels()
        self.convert_codes()
        self.save(save_file_name=save_file_name, file_type=file_type)

    def load_codes_file(self, codes_file):
        self.codes_df = pd.read_csv(codes_file, index_col=None)
        emolog.info("loading: " + codes_file)
        return self

    def get_labels(self):
        self.labels = get_code_labels(self.codes_df)
        return self

    def find_video_length(self, video_file):
        self.video_duration = get_video_length(video_file)
        emolog.info("using video: " + video_file)
        return self

    def convert_codes(self):
        self.proc_codes_df = validate_convert_timestamps(self.labels, self.codes_df, self.video_duration,
                                                         self.sampling_rate, self.interpolate_gaps, do_log=True)
        return self

    def save(self, save_file_name='video_codes_time_series', file_type='csv'):
        save_timeseries(self.proc_codes_df, file_type, save_file_name, do_log=True)


class ValidateTimeSeries:
    def __init__(self):
        """

        """

    def check_gaps(self):

        return self

    def check_offsets(self):

        return

    def check_onsets(self):

        return


# extract the unique code names (assumes Datavyu CSV export format)
def get_code_labels(codes_df):
    """
    Pull the unique labels from the Datavyu codes.

    Parameters
    ---------
    codes_df : DataFrame
        The dataframe of codes created by importing the Datavyu codes using pandas.

    Returns
    -------
    labels : list
        Variable names of the codes from Datavyu
    """

    labels = pd.Series(codes_df.columns).str.split(pat='.', expand=True)
    labels = list(labels[0].unique())
    for x in labels:
        if 'Unnamed' in x:
            labels.remove(x)
    return labels
    

def validate_convert_timestamps(labels, codes_df, video_duration, sampling_rate, interpolate_gaps=True, do_log=False):
    """ This function performs two steps:
        1. Checks for human errors in coding such as incorrect end times or gaps in coding.
        2. Convert the timestamps to time series and optionally interpolate across gaps.

        Parameters
        ----------

        labels : list
            a list of strings which are the unique column variable labels in the dataframe.
            The output of the get_code_labels function.

        codes_df : DataFrame
            the dataframe of Datavyu codes

        video_duration : int
            The length of video in milliseconds, the output of get_video_length

        sampling_rate : int
            The sampling rate in Hz that the file should be saved as.

        interpolate_gaps : bool
            Default is set to True.  If you wish for gaps to be preserved as NaNs, set to False.

        do_log : bool
            Default is set to False. If logging is being used, set to True.

        Returns
        -------
        timeseries_df : DataFrame
            The resampled code time series
        """

    # set up dataframe object to store data
    timeseries_df = pd.DataFrame(columns=labels,
                                 index=range(0, video_duration+int(1000/sampling_rate), int(1000/sampling_rate)))
    timeseries_df.index.name = 'onset'

    for label in labels:
        label_df = codes_df[[label+'.onset', label+'.offset', label+'.code01']].dropna(axis=0, how='any')

        # check if offsets precede onsets
        dur = label_df[label+'.offset'] - label_df[label+'.onset']
        for d in dur:
            if d <= 0:
                msg = "code '{0}' has an offset time that is before the corresponding onset time.".format(label)
                if do_log:
                    emolog.error(msg)
                raise ValueError(msg)

        # check that the offset for the last code is not after the end of the episode
        if label_df.loc[label_df.index[-1], label+'.offset'] > video_duration:
            msg = "Warning: The last offset for code '{0}' is after the end of the video. Correcting.".format(label)
            print(msg)
            if do_log:
                emolog.info(msg)
            label_df.loc[label_df.index[-1], label+'.offset'] = video_duration

        # add codes to the timeseries dataframe
        for x in label_df.index:
            onset = int(label_df.loc[x, label+'.onset'])
            offset = int(label_df.loc[x, label+'.offset'])
            timeseries_df.loc[onset:offset, label] = label_df.loc[x, label+'.code01']

        # check for gaps in the codes and interpolate if interp flag is set to True
        timeseries_df[label] = pd.to_numeric(timeseries_df[label], errors='ignore')
        nans = timeseries_df[label].isna()
        missing = sum(nans)*sampling_rate

        if missing > 0:
            if interpolate_gaps is True and (timeseries_df[label].dtype == float or timeseries_df[label].dtype == int):
                msg = "Warning: there are {0}ms of interpolated codes for '{1}'".format(missing, label)
            else:
                msg = "Warning: there are {0}ms of missing codes for '{1}'".format(missing, label)
            if do_log:
                emolog.info(msg)
            print(msg)

    # interpolate gaps in codes if 'interpolate_gaps' is set to True
    if interpolate_gaps is True:
        timeseries_df = timeseries_df.interpolate(method='nearest', axis=0)
        
    return timeseries_df


def save_timeseries(timeseries_df, outfile_type, outfile_name, do_log=False):
    """
    Parameters
    ----------

    timeseries_df : DataFrame
        The resampled time series from validate_convert_timestamps

    outfile_type : str ('csv','excel','tab','space')
        the file type to save the time series as.

    outfile_name : str
        The file prefix for the output file.

    do_log : bool
        Default is set to False. If logging is being used, set to True.

    Returns
    -------

    None

    """

    # get the date and time right now
    today = datetime.now()
    
    if outfile_type == 'csv':
        # save as a csv
        out_file='{0}_{1}.csv'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S'))
        timeseries_df.to_csv(out_file, na_rep='NA')
    elif outfile_type == 'excel':
        # save as an excel file
        out_file='{0}_{1}.xlsx'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S'))
        timeseries_df.to_excel(out_file, na_rep='NA')
    elif outfile_type == 'tab':
        # save as a tab-delimited file
        out_file='{0}_{1}.txt'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S'))
        timeseries_df.to_csv(out_file, sep='\t', na_rep='NA')
    elif outfile_type == 'space':
        # save as a space-delimited file
        out_file='{0}_{1}.txt'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S'))
        timeseries_df.to_csv(out_file, sep='  ', na_rep='NA')
    else:
        msg = 'Warning: data note saved! Please indicate the file format: csv, excel, tab, space'
        if do_log:
            emolog.warning(msg)
        print(msg)
    
    filepath = abspath(basename(out_file))
    msg = 'Code time series saved at {0}'.format(filepath)
    if do_log:
        emolog.info(msg)
    print(msg)
