# import needed libraries
import pandas as pd
from datetime import datetime
from os.path import abspath, basename
import numpy as np
from os import makedirs
from emocodes.processing.video import get_video_length
import logging as emolog
import markdown
import weasyprint as wp


class CodeTimeSeries:
    """
    This class processes a Datavyu CSV. converting the codes to a time series for bio-behavioral analysis. In the
    EmoCodes system, this class should be run after the codes are validated and any reported errors are corrected.

    Example: Use CodeTimeSeries to convert "datavyu_export.csv" (codes of "myvideo.mp4" completed in Datavyu) to a
    timeseries file with a sampling rate of 1.2 Hz. This will save a file called "video_codes_time_series.csv" (the
    default saved file name).
        >>> import emocodes as ec
        >>> datavyu_file = 'datavyu_export.csv'
        >>> video_file = 'myvideo.mp4'
        >>> ec.CodeTimeSeries(sampling_rate=1.2).proc_codes_file(datavyu_file, video_file)

    Parameters
        ----------
        interpolate_gaps: bool
            Defaults is 'True'. To leave gaps blank (NaNs), set to False.
        sampling_rate: float
            Default is 5 Hz. Desired output sampling rate in Hz (samples per second).
        logging_dir: str
            A filepath to a folder to save the processing logs to. Default is to create a new folder within the current
            directory named "logs" to print logs to.
    """
    def __init__(self, interpolate_gaps=True, sampling_rate=5, logging_dir='./logs'):
        today = datetime.now()

        self.codes_df = None
        self.labels = None
        self.video_duration = None
        self.sampling_rate = sampling_rate
        self.interpolate_gaps = interpolate_gaps
        self.proc_codes_df = None

        # set up logging
        makedirs(logging_dir, exist_ok=True)
        log_file = '{0}/logfile_{1}.log'.format(logging_dir, today.strftime('%Y%m%d'))
        emolog.basicConfig(filename=log_file, level=emolog.DEBUG)

    def proc_codes_file(self, codes_file, video_file, save_file_name='video_codes_time_series', file_type='csv'):
        """
        This method fully processes a Datavyu CSV-exported file to a time series output for further analysis.

        Parameters
        ----------
        codes_file: str
            File path to the Datavyu outputs to convert.
        video_file: str
            Filepath to the video MP4 file that was coded in Datavyu
        save_file_name: str
            Default is "video_code_time_series". The file path and name to save the outputs as.
        file_type: str ('csv','excel','tab','space')
            Default is "csv". The type of file to save the data as.

        """
        if not isinstance(codes_file, pd.DataFrame):
            self.codes_df = pd.read_csv(codes_file, index_col=None)
            emolog.info("loading: " + codes_file)
        self.find_video_length(video_file)
        self.get_labels()
        self.convert_codes()
        self.save(save_file_name=save_file_name, file_type=file_type)

    def get_labels(self):
        """
        Method to make a list of the code labels in a Datavyu CSV-exported file.
        """
        self.labels = get_code_labels(self.codes_df)
        return self

    def find_video_length(self, video_file):
        """
        Method to extract the length of a video MP4 file in milliseconds.

        Parameters
        ----------
        video_file; str
            File path to a video MP4 file

        """
        self.video_duration = get_video_length(video_file)
        emolog.info("using video: " + video_file)
        return self

    def convert_codes(self):
        """
        This method converts a Datavyu style output to a long-form, timeseries style output which can be used for
        further analysis.

        """
        self.proc_codes_df = convert_timestamps(self.labels, self.codes_df, self.video_duration,
                                                self.sampling_rate, self.interpolate_gaps, do_log=True)
        return self

    def save(self, save_file_name='video_codes_time_series', file_type='csv'):
        """
        This method saves the processed codes as a CSV for further analysis.

        Parameters
        ----------
        save_file_name: str
            Default is "video_code_time_series". The file path + name to save the processed codes as.
        file_type: str ('csv','excel','tab','space')
            Default is "csv". The type of file to save the data as.

        """
        save_timeseries(self.proc_codes_df, file_type, save_file_name, do_log=True)


class ValidateTimeSeries:
    """
    This class takes a Datavyu-exported CSV and produces a report of the following common problems:

    - Missing values
    - offsets before onsets
    - offsets of zero
    - not starting with the video
    - not ending with the video file
    - segment durations of zero

    The report also gives the following descriptive information:

    - list of unique values per code
    - number of segments per code
    - list of code segments (cells) with problematic data (offsets, onsets, or values)

    This report can then be used to go back and clean the coding data in Datavyu before further processing.

    Example:
        >>> import emocodes as ec
        >>> codes_file = 'datavyu_export_codes.csv'
        >>> video_file = 'myvideo.mp4'
        >>> ec.ValidateTimeSeries().run(codes_file, video_file)

    """
    def __init__(self):
        self.in_file = None
        self.codes = None
        self.labels = None
        self.video_duration = None
        self.report_name = None
        self.time_report = None
        self.val_report = None

    def run(self, file_name, video_file, report_filename=None):
        today = datetime.now()
        self.in_file = file_name
        self.codes = pd.read_csv(self.in_file, index_col=None)
        self.labels = get_code_labels(self.codes)
        self.video_duration = get_video_length(video_file)
        if report_filename:
            self.report_name = report_filename
        else:
            self.report_name = file_name[:-4] + '_report_{0}'.format(today.strftime('%Y%m%d'))
        self.check_timestamps()
        self.check_values()
        report = self.val_report.merge(self.time_report, left_index=True, right_index=True)
        report.to_csv(self.report_name + '.csv')

        # set filenames for each report output type
        reportmd = self.report_name + '.md'
        reporthtml = self.report_name + '.html'
        reportpdf = self.report_name + '.pdf'

        # write markdown file
        with open(reportmd, 'w') as f:
            f.write('# EmoCodes Code Validation Report\n\n')
            f.write('**Datavyu file:** {0} \n\n'.format(self.in_file))
            f.write('**video file:** {0} \n\n'.format(video_file))
            f.write('**Full Report Table**: {0}\n\n'.format(self.report_name + '.csv'))
            f.write('**Code labels found**: ' + ', '.join(self.labels) + '\n\n')
            f.write('### Timestamps Brief Report \n\n')
            f.write('Please note that the cell numbers are zero-indexed, meaning the count starts at 0, not 1.\n\n')
            f.write('| Label | Cells with Bad Onsets | Cells with Bad Offsets | Cells with Bad Durations |\n')
            f.write('| :---- | :-------------------: | :--------------------: | :----------------------: |\n')
            for c in self.labels:
                f.write('| {0} | {1} | {2} | {3} |\n'.format(c, self.time_report.loc[c, 'segs_bad_onsets'],
                                                             self.time_report.loc[c, 'segs_bad_offsets'],
                                                             self.time_report.loc[c, 'segs_bad_duration']))
            f.write('\n')
            f.write('******\n\n')
            f.write('### Values Brief Report \n\n')
            f.write('Please note that the cell numbers are zero-indexed, meaning the count starts at 0, not 1.\n\n')
            f.write('| Label | Unique Values | # Empty Cells | List Empty Cells |\n')
            f.write('| :---- | :-----------: | :-----------: | :--------------: |\n')
            for c in self.labels:
                f.write('| {0} | {1} | {2} | {3} |\n'.format(c, self.val_report.loc[c, 'unique_values'],
                                                             self.val_report.loc[c, 'num_blank_cells'],
                                                             self.val_report.loc[c, 'list_blank_cells']))
            f.write('\n')
            f.write('******\n\n')

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

    def check_timestamps(self):
        self.time_report = timestamps_report(self.codes, self.video_duration, self.labels)
        return self

    def check_values(self):
        self.val_report = values_report(self.codes, self.labels)
        return self


def timestamps_report(codes_df, video_length, labels):
    """
    This function takes a dataframe of Datavyu-exported codes and produces a report that includes the following for
    each code label timestamps:
    - missing offsets
    - offsets of zero
    - offsets labeled as before their corresponding onsets
    - whether or not the code starts at zero
    - whether or not the code ends with the video
    - overlapping onsets or offsets
    - a list of segments with potentially bad timestamps

    Parameters
    ----------
    codes_df: DataFrame
        A pandas DataFrame object that includes the Datavyu-exported values.
    video_length: int
        video length in milliseconds (output of get_video_length function)
    labels: list
        List of unique code labels included in codes_df (output of get_code_labels function)

    Returns
    -------
    summary_report: DataFrame
        A pandas dataframe with the report for each code in codes_df
    """
    summary_report = pd.DataFrame(index=labels, columns=['starts_at_zero', 'ends_with_video', 'num_bad_durations',
                                                         'segs_bad_duration', 'num_bad_onsets', 'segs_bad_onsets',
                                                         'num_bad_offsets', 'segs_bad_offsets'])

    for label in labels:
        label_df = codes_df.loc[:, [label + '.ordinal', label + '.onset', label + '.offset', label + '.code01']]
        label_df = label_df.dropna(axis=0, subset=[label + '.ordinal'])

        # check durations
        label_df['dur'] = label_df[label + '.offset'] - label_df[label + '.onset']
        dur = sum((label_df['dur'] <= 0))
        segs = label_df.loc[label_df['dur'] <= 0, 'dur'].index.astype(str).to_list()
        summary_report.loc[label, 'num_bad_durations'] = dur
        if len(segs) == 0:
            summary_report.loc[label, 'segs_bad_duration'] = 'None'
        else:
            summary_report.loc[label, 'segs_bad_duration'] = ','.join(segs)

        bad_onsets = []
        bad_offsets = []
        for i in label_df.index:
            # check onsets
            if i > 0:
                if label_df.loc[i, label + '.onset'] <= label_df.loc[i - 1, label + '.offset']:
                    bad_onsets.append(i)
                elif label_df.loc[i, label + '.onset'] >= video_length:
                    bad_onsets.append(i)
            elif i == 0:
                if label_df.loc[i, label + '.onset'] > 0:
                    summary_report.loc[label, 'starts_at_zero'] = 'No'
                elif label_df.loc[i, label + '.onset'] == 0:
                    summary_report.loc[label, 'starts_at_zero'] = 'Yes'
            # check offsets
            if i == label_df.index[-1]:
                if i == video_length:
                    summary_report.loc[label, 'ends_with_video'] = 'Yes'
                else:
                    summary_report.loc[label, 'ends_with_video'] = 'No'
            else:
                if label_df.loc[i, label + '.offset'] == 0:
                    bad_offsets.append(i)
                elif label_df.loc[i, label + '.offset'] < label_df.loc[i, label + '.onset']:
                    bad_offsets.append(i)
            if i > 0:
                if label_df.loc[i, label + '.offset'] < label_df.loc[i - 1, label + '.offset']:
                    bad_offsets.append(i)
        bad_offsets = np.unique(bad_offsets).astype(str)
        summary_report.loc[label, 'num_bad_offsets'] = len(bad_offsets)
        if len(bad_offsets) == 0:
            summary_report.loc[label, 'segs_bad_offsets'] = 'None'
        else:
            summary_report.loc[label, 'segs_bad_offsets'] = ','.join(bad_offsets)
        bad_onsets = np.unique(bad_onsets).astype(str)
        summary_report.loc[label, 'num_bad_onsets'] = len(bad_onsets)
        if len(bad_onsets) == 0:
            summary_report.loc[label, 'segs_bad_onsets'] = 'None'
        else:
            summary_report.loc[label, 'segs_bad_onsets'] = ','.join(bad_onsets)

    return summary_report


def values_report(codes_df, labels):
    """
    This function takes a dataframe of Datavyu-exported codes and produces a report that includes the following for
    each code label:
    - number of values coded
    - number of blank values found
    - list of code segments with blank values (correspond to cells in Datavyu)
    - list of unique values found

    Parameters
    ----------
    codes_df: DataFrame
        Pandas dataframe object with Datavyu CSV data.
    labels: list
        List of unique code labels included in codes_df (output of get_code_labels function)

    Returns
    -------
    summary_report: DataFrame
        A dataframe with the report for each code in codes_df

    """
    summary_report = pd.DataFrame(index=labels, columns=['num_values', 'unique_values', 'num_blank_cells',
                                                         'list_blank_cells'])

    for label in labels:
        label_df = codes_df.loc[:, [label + '.ordinal', label + '.code01']]
        label_df = label_df.dropna(axis=0, subset=[label + '.ordinal'])
        summary_report.loc[label, 'num_blank_cells'] = label_df[label + '.code01'].isna().sum()
        summary_report.loc[label, 'num_values'] = len(label_df) - summary_report.loc[label, 'num_blank_cells']
        unique_values = np.unique(label_df[label + '.code01'])
        unique_values = unique_values[~np.isnan(unique_values)].astype(str)
        summary_report.loc[label, 'unique_values'] =  ','.join(unique_values)
        blanks = []
        for i in label_df.index:
            if np.isnan(label_df.loc[i, label + '.code01']):
                blanks.append(str(i))
        if len(blanks) == 0:
            summary_report.loc[label, 'list_blank_cells'] = 'None'
        else:
            summary_report.loc[label, 'list_blank_cells'] = ','.join(blanks)
    return summary_report


def get_code_labels(codes_df):
    """
    Pull the unique labels from the Datavyu codes.

    Parameters
    ---------
    codes_df: DataFrame
        The dataframe of codes created by importing the Datavyu codes using pandas.

    Returns
    -------
    labels: list
        Variable names of the codes from Datavyu
    """

    labels = pd.Series(codes_df.columns).str.split(pat='.', expand=True)
    labels = list(labels[0].unique())
    for x in labels:
        if 'Unnamed' in x:
            labels.remove(x)
    return labels
    

def convert_timestamps(labels, codes_df, video_duration, sampling_rate, interpolate_gaps=True, do_log=False):
    """ This function performs two steps:
        1. Checks for human errors in coding such as incorrect end times or gaps in coding.
        2. Convert the timestamps to time series and optionally interpolate across gaps.

        Parameters
        ----------

        labels: list
            a list of strings which are the unique column variable labels in the dataframe.
            The output of the get_code_labels function.

        codes_df: DataFrame
            the dataframe of Datavyu codes

        video_duration: int
            The length of video in milliseconds, the output of get_video_length

        sampling_rate: int
            The sampling rate in Hz that the file should be saved as.

        interpolate_gaps: bool
            Default is set to True.  If you wish for gaps to be preserved as NaNs, set to False.

        do_log: bool
            Default is set to False. If logging is being used, set to True.

        Returns
        -------
        timeseries_df: DataFrame
            The resampled code time series
        """

    # set up dataframe object to store data
    timeseries_df = pd.DataFrame(columns=labels,
                                 index=range(0, video_duration+int(1000/sampling_rate), int(1000/sampling_rate)))
    timeseries_df.index.name = 'onset_ms'

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
            if interpolate_gaps and (timeseries_df[label].dtype == float or timeseries_df[label].dtype == int):
                msg = "Warning: there are {0}ms of interpolated codes for '{1}'".format(missing, label)
            else:
                msg = "Warning: there are {0}ms of missing codes for '{1}'".format(missing, label)
            if do_log:
                emolog.info(msg)
            print(msg)

    # interpolate gaps in codes if 'interpolate_gaps' is set to True
    if interpolate_gaps:
        timeseries_df = timeseries_df.interpolate(method='nearest', axis=0)
        
    return timeseries_df


def save_timeseries(timeseries_df, outfile_type, outfile_name, do_log=False):
    """
    Parameters
    ----------

    timeseries_df: DataFrame
        The resampled time series from validate_convert_timestamps
    outfile_type: str ('csv','excel','tab','space')
        the file type to save the time series as.
    outfile_name: str
        The file prefix for the output file.
    do_log: bool
        Default is set to False. If logging is being used, set to True.

    """

    # get the date and time right now
    today = datetime.now()
    
    if outfile_type == 'csv':
        # save as a csv
        out_file='{0}_{1}.csv'.format(outfile_name, today.strftime('%Y%m%d'))
        timeseries_df.to_csv(out_file, na_rep='NA')
    elif outfile_type == 'excel':
        # save as an excel file
        out_file='{0}_{1}.xlsx'.format(outfile_name, today.strftime('%Y%m%d'))
        timeseries_df.to_excel(out_file, na_rep='NA')
    elif outfile_type == 'tab':
        # save as a tab-delimited file
        out_file='{0}_{1}.txt'.format(outfile_name, today.strftime('%Y%m%d'))
        timeseries_df.to_csv(out_file, sep='\t', na_rep='NA')
    elif outfile_type == 'space':
        # save as a space-delimited file
        out_file='{0}_{1}.txt'.format(outfile_name, today.strftime('%Y%m%d'))
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
