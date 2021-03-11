# import needed libraries
import pandas as pd
from datetime import datetime
from os.path import abspath

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
    
    
# validate and convert onset/offset times to a timeseries
def validate_convert_timestamps(labels, codes_df, video_duration, sampling_rate, interpolate_gaps=True):
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

        Returns
        -------
        timeseries_df : DataFrame
            The resampled code time series
        """
    
    # set up dataframe object to store data
    timeseries_df = pd.DataFrame(columns=labels, index=range(int(1000/sampling_rate), video_duration+int(1000/sampling_rate), int(1000/sampling_rate)))
    timeseries_df.index.name = 'time'

    for label in labels:
        label_df = codes_df[[label+'.onset', label+'.offset', label+'.code01']].dropna(axis=0, how='any')

        # check if offsets precede onsets
        dur = label_df[label+'.offset'] - label_df[label+'.onset']
        for d in dur:
            if d <= 0:
                raise ValueError("ERROR: code '{0}' has an offset time that is before the corresponding onset time.".format(label))

        # check that the offset for the last code is not after the end of the episode
        if label_df.loc[label_df.index[-1], label+'.offset'] > video_duration:
            print("Warning: The last offset for code '{0}' is after the end of the video. Correcting.".format(label))
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
                print("Warning: there are {0}ms of interpolated codes for '{1}'".format(missing, label))
            else:
                print("Warning: there are {0}ms of missing codes for '{1}'".format(missing, label))

    # interpolate gaps in codes if 'interpolate_gaps' is set to True
    if interpolate_gaps is True:
        timeseries_df = timeseries_df.interpolate(method='nearest', axis=0)
        
    return timeseries_df


def save_timeseries(timeseries_df, outfile_type, outfile_name):
    """
    Parameters
    ----------

    timeseries_df : DataFrame
        The resampled time series from validate_convert_timestamps

    outfile_type : str ('csv','excel','tab','space')
        the file type to save the time series as.

    outfile_name : str
        The file prefix for the output file.

    Returns
    -------

    None

    """

    # get the date and time right now
    today = datetime.now()
    
    if outfile_type == 'csv':
        # save as a csv
        timeseries_df.to_csv('{0}_{1}.csv'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S')), na_rep='NA')
    elif outfile_type == 'excel':
        # save as an excel file
        timeseries_df.to_excel('{0}_{1}.xlsx'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S')), na_rep='NA')
    elif outfile_type == 'tab':
        # save as a tab-delimited file
        timeseries_df.to_csv('{0}_{1}.txt'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S')), sep='\t', na_rep='NA')
    elif outfile_type == 'space':
        # save as a space-delimited file
        timeseries_df.to_csv('{0}_{1}.txt'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S')), sep='  ', na_rep='NA')
    else:
        print('Warning: data note saved! Please indicate the file format: csv, excel, tab, space')
    
    filepath = abspath('{0}_{1}.txt'.format(outfile_name, today.strftime('%Y%m%d-%H%M%S')))
    print('Code timeseries saved at {0}'.format(filepath))
