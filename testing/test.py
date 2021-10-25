# import needed libraries
import numpy as np
import pandas as pd
from datetime import datetime
import os
from emocodes.processing.video import get_video_length
import markdown
import weasyprint as wp


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


def values_report(codes_df, labels):

    return summary_report


in_file = '/Users/catcamacho/Documents/GitHub/emocodes/testing/data/invalid_codes_clip1.csv'
video_file = '/Users/catcamacho/Documents/GitHub/emocodes/testing/data/sample_clip1.mp4'
video_length = get_video_length(video_file)
codes = pd.read_csv(in_file, index_col=None)
labels = get_code_labels(codes)

