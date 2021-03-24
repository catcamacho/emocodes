# import needed libraries
import pandas as pd
import numpy as np

# TODO: add function on edginess on image
# TODO: make video processing class


def get_video_length(video_file):
    """
    This function checks the length of a video file and returns that value in milliseconds.

    Parameters
    ----------
    video_file : str
        The path to the video file that was coded

    Return
    ------
    file_duration : int
        The duration of the file in milliseconds
    """
    from moviepy.editor import VideoFileClip
    
    clip = VideoFileClip(video_file)
    file_duration = int(clip.duration*1000)
    return file_duration


def compute_luminance(video_file, sampling_rate, video_duration):
    """
    This function produces a Series of the luminance values for a video file.

    Parameters
    ----------

    video_file : str
        the video file to be processed

    sampling_rate : int
        desired sampling rate of outputs in Hz

    video_duration : int
        length of video in milliseconds

    Return
    ------
    lum_series : Series
        The Pandas Series of the luminance measure, resampled according to the user input.
    """

    import cv2

    video = cv2.VideoCapture(video_file)
    frames_lum = []

    end = False
    while not end:
        r, f = video.read()
        if r == 1:
            t = f.mean(axis=0).mean(axis=0)
            lum = 0.299*t[0] + 0.587*t[1] + 0.114*t[2]  # formula from https://www.w3.org/TR/AERT/#color-contrast
            frames_lum.append(lum)
        else:
            end = True

    fps = (len(frames_lum)*1000)/video_duration
    a = np.arange(0, (len(frames_lum)/fps)*1000, 1000/fps)
    b = pd.Series(frames_lum, index=pd.to_datetime(a, unit='ms'), name='luminance')
    b.index.name = 'time'
    lum_series = b.resample('{0}ms'.format(1000/sampling_rate)).mean()

    return lum_series
