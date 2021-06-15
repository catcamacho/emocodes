# import needed libraries
import pandas as pd
import numpy as np

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
    file_duration : float
        The duration of the file in milliseconds
    """
    from moviepy.editor import VideoFileClip
    
    clip = VideoFileClip(video_file)
    file_duration = int(clip.duration*1000)
    return file_duration


def extract_visual_features(video_file, sampling_rate, video_duration):
    """
    This function produces a Series of the luminance values for a video file.

    Parameters
    ----------

    video_file : str
        the video file to be processed

    sampling_rate : float
        desired sampling rate of outputs in Hz

    video_duration : float
        length of video in milliseconds

    Return
    ------
    lum_df : DataFrame
        The pandas dataframe containing luminance and RGB time series at sampling_rate
    """

    from cv2 import VideoCapture

    video = VideoCapture(video_file)
    frames_lum = []
    rgb = []

    end = False
    while not end:
        r, f = video.read()
        if r == 1:
            t = f.mean(axis=0).mean(axis=0)
            lum = 0.299*t[0] + 0.587*t[1] + 0.114*t[2]
            rgb.append(t)
            frames_lum.append(lum)
        else:
            end = True

    fps = (len(frames_lum)*1000)/video_duration
    a = np.arange(0, (len(frames_lum)/fps)*1000, 1000/fps)
    b = pd.DataFrame(frames_lum, index=pd.to_datetime(a, unit='ms'), columns=['luminance'])
    b[['R','G','B']] = rgb
    b.index.name = 'time_ms'
    lum_df = b.resample('{0}ms'.format(1000/sampling_rate)).mean()

    return lum_df


def extract_audio_features(video_file, sampling_rate):
    """
    This function extracts audio intensity (continuous variable) and beats (categorical) from the audio of a video file using the pliers library.
    If you use this function, please cite the pliers library directly:
    https://github.com/PsychoinformaticsLab/pliers#how-to-cite

    Parameters
    ----------
    video_file : str
        file path to video file to be processed
    sampling_rate : float
        the desired sampling rate in Hz for the output

    Returns
    -------
    low_level_audio_df : DataFrame
        Pandas dataframe with a column per low-level feature (index is time).
    """
    resample_rate = 1000 / sampling_rate

    # audio RMS to capture changes in intensity
    from pliers.extractors import RMSExtractor
    rmsext = RMSExtractor()
    rmsres = rmsext.transform(video_file)
    rmsres_df = rmsres.to_df()

    # Identify major beats in audio
    from pliers.extractors import BeatTrackExtractor
    btext = BeatTrackExtractor()
    bteres = btext.transform(video_file)
    bteres_df = bteres.to_df()

    # combine and resample
    low_level_audio_df = pd.DataFrame()
    low_level_audio_df['time_ms'] = rmsres_df['onset'] * 1000
    low_level_audio_df['rms'] = rmsres_df['rms']
    low_level_audio_df['beats'] = 0
    for b in bteres_df['beat_track']:
        low_level_audio_df.loc[b,'beats'] = 1

    low_level_audio_df.index = pd.to_datetime(rmsres_df['onset'], unit='s')
    low_level_audio_df = low_level_audio_df.resample('{0}ms'.format(resample_rate)).mean()
    low_level_audio_df['beats'][low_level_audio_df['beats']>0] = 1
    low_level_audio_df.index = range(0,low_level_audio_df.shape[0])
    low_level_audio_df.index.name = None

    return low_level_audio_df
