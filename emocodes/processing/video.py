# import needed libraries
import pandas as pd
import librosa
import numpy as np
import os.path
from subprocess import check_call


class ExtractVideoFeatures:
    """

    """

    def __init__(self, sampling_rate=1):
        """

        Parameters
        ----------
        sampling_rate : float
            The sampling rate in Hz
        """

        self.sampling_rate = sampling_rate
        self.resampled_video = None
        self.audio_features_df = None
        self.visual_features_df = None
        self.video_features_df = None

    def extract_features(self, video_file):
        """

        Parameters
        ----------
        video_file : str
            The filepath to the video file to be processed

        Returns
        -------

        """
        self.extract_visual_features(video_file)
        self.extract_audio_features(video_file)
        return self

    def extract_audio_features(self, video_file):
        """

        Parameters
        ----------
        video_file : str
            The filepath to the video file to be processed

        Returns
        -------

        """
        print("extracting audio features...")
        self.audio_features_df = extract_audio_features(video_file, self.sampling_rate)
        print("done!")
        return self

    def extract_visual_features(self, video_file):
        """

        Parameters
        ----------
        video_file : str
            The filepath to the video file to be processed

        Returns
        -------

        """
        print("extracting video features...")
        self.resampled_video = resample_video(video_file, self.sampling_rate)
        self.visual_features_df = extract_visual_features(self.resampled_video)
        print("done!")
        return self

    def combine_audio_visual_dfs(self, outfile_name):

        return self


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
    file_duration = int(clip.duration * 1000)
    return file_duration


def resample_video(video_file, sampling_rate):
    """

    Parameters
    ----------
    video_file : str
        file path to video to be resampled.
    sampling_rate : float
        Desired sampling rate in Hz

    Returns
    -------
    resampled_video : pliers video object with resampled video frames

    """

    from pliers.stimuli import VideoStim
    from pliers.filters import FrameSamplingFilter

    video = VideoStim(video_file)
    resamp_filter = FrameSamplingFilter(hertz=sampling_rate)
    resampled_video = resamp_filter.transform(video)

    return resampled_video


def extract_visual_features(video_frames):
    """
    This function extracts luminance, vibrance, saliency, and sharpness from the frames of a video
    using the pliers library. If you use this function, please cite the pliers library directly:
    https://github.com/PsychoinformaticsLab/pliers#how-to-cite

    Parameters
    ----------
    video_frames : object
        pliers video object with frames to analyze

    Returns
    -------
    low_level_video_df : DataFrame
        Pandas dataframe with a column per low-level feature.py (index is time).
    """

    # extract video luminance
    from pliers.extractors import BrightnessExtractor
    brightext = BrightnessExtractor()
    brightres = brightext.transform(video_frames)
    brightres_df = pd.DataFrame(columns=brightres[0].to_df().columns)
    for a, ob in enumerate(brightres):
        t = ob.to_df()
        t['order'] = a
        brightres_df = brightres_df.append(t, ignore_index=True)

    # extract saliency
    from pliers.extractors import SaliencyExtractor
    salext = SaliencyExtractor()
    salres = salext.transform(video_frames)
    salres_df = pd.DataFrame(columns=salres[0].to_df().columns)
    for a, ob in enumerate(salres):
        t = ob.to_df()
        t['order'] = a
        salres_df = salres_df.append(t, ignore_index=True)

    # extract sharpness
    from pliers.extractors import SharpnessExtractor
    sharpext = SharpnessExtractor()
    sharpres = sharpext.transform(video_frames)
    sharpres_df = pd.DataFrame(columns=sharpres[0].to_df().columns)
    for a, ob in enumerate(sharpres):
        t = ob.to_df()
        t['order'] = a
        sharpres_df = sharpres_df.append(t, ignore_index=True)

    # extract vibrance
    from pliers.extractors import VibranceExtractor
    vibext = VibranceExtractor()
    vibres = vibext.transform(video_frames)
    vibres_df = pd.DataFrame(columns=vibres[0].to_df().columns)
    for a, ob in enumerate(vibres):
        t = ob.to_df()
        t['order'] = a
        vibres_df = vibres_df.append(t, ignore_index=True)

    # combine into 1 dataframe
    low_level_video_df = brightres_df.merge(salres_df[salres_df.columns[4:]], left_index=True, right_index=True)
    low_level_video_df = low_level_video_df.merge(sharpres_df[sharpres_df.columns[4:]], 
                                                  left_index=True, right_index=True)
    low_level_video_df = low_level_video_df.merge(vibres_df[vibres_df.columns[4:]], left_index=True, right_index=True)
    low_level_video_df['time_ms'] = low_level_video_df['onset']*1000
    return low_level_video_df


def extract_sound_from_video(video_file):
    """

    Parameters
    ----------
    video_file : str
        File path to the video file to be processed.

    Returns
    -------
    audio_file : str
        file path to the extractive video audio file.
    """
    from moviepy.editor import VideoFileClip

    file_name, ext = os.path.splitext(video_file)
    video = VideoFileClip(video_file)
    audio_file = file_name + '_sound.mp3'
    video.audio.write_audiofile(audio_file)
    return audio_file


def extract_audio_features(in_file, sampling_rate):
    """
    This function extracts audio intensity, tempo, and beats from the audio of a video file using the pliers library.
    If you use this function, please cite the pliers library directly:
    https://github.com/PsychoinformaticsLab/pliers#how-to-cite

    Parameters
    ----------
    in_file : str
        file path to video or audio file to be processed
    sampling_rate : float
        the desired sampling rate in Hz for the output

    Returns
    -------
    low_level_audio_df : DataFrame
        Pandas dataframe with a column per low-level feature.py (index is time).
    """

    resample_rate = int(1000 / sampling_rate)

    # compute tempo on a continuous basis
    y, sr = librosa.load(in_file)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    time_seconds = np.arange(1, len(y)) / sr
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, ac_size=1, aggregate=None)
    n_samples = len(dtempo)
    dtempo_samprate = time_seconds[-1] / n_samples
    time_ms = np.arange(0, time_seconds[-1], dtempo_samprate) * 1000
    dtempo_df = pd.DataFrame(dtempo, columns=['dynamic_tempo'], index=pd.to_datetime(time_ms, unit='ms'))
    dtempo_df = dtempo_df.resample('{0}ms'.format(resample_rate)).mean()

    # audio RMS to capture changes in intensity
    from pliers.extractors import RMSExtractor
    rmsext = RMSExtractor()
    rmsres = rmsext.transform(in_file)
    rmsres_df = rmsres.to_df()

    # Identify major beats in audio
    from pliers.extractors import BeatTrackExtractor
    btext = BeatTrackExtractor()
    bteres = btext.transform(in_file)
    bteres_df = bteres.to_df()

    # combine features into one dataframe
    low_level_audio_df = pd.DataFrame()
    low_level_audio_df['time_ms'] = rmsres_df['onset'] * 1000
    low_level_audio_df['rms'] = rmsres_df['rms']
    low_level_audio_df['beats'] = 0
    for b in bteres_df['beat_track']:
        low_level_audio_df.loc[b, 'beats'] = 1

    low_level_audio_df.index = pd.to_datetime(rmsres_df['onset'], unit='s')
    low_level_audio_df = low_level_audio_df.resample('{0}ms'.format(resample_rate)).mean()
    low_level_audio_df['onset_ms'] = low_level_audio_df['time_ms'] - low_level_audio_df['time_ms'][0]
    low_level_audio_df['beats'][low_level_audio_df['beats'] > 0] = 1
    low_level_audio_df['dynamic_tempo'] = dtempo_df['dynamic_tempo']
    low_level_audio_df.index = range(0, low_level_audio_df.shape[0])
    low_level_audio_df.index.name = None

    return low_level_audio_df
