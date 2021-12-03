# import needed libraries
import pandas as pd
import librosa
import numpy as np
import os.path


class ExtractVideoFeatures:
    """
    This class can be used to extract the following low-level features from an MP4 file:

    - **Luminance**: The frame-by-frame brightness level
    - **Vibrance**: The variance of color channels of each frame
    - **Saliency**: Fraction of highly salient visual information for each frame according to the Itti & Koch \
    algorithm: https://doi.org/10.1109/34.730558
    - **Sharpness**: Degree of blur or sharpness of each frame
    - **Dynamic Tempo**: the rolling tempo of the audio track
    - **Loudness**: Operationalized as the root-mean-square of the audio amplitude
    - **Beats**: if a musical beat falls on that timestamp. For files of less than 30Hz, this variable is likely not \
    useful.

    Example Usage:
        >>> import emocodes as ec
        >>> video_file = 'video.mp4'
        >>> sampling_rate = 5 # in Hz
        >>> outfile = 'outputs/video_features'
        >>> features_df = ec.ExtractVideoFeatures().extract_features(video_file, sampling_rate, outfile)
    """

    def __init__(self):
        self.sampling_rate = None
        self.video = None
        self.audio_features_df = None
        self.visual_features_df = None
        self.combined_df = None
        self.resampled_features = None

    def extract_features(self, video_file, sampling_rate=1, outfile=None):
        """
        This method extracts the frame-by-frame visual and aural features from an MP4 file.

        Parameters
        ----------
        video_file: str
            The filepath to the video file to be processed. Must be MP4
        sampling_rate: float
            The desired output sampling rate in Hz.
        outfile: str
            Optional. The desired output filename for the features CSV. If None, defaults to the path and name of the
            MP4 video file with '.mp4' replaced with '_features.csv'
        """

        self.sampling_rate = sampling_rate
        self.video = video_file
        self.extract_visual_features(self.video)
        self.extract_audio_features(self.video)
        self.resample_features(self.sampling_rate)

        # harmonize time scales and columns across dataframes and combine
        self.audio_features_df.index = self.audio_features_df['onset_ms']
        self.audio_features_df.index.name = 'time_ms'
        self.audio_features_df = self.audio_features_df.drop('onset_ms', axis=1)
        self.visual_features_df.index = self.visual_features_df['onset_ms']
        self.visual_features_df.index.name = 'time_ms'
        self.visual_features_df = self.visual_features_df.drop(['onset_ms','duration'], axis=1)
        self.combined_df = self.visual_features_df.merge(self.audio_features_df, left_index=True, right_index=True)

        # save features as a csv
        if not outfile:
            outfile = video_file.replace('.mp4', '')

        self.combined_df.to_csv(outfile + '_features.csv')
        return self

    def extract_audio_features(self, video_file):
        """
        This method extracts the frame by frame audio features from a video input.

        Parameters
        ----------
        video_file: str
            The filepath to the video file to be processed. Must be MP4.

        """
        print("extracting audio features...")
        self.audio_features_df = extract_audio_features(video_file)
        print("done!")
        return self

    def extract_visual_features(self, video_file):
        """
        This method extracts the visual features from the video input.
        Parameters
        ----------
        video_file: str
            The filepath to the video file to be processed. Must be MP4

        """
        print("extracting video features...")
        self.video = video_file
        self.visual_features_df = extract_visual_features(self.video)
        print("done!")
        return self

    def resample_features(self, sampling_rate):
        """
        This method resamples the available feature dataframes.

        """
        self.sampling_rate = sampling_rate
        resample = 1000/sampling_rate
        if isinstance(self.visual_features_df, pd.DataFrame):
            self.visual_features_df = resample_df(self.visual_features_df, sampling_rate, time_col_units='ms')
            if 'onset_ms' in self.visual_features_df.columns:
                self.visual_features_df['onset_ms'] = self.visual_features_df['onset_ms'] - \
                                                      self.visual_features_df['onset_ms'][0]

        if isinstance(self.audio_features_df, pd.DataFrame):
            self.audio_features_df = resample_df(self.audio_features_df, sampling_rate, time_col_units='ms')
            if 'onset_ms' in self.audio_features_df.columns:
                self.audio_features_df['onset_ms'] = self.audio_features_df['onset_ms'] - \
                                                     self.audio_features_df['onset_ms'][0]
                # fix rounding
                self.audio_features_df['onset_ms'] = \
                    round(self.audio_features_df['onset_ms']/resample).astype(int) * resample

        return self


def resample_df(df, sampling_rate, time_col=None, time_col_units='ms'):
    """
    This function resamples an input dataframe to a desired sampling rate. The default parameters assume that the index
    of the dataframe is a DateTimeIndex. If that is not the case, the time_col and time_col_units parameters are
    required.

    Parameters
    ----------
    df: DataFrame
        The dataframe to be resampled.
    sampling_rate: float
        The desired output sampling rate in Hz
    time_col: str
        The column label containing time information for the dataframe. If the index is a DateTimeIndex object, this
        parameter is not required.
    time_col_units: 'ms', 's', 'm', or 'h'
        The time units (milliseconds, seconds, minutes, or hours). If the index is a DateTimeIndex object, this
        parameter is not required.

    Returns
    -------
    resampled_df: DataFrame
        The output resamples DataFrame (with a DateTime index)
    """

    resample_rate = 1000/sampling_rate

    if not time_col:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError('dataframe does not have Datetime index and no time column is specified. Cannot resample.')
        else:
            resampled_df = df.resample('{0}ms'.format(resample_rate)).mean()
    else:
        if time_col_units == 'ms':
            df.index = pd.to_datetime(df[time_col], unit='ms')
            resampled_df = df.resample('{0}ms'.format(resample_rate)).mean()
            resampled_df[time_col] = resampled_df[time_col] - resampled_df.loc[0, time_col]
        elif time_col_units == 's':
            df.index = pd.to_datetime(df[time_col], unit='s')
            resampled_df = df.resample('{0}ms'.format(resample_rate)).mean()
            resampled_df[time_col] = resampled_df[time_col] - resampled_df.loc[0, time_col]
        elif time_col_units == 'm':
            df.index = pd.to_datetime(df[time_col], unit='m')
            resampled_df = df.resample('{0}ms'.format(resample_rate)).mean()
            resampled_df[time_col] = resampled_df[time_col] - resampled_df.loc[0, time_col]
        elif time_col_units == 'h':
            df.index = pd.to_datetime(df[time_col], unit='h')
            resampled_df = df.resample('{0}ms'.format(resample_rate)).mean()
            resampled_df[time_col] = resampled_df[time_col] - resampled_df.loc[0, time_col]
        else:
            raise ValueError('The time_col_units variable must be either ms, s, m, or h denoting milliseconds, \
            seconds,minutes, or hours respectively.')

    return resampled_df


def get_video_length(video_file):
    """
    This function checks the length of a video file and returns that value in milliseconds.

    Parameters
    ----------
    video_file: str
        The path to the video file that was coded

    Return
    ------
    file_duration: float
        The duration of the file in milliseconds
    """
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(video_file)
    file_duration = int(clip.duration * 1000)
    return file_duration


def resample_video(video_file, sampling_rate):
    """
    This function resamples a video to the desired sampling rate.  Can be useful for making video with high sampling
    rates more tractable for analysis.

    Parameters
    ----------
    video_file: str
        file path to video to be resampled.
    sampling_rate: float
        Desired sampling rate in Hz

    Returns
    -------
    resampled_video: pliers video object with resampled video frames

    """

    from pliers.stimuli import VideoStim
    from pliers.filters import FrameSamplingFilter

    video = VideoStim(video_file)
    resamp_filter = FrameSamplingFilter(hertz=sampling_rate)
    resampled_video = resamp_filter.transform(video)

    return resampled_video


def extract_visual_features(video_file):
    """
    This function extracts luminance, vibrance, saliency, and sharpness from the frames of a video
    using the pliers library. If you use this function, please cite the pliers library directly:
    https://github.com/PsychoinformaticsLab/pliers#how-to-cite

    Parameters
    ----------
    video_file: str
        Path to video file to analyze.

    Returns
    -------
    low_level_video_df: DataFrame
        Pandas dataframe with a column per low-level feature.py (index is time).
    """

    # extract video luminance
    print('Extracting brightness...')
    from pliers.extractors import BrightnessExtractor
    brightext = BrightnessExtractor()
    brightres = brightext.transform(video_file)
    brightres_df = pd.DataFrame(columns=brightres[0].to_df().columns)
    for a, ob in enumerate(brightres):
        t = ob.to_df()
        t['order'] = a
        brightres_df = brightres_df.append(t, ignore_index=True)

    # extract saliency
    print('Extracting saliency...')
    from pliers.extractors import SaliencyExtractor
    salext = SaliencyExtractor()
    salres = salext.transform(video_file)
    salres_df = pd.DataFrame(columns=salres[0].to_df().columns)
    for a, ob in enumerate(salres):
        t = ob.to_df()
        t['order'] = a
        salres_df = salres_df.append(t, ignore_index=True)

    # extract sharpness
    print('Extracting sharpness...')
    from pliers.extractors import SharpnessExtractor
    sharpext = SharpnessExtractor()
    sharpres = sharpext.transform(video_file)
    sharpres_df = pd.DataFrame(columns=sharpres[0].to_df().columns)
    for a, ob in enumerate(sharpres):
        t = ob.to_df()
        t['order'] = a
        sharpres_df = sharpres_df.append(t, ignore_index=True)

    # extract vibrance
    print('Extracting vibrance...')
    from pliers.extractors import VibranceExtractor
    vibext = VibranceExtractor()
    vibres = vibext.transform(video_file)
    vibres_df = pd.DataFrame(columns=vibres[0].to_df().columns)
    for a, ob in enumerate(vibres):
        t = ob.to_df()
        t['order'] = a
        vibres_df = vibres_df.append(t, ignore_index=True)

    # combine into 1 dataframe
    print('Combining data...')
    low_level_video_df = brightres_df.merge(salres_df[salres_df.columns[4:]], left_index=True, right_index=True)
    low_level_video_df = low_level_video_df.merge(sharpres_df[sharpres_df.columns[4:]],
                                                  left_index=True, right_index=True)
    low_level_video_df = low_level_video_df.merge(vibres_df[vibres_df.columns[4:]], left_index=True, right_index=True)
    low_level_video_df['onset_ms'] = low_level_video_df['onset']*1000
    low_level_video_df.index = pd.to_datetime(low_level_video_df['onset_ms'], unit='ms')
    low_level_video_df = low_level_video_df.drop(['max_saliency', 'max_y', 'max_x',
                                                  'onset', 'object_id','order'], axis=1)
    low_level_video_df.index.name = None
    print('Visual feature extraction complete.')
    return low_level_video_df


def extract_sound_from_video(video_file):
    """
    This function pulls just the audio track from a video MP4 file and saves it as an MP3.

    Parameters
    ----------
    video_file: str
        File path to the video file to be processed.

    Returns
    -------
    audio_file: str
        file path to the extractive video audio file.
    """
    from moviepy.editor import VideoFileClip

    file_name, ext = os.path.splitext(video_file)
    video = VideoFileClip(video_file)
    audio_file = file_name + '_sound.mp3'
    video.audio.write_audiofile(audio_file)
    return audio_file


def extract_audio_features(in_file):
    """
    This function extracts audio intensity, tempo, and beats from the audio of a video file using the pliers library.
    If you use this function, please cite the pliers library directly:
    https://github.com/PsychoinformaticsLab/pliers#how-to-cite

    Parameters
    ----------
    in_file : str
        file path to video or audio file to be processed

    Returns
    -------
    low_level_audio_df : DataFrame
        Pandas dataframe with a column per low-level feature (index is time).
    """

    # compute tempo on a continuous basis
    print('Extracting dynamic tempo...')
    y, sr = librosa.load(in_file)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    time_seconds = np.arange(1, len(y)) / sr
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, ac_size=10, aggregate=None)
    n_samples = len(dtempo)
    dtempo_samp_rate = time_seconds[-1] / n_samples
    time_ms = np.arange(0, time_seconds[-1], dtempo_samp_rate) * 1000
    dtempo_df = pd.DataFrame(dtempo, columns=['dynamic_tempo'], index=pd.to_datetime(time_ms, unit='ms'))
    resamp_dtempo_df = dtempo_df.resample('10ms').mean()
    if resamp_dtempo_df['dynamic_tempo'].isnull().values.any():
        resamp_dtempo_df = dtempo_df.resample('10ms').bfill()

    # audio RMS to capture changes in intensity
    print('Extracting loudness...')
    from pliers.extractors import RMSExtractor
    rmsext = RMSExtractor()
    rmsres = rmsext.transform(in_file)
    rmsres_df = rmsres.to_df()

    # Identify major beats in audio
    print('Extracting major music beats...')
    from pliers.extractors import BeatTrackExtractor
    btext = BeatTrackExtractor()
    bteres = btext.transform(in_file)
    bteres_df = bteres.to_df()

    # combine features into one dataframe
    print('Aggregating data...')
    low_level_audio_df = pd.DataFrame()
    low_level_audio_df['onset_ms'] = rmsres_df['onset'] * 1000
    low_level_audio_df['rms'] = rmsres_df['rms']
    low_level_audio_df['beats'] = 0
    for b in bteres_df['beat_track']:
        low_level_audio_df.loc[b, 'beats'] = 1

    low_level_audio_df.index = pd.to_datetime(rmsres_df['onset'], unit='s')
    low_level_audio_df = low_level_audio_df.resample('10ms').mean()
    low_level_audio_df['onset_ms'] = low_level_audio_df['onset_ms'] - low_level_audio_df['onset_ms'][0]
    low_level_audio_df['dynamic_tempo'] = resamp_dtempo_df['dynamic_tempo']
    low_level_audio_df.index.name = None
    print('Auditory feature extraction complete.')
    return low_level_audio_df