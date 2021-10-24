import pandas as pd
import librosa
import numpy as np

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
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, ac_size=1, aggregate=None)
    n_samples = len(dtempo)
    dtempo_samp_rate = time_seconds[-1] / n_samples
    time_ms = np.arange(0, time_seconds[-1], dtempo_samp_rate) * 1000
    dtempo_df = pd.DataFrame(dtempo, columns=['dynamic_tempo'], index=pd.to_datetime(time_ms, unit='ms'))

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
    low_level_audio_df['onset_ms'] = low_level_audio_df['onset_ms'] - low_level_audio_df['onset_ms'][0]
    low_level_audio_df['beats'][low_level_audio_df['beats'] > 0] = 1
    low_level_audio_df['dynamic_tempo'] = dtempo_df['dynamic_tempo']
    low_level_audio_df.index.name = None
    print('Auditory feature extraction complete.')
    return low_level_audio_df


video_file = '/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/Videos/The_Present_0321.mp4'

audio_df = extract_audio_features(video_file)
print(audio_df.head())