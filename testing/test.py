import pandas as pd
#from emocodes.processing.video import ExtractVideoFeatures, resample_video

#evf = ExtractVideoFeatures(sampling_rate=1)
#evf.extract_visual_features(video_file='/Users/catcamacho/Box/EmoCodes_project/reliability_data/episodes/AHKJ_S1E2.mp4')
#evf.extract_audio_features(video_file='/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/Videos/The_Present_0321.mp4')

#evf.visual_features_df.to_csv('/Users/catcamacho/Box/EmoCodes_project/reliability_data/episodes/AHKJ_visual_features.csv')
#evf.audio_features_df.to_csv('movieDM_audio_features.csv')

#video_frames = evf.resampled_video
#video_frames = resample_video(video_file='/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/Videos/The_Present_0321.mp4',sampling_rate=1 / 0.8)

# extract visual flow
#from pliers.extractors import FarnebackOpticalFlowExtractor

#opflow = FarnebackOpticalFlowExtractor()
#ofres = opflow.transform('/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/Videos/Despicable_Me_1000.mp4')

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
    low_level_video_df = low_level_video_df.merge(sharpres_df[sharpres_df.columns[4:]], left_index=True, right_index=True)
    low_level_video_df = low_level_video_df.merge(vibres_df[vibres_df.columns[4:]], left_index=True, right_index=True)
    low_level_video_df['time_ms'] = low_level_video_df['onset']*1000
    return low_level_video_df

video_file = '/Users/catcamacho/Box/CCP/EmoCodes_project/reliability_data/episodes/MLP_S8E3_20.mp4'
video_frames = resample_video(video_file, 1)
features_df = extract_visual_features(video_frames)
out_file = '/Users/catcamacho/Box/CCP/EmoCodes_project/reliability_data/episodes/MLP_visual_features.csv'
features_df.to_csv(out_file)
