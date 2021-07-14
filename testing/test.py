
video_file = '/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/Videos/The_Present_0321.mp4'
sampling_rate = 10
import os.path
from subprocess import check_call


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


def separate_voice_music(audio_file):
    """

    Parameters
    ----------
    audio_file : str
        The audio file to be processed

    Returns
    -------
    voice_file : str
        The file path to the separated voice track from the video file

    music_file : str
        The file path to the separated music track from the video file

    """

    out_folder = os.path.dirname(audio_file)
    check_call(['spleeter', 'separate','-p','spleeter:2stems', '-o', out_folder, audio_file])
    foldername = os.path.splitext(os.path.basename(audio_file))[0]
    voice_file = out_folder + '/' + foldername + '/vocals.wav'
    music_file = out_folder + '/' + foldername + '/accompaniment.wav'
    return(voice_file, music_file)


#audio_file = extract_sound_from_video(video_file)
audio_file = '/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/Videos/The_Present_0321_sound_short.mp3'
separate_voice_music(audio_file)
#from spleeter.separator import Separator

#separator = Separator('spleeter:2stems')
#out_folder = os.path.dirname(audio_file)
#separator.separate_to_file(audio_file, out_folder)
