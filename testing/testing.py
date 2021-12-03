import emocodes as ec
import pandas as pd

video_file = '/Users/catcamacho/Documents/GitHub/emocodes/testing/data/sample_clip1.mp4' # must be MP4
sampling_rate = 10 # in Hz

v = ec.ExtractVideoFeatures().extract_features(video_file, sampling_rate)