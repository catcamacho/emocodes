import pandas as pd
from emocodes import hrf_convolve_features, pairwise_ips, pairwise_corr, vif_collineary, feature_freq_power
from fpdf import FPDF
import os.path

features = pd.read_csv('/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/processing/low_level/movieDM_low_av_features.csv', index_col=0)
time_col='onset'
units='s'
column_names=['brightness', 'frac_high_saliency', 'sharpness', 'vibrance','rms']
sampling_rate=1.2
convolve_hrf = True

if convolve_hrf:
    hrf_conv_features = hrf_convolve_features(features, column_names=column_names,
                                              time_col=time_col)

ips_df, ips = pairwise_ips(hrf_conv_features, column_names=column_names)
corrmat_df = pairwise_corr(hrf_conv_features, column_names=column_names, nan_policy='omit')
vif = vif_collineary(hrf_conv_features, column_names=column_names)
power = feature_freq_power(hrf_conv_features, time_col=time_col, units=units,
                           column_names=column_names, sampling_rate=sampling_rate)

# compile into a report
pdf = FPDF('P', 'mm', 'A4')
pdf.add_page()
pdf.set_font('arial', 'B', 18)
pdf.image(os.path.abspath('logos/circle_bw.png'), 10, 8, 20)
pdf.cell(65, 20, 'Feature Report', 0, 0, 'R')
pdf.ln(10)
pdf.set_font('arial', 'B', 14)
pdf.cell(20, 20, 'Mean Pair-Wise Instantaneous Phase Synchrony', 0, 0, 'L')

pdf.output('test.pdf')
