import pandas as pd
from emocodes import hrf_convolve_features, pairwise_ips, pairwise_corr, vif_collineary
from fpdf import FPDF
import os.path
from os import makedirs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='white')

features = pd.read_csv('/Users/catcamacho/Box/CCP/HBN_study/HBN_video_coding/processing/low_level/movieDM_low_av_features.csv', index_col=0)
time_col='onset'
units='s'
column_names=['brightness', 'frac_high_saliency', 'sharpness', 'vibrance', 'rms']
sampling_rate=1.2
convolve_hrf = True
makedirs('./figs', exist_ok=True)

if convolve_hrf:
    hrf_conv_features = hrf_convolve_features(features, column_names=column_names,
                                              time_col=time_col)

# plot IPS data
ips_df, ips = pairwise_ips(hrf_conv_features, column_names=column_names)
height = len(ips_df)*0.9
plt.figure(figsize=(len(ips_df)+1.5, height))
ax = sns.heatmap(ips_df, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha='right')
plt.tight_layout()
plt.savefig('./figs/ips_heatmap.png')
plt.close()

# plot Corr data
corrmat_df = pairwise_corr(hrf_conv_features, column_names=column_names, nan_policy='omit')
plt.figure(figsize=(len(ips_df)+1.5, height))
ax = sns.heatmap(corrmat_df, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha='right')
plt.tight_layout()
plt.savefig('./figs/corrmat.png')
plt.close()

# plot VIF scores
vif = vif_collineary(hrf_conv_features, column_names=column_names)
plt.figure(figsize=(len(vif), 5))
ax = vif.plot(kind='bar')
ax.axhline(2, color = 'blue')
ax.axhline(5, color = 'green')
ax.axhline(10, color = 'red')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha='right')
plt.tight_layout()
plt.savefig('./figs/vif.png')
plt.close()

# compile into a report
pdf = FPDF('P', 'mm', 'A4')
pdf.add_page()
pdf.set_font('arial', 'B', 18)
pdf.image(os.path.abspath('../logos/circle_bw.png'), 10, 8, 20)
pdf.cell(80, 20, 'Feature Report - Page 1', 0, 0, 'R')
pdf.ln(10)
pdf.set_font('arial', 'B', 14)
pdf.cell(20, 40, 'Mean Pair-Wise Instantaneous Phase Synchrony', 0, 0, 'L')
pdf.image('./figs/ips_heatmap.png', 20, 50, 80)
pdf.add_page()
pdf.set_font('arial', 'B', 18)
pdf.image(os.path.abspath('../logos/circle_bw.png'), 10, 8, 20)
pdf.cell(80, 20, 'Feature Report - Page 2', 0, 0, 'R')
pdf.set_font('arial', 'B', 14)
pdf.cell(20, 40, 'Pair-Wise Correlation', 0, 0, 'L')
pdf.image('./figs/corrmat.png', 20, 50, 80)
pdf.add_page()
pdf.set_font('arial', 'B', 18)
pdf.image(os.path.abspath('../logos/circle_bw.png'), 10, 8, 20)
pdf.cell(80, 20, 'Feature Report - Page 3', 0, 0, 'R')
pdf.set_font('arial', 'B', 14)
pdf.cell(20, 40, 'Variance Inflation Factor', 0, 0, 'L')
pdf.image('./figs/vif.png', 20, 50, 80)
pdf.output('test.pdf')
