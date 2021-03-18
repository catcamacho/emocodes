from emocodes import GraphTimeSeries
from pandas import read_csv
from glob import glob
from os.path import basename

files = ['']

for file in files:
    filename = basename(file)[:-20]
    file_df = read_csv(file, index_col=0)
    gt = GraphTimeSeries(code_series_df=file_df)
    gt.graph_all(num_outfile=filename+'_numeric', cat_outfile=filename+'_categorical', file_type='svg')
