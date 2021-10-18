import pandas as pd
import pingouin as pg
import os


class InterraterReliability:
    def __init__(self):
        """
        This class can be used to compute metrics of interrater reliability from a list of dataframes/codes (with
        identical column names).
        """
        self.list_of_codes = None
        self.list_of_raters = None
        self.long_codes = None
        self.iccs = None
        self.exact = None
        self.rater_col_name = None

    def df_list_to_long_df(self, list_of_codes, list_of_raters=None, rater_col_name='rater'):
        """

        Parameters
        ----------
        list_of_codes
        list_of_raters
        rater_col_name

        Returns
        -------

        """
        self.list_of_codes = list_of_codes
        self.list_of_raters = list_of_raters
        self.rater_col_name = rater_col_name
        self.long_codes = compile_ratings(self.list_of_codes, self.list_of_raters, self.rater_col_name)
        return self

    def compute_iccs(self, column_labels):
        """

        Parameters
        ----------
        column_labels

        Returns
        -------

        """
        self.iccs = interrater_iccs(self.long_codes, self.rater_col_name, column_labels)
        return self

    def save_iccs(self, out_file_name):
        """

        Parameters
        ----------
        out_file_name

        Returns
        -------

        """
        self.iccs.to_csv(out_file_name + '.csv')
        print('ICCs saved at {0}.csv')

    def compute_compile_iccs(self, list_of_codes, list_of_raters=None, column_labels=None, rater_col_name='rater',
                             out_file_name='interrater_iccs'):
        """

        Parameters
        ----------
        list_of_codes
        list_of_raters
        column_labels
        out_file_name
        rater_col_name

        Returns
        -------

        """
        self.list_of_codes = list_of_codes
        self.list_of_raters = list_of_raters
        self.rater_col_name = rater_col_name
        self.df_list_to_long_df(self, self.list_of_codes, list_of_raters=self.list_of_raters,
                                rater_col_name=self.rater_col_name)
        self.compute_iccs(self, column_labels)
        self.save_iccs(self, out_file_name)
        print(self.iccs)


class Consensus:
    def __init__(self, threshold=0.9):
        '''

        Parameters
        ----------
        threshold


        '''
        self.codes_list = None
        self.raters = None
        self.original_codes = None
        self.threshold = threshold
        self.consensus_scores = None

    def training_consensus(self,trainee_codes_list, exemplar_code_file, trainee_list=None):
        self.codes_list = trainee_codes_list
        self.raters = trainee_list
        self.original_codes = exemplar_code_file
        self.consensus_scores = compute_exact_match(self.codes_list, self.raters, reference=exemplar_code_file)

    def interrater_consensus(self, codes_list, trainee_list=None):
        self.codes_list = codes_list
        self.raters = trainee_list
        self.consensus_scores = compute_exact_match(self.codes_list, self.raters, reference=None)


def compile_ratings(list_dfs, list_raters=None, rater_col_name='rater', index_label='time'):
    """

    Parameters
    ----------
    list_dfs : list
        A list of DataFrames or CSV files containing dataframes.
    list_raters : list
        Default is None. A list of preferred rater names. If none are passed, default is to use "raterXX"
        (e.g., "rater01' for the first dataframe)
    rater_col_name : str
        Default is 'rater'. Name for the column containing rater information.
    index_label : str
        Default is 'time'.  Name of column with time index information.

    Returns
    -------
    single_df : DataFrame
    """

    dfs = []
    for i, file in enumerate(list_dfs):
        if list_raters:
            rater = list_raters[i]
        else:
            rater = 'rater{0}'.format(i.astype(str).zfill(2))
        temp = pd.load_csv(file, index_col=index_label)
        temp[rater_col_name] = rater
        dfs.append(temp)

    single_df = pd.concat(dfs)

    return single_df


def interrater_iccs(ratings, rater_col_name='rater', index_label='time', column_labels=None):
    """

    Parameters
    ----------
    index_label : str
        The label denoting each measurement. This must be consistent across all raters. Default is "time".
    ratings: DataFrame
        DataFrame with the ratings information stored in a long format.
    rater_col_name : str
        The name of the column containing rater information. Default is "rater"
    column_labels : list
        The list of variables to computer inter-rater ICCs for. Default is None, which means it will compute ICCs for
        every column in the DataFrame not equal to the rater_col_name or the index_label.

    Returns
    -------
    icc_df : DataFrame
        The dataframe object containing instance-level and overall intraclass correlation values.

    """
    icc_df = pd.DataFrame(columns=['instance_level_ICC', 'instance_level_consistency',
                                   'overall_mean_ICC'])
    for x in column_labels:
        icc = pg.intraclass_corr(data=ratings, targets=index_label, raters=rater_col_name,
                                 ratings=x, nan_policy='omit').round(3)
        icc_df.loc[x, 'instance_level_ICC'] = icc.loc[1, 'ICC']
        icc_df.loc[x, 'overall_mean_ICC'] = icc.loc[4, 'ICC']

        # evaluate item-level ICCs
        if icc.loc[1, 'ICC'] < 0.50:
            icc_df.loc[x, 'instance_level_consistency'] = 'poor'
        elif (icc.loc[1, 'ICC'] >= 0.50) & (icc.loc[1, 'ICC'] < 0.75):
            icc_df.loc[x, 'instance_level_consistency'] = 'moderate'
        elif (icc.loc[1, 'ICC'] >= 0.75) & (icc.loc[1, 'ICC'] < 0.90):
            icc_df.loc[x, 'instance_level_consistency'] = 'good'
        elif icc.loc[1, 'ICC'] >= 0.90:
            icc_df.loc[x, 'instance_level_consistency'] = 'excellent'

    return icc_df


def compute_exact_match(ratings_list, raters_list, reference):
    '''

    Parameters
    ----------
    ratings_list : list
        List of dataframe objects or CSV filenames of saved dataframes.
    raters_list : list
        List of raters corresponding to each ratings DataFrame in the ratings_list.
    reference : DataFrame or filepath or None
        The DataFrame object or CSV filename of the DataFrame object to compare each DataFrame in ratings_list to.
        If None, this function performs a pair-wise comparison instead.

    Returns
    -------
    extact_match_stats : DataFrame
        A DataFrame with the match statistic for each pair of raters and for each column in the codes.
    '''

    exact_match_stats = pd.DataFrame(columns=['RatingsA', 'RatingsB', 'ColumnVariable', 'PercentOverlap'])
    if not isinstance(ratings_list[0], pd.DataFrame):
        if not os.path.isfile(ratings_list[0]):
            raise 'ERROR: ratings_list must be list of either pandas DataFrames OR a list of pandas DataFrames saved as CSVs.'
        else:
            ratings_dfs = []
            for a in ratings_list:
                df = pd.read_csv(a, index_col=0)
                ratings_dfs.append(df)
    else:
        ratings_dfs = ratings_list

    if reference:
        if not isinstance(reference, pd.DataFrame):
            if not os.path.isfile(reference):
                raise 'ERROR: reference file must be DataFrame, filepath, or None.'
            else:
                reference = pd.read_csv(reference, index_col=0)

        variables = reference.columns
        for i, a in enumerate(ratings_dfs):
            variables = a.columns
            for h, b in enumerate(variables):
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'RatingsA'] = raters_list[i]
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'RatingsB'] = 'reference'
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'ColumnVariable'] = b
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'PercentOverlap'] = (a[b] == reference[b]).mean() * 100
    else:
        from itertools import combinations
        variables = ratings_dfs[0].columns
        r = range(0,len(raters_list))
        combs = combinations(r,2)
        for i, a in enumerate(combs):
            for h, b in enumerate(variables):
                df1 = ratings_list[a[0]]
                df2 = ratings_list[a[1]]
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'RatingsA'] = raters_list[a[0]]
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'RatingsB'] = raters_list[a[1]]
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'ColumnVariable'] = b
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'PercentOverlap'] = (df1[b] == df2[b]).mean() * 100
    return exact_match_stats

def mismatch_segments_list(df1, df2, time_column=0):
    """
    This function compares two columns of the same name across two input dataframes and returns a dataframe of segments
    that are nonmatching.  Units are of whatever the index or time variable is.

    Parameters
    ----------
    df1 : DataFrame object OR filepath
        The dataframe to compare to df2. Index must be the time or count variable.
    df2 : DataFrame object OR filepath
        The dataframe to compare to df1.  Index must be time or count variable
    time_column : str OR int
        name or index of column to use as the time variable. Default is 0 (first column)

    Returns
    -------
    nonmatching_segments

    """

    if not isinstance(df1, pd.DataFrame):
        if not os.path.isfile(df1):
            raise 'ERROR: df1 must be list of either pandas DataFrames OR a list of pandas DataFrames saved as CSVs.'
        else:
            df1 = pd.read_csv(df1, index_col=time_column)

    if not isinstance(df2, pd.DataFrame):
        if not os.path.isfile(df2):
            raise 'ERROR: df2 must be list of either pandas DataFrames OR a list of pandas DataFrames saved as CSVs.'
        else:
            df2 = pd.read_csv(df2, index_col=time_column)

    nonmatching_segments = pd.DataFrame(columns=['variable','mismatch_onset','mismatch_offset'])
    # column by column, return onsets and offsets for non-matching segments
    variables = list(set(df1.columns) & set(df2.columns))
    gap = df1.index[1] - df1.index[0]
    i = 0
    for j, v in enumerate(variables):
        mismatch = df1.index[df1[v] != df2[v]]
        for k, a in enumerate(mismatch):
            # find onsets
            if k==0:
                onset = k
            while mismatch[k+1] - mismatch[k] == gap:
                pass

    return nonmatching_segments