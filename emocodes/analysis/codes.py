import pandas as pd
import pingouin as pg
import os
from itertools import combinations


class InterraterReliability:
    """
    This class can be used to compute metrics of interrater reliability from a list of dataframes/codes (with
    identical column names).
    """
    def __init__(self):
        self.list_of_codes = None
        self.list_of_raters = None
        self.long_codes = None
        self.iccs = None
        self.exact = None
        self.rater_col_name = 'rater'

    def df_list_to_long_df(self, list_of_codes, list_of_raters=None):
        """
        This method combines input dataframes in to one long, stacked dataframe, differentiating each by "rater".
        The index is presumed to be time and is preserved.

        Parameters
        ----------
        list_of_codes: list
            list of DataFrame objects OR filepaths to CSVs containing DataFrame objects to stack.
        list_of_raters: list
            Optional. Custom list of rater names/identifiers. If none are entered, defaults to naming as "rater01,
            rater02..." and so on.

        """
        self.list_of_codes = list_of_codes
        if not isinstance(list_of_raters, list):
            list_of_raters = ['rater{0}'.format(str(i).zfill(2)) for i in range(0, len(list_of_codes))]

        self.list_of_raters = list_of_raters
        self.long_codes = compile_ratings(self.list_of_codes, list_raters=self.list_of_raters)
        return self

    def compute_iccs(self, column_labels=None):
        """
        This method computes the intraclass correlation across raters in a dataset.

        Parameters
        ----------
        column_labels: list
            List of string column labels to compute ICCs for. Default is "None", which computes ICCs for all columns
            not the index/time column and "rater".

        """
        self.iccs = interrater_iccs(self.long_codes, self.rater_col_name, column_labels=column_labels)
        return self

    def save_iccs(self, out_file_name):
        """
        This function saves the ICC results table to a CSV.

        Parameters
        ----------
        out_file_name: str
            File path and file name to save the ICC results table.

        """
        self.iccs.to_csv(out_file_name + '.csv')
        print('ICCs saved at {0}.csv'.format(out_file_name))

    def compute_compile_iccs(self, list_of_codes, list_of_raters=None, column_labels=None,
                             out_file_name='interrater_iccs'):
        """
        This method takes a list of dataframes and computes the interrater reliability for each column.

        Parameters
        ----------
        list_of_codes: list
            A list of codes to compute the ICCs of.  List can be of DataFrame objects or of filepaths to CSVs containing
            DataFrame objects.
        list_of_raters: list
            A list of strings to label the individual raters for the codes in the list_of_codes. If None, this function
            creates a list of ['rater01','rater02',..] and so on.
        column_labels: list OR None
            The columns to compute ICCs for.  If None, will compute ICCs for all columns except for the 'time' and
            'rater' columns.
        out_file_name: str
            The filepath and file name to save the ICC results table as.

        """
        self.list_of_codes = list_of_codes
        if not list_of_raters:
            list_of_raters = ['rater{0}'.format(str(i).zfill(2)) for i in range(0, len(list_of_codes))]
        self.list_of_raters = list_of_raters
        self.df_list_to_long_df(self.list_of_codes, list_of_raters=self.list_of_raters)
        self.compute_iccs(column_labels)
        self.save_iccs(out_file_name)
        print(self.iccs)


class Consensus:
    """
    This class can be used to compute the consensus (percent overlap) between two or more sets of codes.

    Use Case 1: compute overlap between trainee codes and exemplar codes

        >>> con = Consensus()
        >>> con.training_consensus([trainee1_codes_df, trainee2_codes_df], original_codes_df, ['Lizzi','Cat'])
        >>> con.consensus_scores.to_csv('consensus_scores.csv') #save scores table as a csv
        >>> con.mismatch_segments.to_csv('mismatched_segments.csv') #save the list of mismatched time segments as a csv

    Use Case 2: compute overlap pairwise between 2 or more raters

        >>> con = Consensus()
        >>> con.interrater_consensus([Lizzi_codes_df, Cat_codes_df], ['Lizzi','Cat'])
        >>> con.consensus_scores.to_csv('consensus_scores.csv') #save scores table as a csv
        >>> con.mismatch_segments.to_csv('mismatched_segments.csv') #save the list of mismatched time segments as a csv

    """
    def __init__(self):
        self.codes_list = None
        self.raters = None
        self.original_codes = None
        self.consensus_scores = None
        self.mismatch_segments = None

    def training_consensus(self, trainee_codes_list, exemplar_code_file, trainee_list=None):
        """
        This method computes consensus ratings for each set of trainee codes against an exemplar/master set. It produces
        a report of the percent overlap between the codes as well as a list of nonmatching segments.

        Parameters
        ----------
        trainee_codes_list: list
            A list of codes.  List can be of DataFrame objects or of filepaths to CSVs containing DataFrame objects.
        exemplar_code_file: filepath OR DataFrame
            The DataFrame to compare each of the trainee codes to. Can be the string filename to a CSV or a DataFrame
            object.
        trainee_list: list
            Optional. A list of strings with rater names to use.  If None, will automatically assign "rater01",
            rater02", etc.

        """
        self.codes_list = trainee_codes_list
        if not trainee_list:
            trainee_list = ['rater{0}'.format(str(i).zfill(2)) for i in range(0, len(trainee_codes_list))]

        self.raters = trainee_list
        self.original_codes = exemplar_code_file
        self.consensus_scores = compute_exact_match(self.codes_list, self.raters, reference=self.original_codes)
        results = []
        for i, df in enumerate(self.codes_list):
            res = mismatch_segments_list(df, self.original_codes, time_column=0)
            res['rater'] = self.raters[i]
            results.append(res)

        self.mismatch_segments = pd.concat(results)
        return self

    def interrater_consensus(self, codes_list, rater_list=None):
        """
        This method compares a list of codes pairwise and produces 1) a measure of overlap for each code and 2) a
        list of timestamps for the mismatched segments.

        Parameters
        ----------
        codes_list: list
            List of dataframe objects OR list of file paths to CSVs containing dataframe objects
        rater_list: list
            Optional.  List of identifiers for the list of codes.

        """
        self.codes_list = codes_list
        if not rater_list:
            rater_list = ['rater{0}'.format(str(i).zfill(2)) for i in range(0, len(codes_list))]

        self.raters = rater_list
        self.consensus_scores = compute_exact_match(self.codes_list, self.raters, reference=None)

        r = range(0, len(self.raters))
        combs = combinations(r, 2)
        results = []
        for c in combs:
            res = mismatch_segments_list(self.codes_list[c[0]], self.codes_list[c[1]])
            res['rater1'] = self.raters[c[0]]
            res['rater2'] = self.raters[c[1]]
            results.append(res)

        self.mismatch_segments = pd.concat(results)
        return self


def compile_ratings(list_dfs, list_raters=None):
    """
    This function takes a list of dataframes (one per rater) and stacks them, preserving the time index.

    Parameters
    ----------
    list_dfs: list
        A list of DataFrames or CSV files containing dataframes.
    list_raters: list
        Default is None. A list of preferred rater names. If none are passed, default is to use "raterXX"
        (e.g., "rater01' for the first dataframe)

    Returns
    -------
    single_df: DataFrame
        A single dataframe of the input dataframes stacked, preserving the index.
    """

    ratings_dfs = []
    for i, df in enumerate(list_dfs):
        if list_raters:
            rater = list_raters[i]
        else:
            rater = 'rater{0}'.format(str(i).zfill(2))

        if not isinstance(df, pd.DataFrame):
            if not os.path.isfile(df):
                raise 'ERROR: ratings_list must be list of either pandas DataFrames OR a list of pandas DataFrames ' \
                      'saved as CSVs. '
            else:
                df = pd.read_csv(df, index_col=0)
                df['rater'] = rater
                ratings_dfs.append(df)
        else:
            df['rater'] = rater
            ratings_dfs.append(df)

    single_df = pd.concat(ratings_dfs)

    return single_df


def interrater_iccs(ratings, rater_col_name='rater', index_label='onset_ms', column_labels=None):
    """
    This function computes the interrater ICCs using the Pingouin library. By default it computes the absolute agreement
    between raters assuming a random sample of raters at each target (each rating at each instance).
    Read more on ICC2 at https://pingouin-stats.org/generated/pingouin.intraclass_corr.html#pingouin.intraclass_corr

    Parameters
    ----------
    index_label: str
        The label denoting each measurement. This must be consistent across all raters. Default is "onset_ms".
    ratings: DataFrame
        DataFrame with the ratings information stored in a long format.
    rater_col_name: str
        The name of the column containing rater information. Default is "rater"
    column_labels: list
        The list of variables to computer inter-rater ICCs for. Default is None, which means it will compute ICCs for
        every column in the DataFrame not equal to the rater_col_name or the index_label.

    Returns
    -------
    icc_df: DataFrame
        The dataframe object containing instance-level and overall intraclass correlation values.

    """
    if not column_labels:
        column_labels = ratings.columns.to_list()
        column_labels.remove(rater_col_name)

    if index_label in column_labels:
        column_labels.remove(index_label)
    else:
        ratings[index_label] = ratings.index
        ratings.index.name = 'index'

    icc_df = pd.DataFrame(columns=['instance_level_ICC', 'instance_level_consistency'])

    for i, x in enumerate(column_labels):
        icc = pg.intraclass_corr(data=ratings, targets=index_label, raters=rater_col_name,
                                 ratings=x, nan_policy='omit')
        icc_df.loc[x, 'instance_level_ICC'] = icc.loc[1, 'ICC']

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
    """
    This function computes the percent overlap between ratings. It can be run with a reference file that all code files
    are compared against, or it can be run without a reference in which case all codes will be compared pair-wise.

    Parameters
    ----------
    ratings_list: list
        List of dataframe objects or CSV filenames of saved dataframes.
    raters_list: list
        List of raters corresponding to each ratings DataFrame in the ratings_list.
    reference: DataFrame or filepath or None
        The DataFrame object or CSV filename of the DataFrame object to compare each DataFrame in ratings_list to.
        If None, this function performs a pair-wise comparison instead.

    Returns
    -------
    exact_match_stats: DataFrame
        A DataFrame with the match statistic for each pair of raters and for each column in the codes.
    """

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

        for i, a in enumerate(ratings_dfs):
            variables = a.columns
            for h, b in enumerate(variables):
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'RatingsA'] = raters_list[i]
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'RatingsB'] = 'reference'
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'ColumnVariable'] = b
                exact_match_stats.loc['{0}_{1}'.format(i, h), 'PercentOverlap'] = (a[b] == reference[b]).mean() * 100
    else:
        variables = ratings_dfs[0].columns
        r = range(0, len(raters_list))
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
    that are nonmatching.  Units are of whatever the index or time variable is. Note that this function only checks
    columns that exist in BOTH dataframes.

    Parameters
    ----------
    df1: DataFrame object OR filepath
        The dataframe to compare to df2. Index must be the time or count variable.
    df2: DataFrame object OR filepath
        The dataframe to compare to df1.  Index must be time or count variable
    time_column: str OR int
        name or index of column to use as the time variable. Default is 0 (first column)

    Returns
    -------
    nonmatching_segments: DataFrame
        A table listing all the segments during which the code in question is not in agreement between the two sets of
        ratings.  Time is in the same units/notation as the index.

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

    nonmatching_segments = pd.DataFrame(columns=['variable', 'mismatch_onset', 'mismatch_offset'])

    # column by column, return onsets and offsets for non-matching segments
    variables = list(set(df1.columns) & set(df2.columns))

    for j, v in enumerate(variables):
        mismatch = (df1[v] != df2[v]).astype(int)
        mismatch = mismatch.to_frame()
        mismatch['time'] = mismatch.index
        mismatch['segment'] = (mismatch[v].diff(1) != 0).astype(int).cumsum()
        res = pd.DataFrame({'start': mismatch.groupby('segment').time.first(),
                            'end': mismatch.groupby('segment').time.last(),
                            'value': mismatch.groupby('segment')[v].mean()}).reset_index(drop=True)
        mismatch = res.loc[res['value'] == 1, :]

        for i, e in enumerate(mismatch.index):
            nonmatching_segments.loc['{0}_{1}'.format(j, i), 'variable'] = v
            nonmatching_segments.loc['{0}_{1}'.format(j, i), 'mismatch_onset'] = mismatch.loc[e, 'start']
            nonmatching_segments.loc['{0}_{1}'.format(j, i), 'mismatch_offset'] = mismatch.loc[e, 'end']

    return nonmatching_segments