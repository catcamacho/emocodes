import pandas as pd
import pingouin as pg

# TODO: write ICC class
# TODO: write consensus code class


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

    def trainee_consensus(self,trainee_codes_list, trainee_list, exemplar_code_file):
        self.codes_list = trainee_codes_list
        self.raters = trainee_list
        self.original_codes = exemplar_code_file
        self.consensus_scores = compute_exact_match(ratings_list, raters_list, reference=exemplar_code_file)

    def interrater_consensus(self, codes_list, trainee_list):
        self.codes_list = codes_list
        self.raters = trainee_list
        self.consensus_scores = compute_exact_match(ratings_list, raters_list, reference=None)


def compile_ratings(list_dfs, list_raters=None, rater_col_name='rater', index_label='time'):
    """

    Parameters
    ----------
    list_dfs : list
        
    list_raters : list
    rater_col_name : str
    index_label : str

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
    raters_list : list
    reference : str

    Returns
    -------
    extact_match_stats : DataFrame

    '''

    return(extact_match_stats)