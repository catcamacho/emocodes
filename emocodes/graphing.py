import matplotlib.pyplot as plt

# TODO: make function to graph values
# TODO: add documentation


class GraphTimeSeries:
    
    def __init__(self, code_series_df):
        self.codes_series_df = code_series_df
        self.numeric_cols = None
        self.categorical_cols = None

    def graph_all(self, num_outfile='numerical_plots', cat_outfile='categorical_plots', file_type='png'):
        """Fully plot all variables in the dataframe"""
        self.sort_column_types()
        self.plot_numeric(num_outfile, file_type)
        self.plot_categorical(cat_outfile, file_type)
        return self

    def sort_column_types(self):
        """Sort columns by their data type for graphing"""
        df = self.codes_series_df
        num_vars = []
        cat_vars = []
        for x in df.columns:
            if df[x].dtype == float or df[x].dtype == int:
                num_vars.append(x)
            else:
                cat_vars.append(x)
        self.numeric_cols = num_vars
        self.categorical_cols = cat_vars
        return self

    def plot_numeric(self, outfile_name='numerical_plots', file_type='png'):
        """Save time plots of all numeric series."""
        make_num_plot(self.numeric_cols, self.codes_series_df, outfile_name, file_type)

    def plot_categorical(self, outfile_name='categorical_plots', file_type='png'):
        """Save pie plots of all categorical series."""
        make_cat_plot(self.categorical_cols, self.codes_series_df, outfile_name, file_type)


def make_cat_plot(cat_col_names, codes_df, outfile_name='categorical_plots', file_type='png'):
    """
    This function takes the categorical data within a data frame and plots separate pie graphs for each.

    Parameters
    ----------
    :param outfile_name: str
        The name of the out file name for the plots

    :param cat_col_names: list
        The list of columns for categorical data to be graphed

    :param codes_df: DataFrame
        The dataframe with the code data

    :param file_type: str
        Default is 'png'. A string denoting the type of image to save the plot as. Must be 'svg','png', 'pdf', or 'eps'

    :return: None
    """
    df = codes_df.loc[:, cat_col_names]
    fig_height = 4 * len(cat_col_names)

    if len(cat_col_names)
    fig, ax = plt.subplots(len(cat_col_names), figsize=(4, fig_height))
    for i, x in enumerate(cat_col_names):
        df[x].value_counts().plot(kind='pie', ax=ax[i], xlabel=' ', ylabel=' ').set_title(x)
    plt.tight_layout()
    plt.savefig('{0}.{1}'.format(outfile_name, file_type))


def make_num_plot(num_col_names, codes_df, outfile_name='numerical_plots', file_type='png'):
    """
    This function takes the numeric data within a data frame and plots separate line graphs for each.

    Parameters
    ---------
    :param outfile_name: str
        The name of the file to save the image plot as.

    :param num_col_names: list
        The list of columns for numeric data to be graphed

    :param codes_df: DataFrame
        The dataframe with the code data

    :param file_type: str
        Default is 'png'. A string denoting the type of image to save the plot as. Must be 'svg','png', 'pdf', or 'eps'

    :returns: None
    --------
    """

    df = codes_df.loc[:, num_col_names]
    df.resample('1s')
    fig_height = 1.5 * len(num_col_names)
    plt.figure()
    df.plot.line(figsize=(12, fig_height), title=num_col_names, subplots=True, legend=False, xlim=(0, df.index[-1]))
    plt.tight_layout()
    plt.savefig('{0}.{1}'.format(outfile_name, file_type))
