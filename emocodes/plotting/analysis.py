import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='white')


def plot_heatmap(data):
    """
    This function plots a heatmap.

    Parameters
    ----------
    data: DataFrame
        NxN dataframe to plot.

    Returns
    -------
    fig: object
        matplotlib figure object of the plot

    """
    plt.figure(figsize=(6, 5.5))
    fig = sns.heatmap(data, center=0, annot=True)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30, ha='right')
    fig.set_yticklabels(fig.get_yticklabels(), rotation=90, ha='right')
    plt.tight_layout()
    return fig


def plot_vif(vif_scores):
    """
    This function plots variance inflation factor scores with the horizontal lines denoting the standard cut offs:

     - <2 = not collinear
     - 2-5 = weakly collinear and likely okay to include together in a model
     - 5-10 = moderately collinear, proceed with caution
     - >10 = highly collinear, do not include together in a multiple linear regression model

    Parameters
    ----------
    vif_scores: Series
        VIF scores to plot.

    Returns
    -------
    fig: object
        matplotlib figure object of the plot
    """
    if len(vif_scores) < 6:
        w = len(vif_scores)
    else:
        w = 6

    plt.figure(figsize=(w, 4))
    fig = vif_scores.plot(kind='bar')
    fig.axhline(2, color='green')
    fig.axhline(5, color='orange')
    fig.axhline(10, color='red')
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    return fig
    
