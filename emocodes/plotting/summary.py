import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='white')


def plot_heatmap(data, figsize=(12,10), vmin=-0.5, vmax=0.5):
    """

    Parameters
    ----------
    data
    figsize

    Returns
    -------

    """
    ax = sns.heatmap(data, figsize=figsize, center=0, vmin=vmin, vmax=vmax)
    
