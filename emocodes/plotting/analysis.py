import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='white')


def plot_heatmap(data):
    """

    Parameters
    ----------
    data

    Returns
    -------
    fig

    """
    fig = sns.heatmap(data, figsize=(12, 10), center=0, annot=True)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    return fig

def plot_vif(vif_scores):
    """

    Parameters
    ----------
    vif_scores

    Returns
    -------
    fig
    """
    if len(vif_scores) < 8:
        w = len(vif_scores)
    else:
        w = 8

    plt.figure(figsize=(w, 5))
    fig = vif_scores.plot(kind='bar')
    fig.axhline(2, color='blue')
    fig.axhline(5, color='green')
    fig.axhline(10, color='red')
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    return fig
    
