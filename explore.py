### Imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Plotting ###

def plot_variable_pairs (df):
    """ 
    Purpose
        Plot pairwise relationships of zillow dataframe utlizing sample size 1000
        Ignores Categorical columns
    
    Parameters
        df: a dataframe containing zillow real estate data

    Returns
        None
    """

    #initialize a seaborn pairgrid
    g = sns.PairGrid(df.drop(columns=['year_built', 'fips']), diag_sharey=False, corner=True)

    #fills out the pairgrid with a regplot
    g.map(sns.regplot, line_kws={'color':'r'})

    return 


def plot_categorical_and_continuous_vars(df):
    """ 
    Purpose
        
    Parameters
        df: 

    Returns
    None
    """

    categorical_col = ['year_built', 'fips']

    continuous_col = df.drop(columns=['year_built', 'fips']).columns

    g = sns.PairGrid(data=df, 
                y_vars=continuous_col,
                x_vars=categorical_col, 
                height=4, 
                aspect=2)
    g.map(sns.swarmplot)
    g.axes[0,0].set_xticks(np.arange(0,104,20));

    g = sns.PairGrid(data=df, 
                y_vars=continuous_col,
                x_vars=categorical_col, 
                height=4, 
                aspect=2)
    g.map(sns.boxplot)
    g.axes[0,0].set_xticks(np.arange(0,104,20));

    g = sns.PairGrid(data=df, 
                y_vars=continuous_col,
                x_vars=categorical_col, 
                height=4, 
                aspect=2)

    g.map(sns.barplot)
    g.axes[0,0].set_xticks(np.arange(0,104,20));
    
    return